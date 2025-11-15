import os
import json
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # 필요 없으면 나중에 삭제해도 됨

from Moderation_API import (
    get_guard_resources,
    classify_prompt as guard_classify_internal,
)

# =========================
# 0) Hugging Face 토큰 설정
# =========================
HF_TOKEN = "발급한 토큰으로 바꾸기"  # ⚠ 깃에 올릴 땐 빼기!

os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HF_HUB_TOKEN"] = HF_TOKEN
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN


# =========================
# 1) 왜곡 판별기 (다른 사람이 만든 모델을 붙이는 자리)
# =========================
"""
여기서는 '정상 / 비정상'을 판별하는 분류기를 로딩하고,
입력 프롬프트를 넣어 라벨과 신뢰도를 돌려주는 인터페이스만 정의한다.

다른 사람이 만든 분류기 구조에 맞게
- load_cls_model()
- classify_prompt()
두 함수만 구현하면, 아래 생성 파이프라인은 그대로 쓸 수 있다.
"""


def load_cls_model():
    """
    우리 쪽 가드 모델(Moderation_API.py)을 로딩하는 부분.

    - get_guard_resources()를 통해
      (nli_clf, fact_texts, fact_embs)를 한 번만 메모리에 올린다.
    - 이 함수는 (tokenizer, model) 튜플을 리턴해야 하므로,
      편의상 tokenizer 자리에 nli_clf, model 자리에 (fact_texts, fact_embs)를 넣어둔다.
    """
    nli_clf, fact_texts, fact_embs = get_guard_resources()

    # 아래 튜플은 main 쪽에서 cls_tokenizer, cls_model 변수로 받게 됨
    # cls_tokenizer → nli_clf
    # cls_model     → (fact_texts, fact_embs)
    return nli_clf, (fact_texts, fact_embs)


def classify_prompt(prompt: str, cls_tokenizer, cls_model):
    """
    우리 가드 모델의 classify_prompt()를 감싸서
    - label: "정상" / "비정상"
    - confidence: float
    형태로 반환하는 어댑터 역할을 한다.

    cls_tokenizer: load_cls_model()에서 넘긴 nli_clf
    cls_model    : load_cls_model()에서 넘긴 (fact_texts, fact_embs)
    """

    nli_clf = cls_tokenizer
    fact_texts, fact_embs = cls_model

    # Moderation_API.classify_prompt() 호출
    res = guard_classify_internal(
        prompt,
        nli_clf=nli_clf,
        fact_texts=fact_texts,
        fact_embs=fact_embs,
    )

    # harmful 여부를 distortion 라벨로 매핑
    if res.get("in_domain", False) and res.get("harmful", False):
        label = "비정상"
    else:
        # 도메인 밖이거나 harmful=False 인 경우 모두 "정상"으로 취급
        label = "정상"

    # confidence는 CONTRADICT 확률 그대로 사용 (0~1)
    confidence = float(res.get("max_contradict_prob", 0.0))

    return label, confidence


# =========================
# 2) 생성기 (Llama – 질문 교정 전용)
# =========================

GEN_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

gen_pipe = transformers.pipeline(
    "text-generation",
    model=GEN_MODEL_ID,
    model_kwargs={"dtype": torch.bfloat16},  # torch_dtype 대신 dtype
    device_map="auto",
    token=HF_TOKEN,
)

SYSTEM_PROMPT = (
    "당신은 한국 역사, 문화, 영토·전쟁·인권 문제를 포함한 다양한 주제에 대해, "
    "한국어로 작성된 질문을 다듬는 '질문 교정기'입니다.\n\n"
    "당신의 역할은 절대로 질문에 대한 '답변'을 생성하지 말고, 오직 '질문'만 다시 작성하는 것입니다.\n"
    "가능하면 입력으로 주어진 원래 질문의 말투와 종결체(반말, 존댓말, 해요체, ~냐?, ~니? 등)를 최대한 유지하세요.\n"
    "다만, 원래 질문이 지나치게 공격적이거나 모욕적인 표현을 포함하는 경우에는, "
    "전반적인 문체는 유지하되 표현 수위를 낮추고 순화된 표현으로 바꾸십시오.\n\n"
    "입력으로 distortion 값과 원래 질문(prompt)이 주어집니다.\n"
    "distortion이 '정상'인 경우:\n"
    "  - 사용자의 질문 의미와 범위를 그대로 유지하면서, 문장을 더 자연스럽고 명확하게 다듬으세요.\n"
    "  - 불필요하게 새로운 논점이나 주제를 추가하지 말고, 사용자가 실제로 묻고 있는 내용만 다루도록 하세요.\n\n"
    "distortion이 '비정상'인 경우:\n"
    "  - 역사·사실 왜곡, 식민지 지배나 전쟁 범죄의 미화, 피해자 비하, 혐오·폭력 표현이 있다면,\n"
    "    그 부분만 보다 사실에 가까운 표현과 중립적인 어휘로 수정하세요.\n"
    "  - 그래도 사용자가 궁금해하는 핵심 질문(무엇을 알고 싶어 하는지)은 유지해야 하며, "
    "    질문의 범위를 불필요하게 넓히거나 새로운 의제를 추가하지 마세요.\n"
    "  - 특정 국가나 집단 전체를 감정적으로 비난하기보다는, '어떤 입장이나 논쟁이 있다'라는 식으로 "
    "    질문을 표현하도록 조정하세요.\n\n"
    "중요:\n"
    "  - 당신은 절대 질문에 대한 '답변'을 작성하면 안 됩니다.\n"
    "  - 오직 교정된 질문 하나만 생성하고, 그 질문을 JSON 형식의 'rewritten_prompt' 필드에 담아 출력하십시오.\n\n"
    "출력 형식 (반드시 이 형식만 사용):\n"
    "{\"rewritten_prompt\": \"...\"}"
)


def generate_rewritten_prompt(prompt: str, distortion: str) -> str:
    """
    prompt    : 원본 프롬프트
    distortion: '정상' 또는 '비정상'
    → LLM이 왜곡 교정이 반영된 '교정된 질문'만 생성
    """

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "다음은 사용자의 원래 질문입니다.\n"
                "distortion 값에 따라 위 지침을 적용하여, 질문을 교정해 주세요.\n\n"
                f'distortion: "{distortion}"\n'
                f'prompt: "{prompt}"\n\n'
                "질문에 대한 답변은 절대 작성하지 말고, 교정된 질문 하나만 JSON 형식으로 돌려주세요.\n"
                "형식 예시: {\"rewritten_prompt\": \"...\"}"
            ),
        },
    ]

    terminators = [
        gen_pipe.tokenizer.eos_token_id,
        gen_pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    outputs = gen_pipe(
        messages,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=False,
        temperature=0.0,
    )

    assistant_msg = outputs[0]["generated_text"][-1]["content"]

    # JSON 부분만 파싱
    start = assistant_msg.find("{")
    end = assistant_msg.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"JSON을 찾을 수 없습니다:\n{assistant_msg}")

    json_str = assistant_msg[start : end + 1]
    data = json.loads(json_str)

    rewritten = data.get("rewritten_prompt", "").strip()
    return rewritten


# =========================
# 3) 엔드 투 엔드 파이프라인 (질문 교정만)
# =========================

if __name__ == "__main__":
    # 🔧 다른 사람이 만든 분류기 로딩 부분
    cls_tokenizer, cls_model = load_cls_model()
    print("Device set to use", gen_pipe.device)

    while True:
        user_prompt = input("\n프롬프트를 입력하세요 (종료: q): ").strip()

        if user_prompt.lower() == "q":
            print("종료합니다.")
            break

        if user_prompt == "":
            print("⚠ 빈 프롬프트입니다. 다시 입력해 주세요.")
            continue

        # 1) 왜곡 여부 판별
        distortion_label, conf = classify_prompt(user_prompt, cls_tokenizer, cls_model)
        print(f"\n[분류 결과] {distortion_label} (신뢰도: {conf:.3f})")

        print("→ LLM이 교정된 질문을 생성하는 중입니다...")

        # 2) 교정된 질문 생성 (답변 X)
        rewritten = generate_rewritten_prompt(user_prompt, distortion_label)

        print("\n=== 교정된 질문 ===")
        print(rewritten)
