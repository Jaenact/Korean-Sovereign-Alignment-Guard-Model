# Moderation_API.py
# 한국 현대사/영토/일제강점기 인권 문제에 대한 왜곡 여부를 판단하는 가드 모델
# - NLI(팩트와의 관계) + Intent(발화 의도) 분리
# - 최종 판정: CONTRADICT ∧ ASSERTION 일 때만 HARMFUL

import json  # JSON 파일 처리를 위해 import
from pathlib import Path  # 경로 관리를 위해 import
from typing import List, Dict, Tuple  # 타입 힌트를 위해 import

import numpy as np  # 수치 계산을 위해 import
from sklearn.linear_model import LogisticRegression  # 로지스틱 회귀 분류기를 위해 import
from sklearn.metrics import classification_report, confusion_matrix  # 평가 지표 출력을 위해 import
from sklearn.model_selection import train_test_split  # train/test 분할을 위해 import
from joblib import dump, load  # 학습된 모델 저장/로드를 위해 import

from openai import OpenAI  # OpenAI API 사용을 위해 import

# -----------------------------------
# 0. OpenAI 클라이언트 / 전역 설정
# -----------------------------------

# ⚠️ API 키는 환경변수 OPENAI_API_KEY 에 설정해두는 것을 권장
client = OpenAI(api_key="API 키로 변경")  # OpenAI 클라이언트 객체 생성

EMBEDDING_MODEL = "text-embedding-3-small"  # 사용할 임베딩 모델 이름

# 데이터/모델 파일 경로 (모두 Moderation_API.py와 같은 폴더 기준)
BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "./augmented_dataset.json"            # harmful/benign/ideal_answer 데이터셋
NLI_EMB_CACHE_PATH = BASE_DIR / "nli_pair_embeddings.npy"     # NLI 학습용 임베딩 캐시
NLI_LABEL_CACHE_PATH = BASE_DIR / "nli_pair_labels.npy"       # NLI 학습용 라벨 캐시
NLI_MODEL_PATH = BASE_DIR / "nli_contradict_classifier.joblib"  # NLI 분류기

FACT_POSITIONS_PATH = BASE_DIR / "korea_fact_positions.json"  # 대한민국 입장 팩트 데이터셋
FACT_TEXTS_PATH = BASE_DIR / "./fact.json"                # fact_text만 모아둔 파일
FACT_EMB_CACHE_PATH = BASE_DIR / "fact_embeddings.npy"        # fact_text 임베딩 캐시

# 도메인 키워드 리스트 (영토/고대사/전쟁/일제 인권 문제 관련 키워드)
DOMAIN_KEYWORDS = [
    "고구려", "발해", "한사군", "삼국시대", "고대사",
    "독도", "다케시마", "동해", "일본해",
    "이어도",
    "NLL", "북방한계선", "서해",
    "일제", "식민지", "식민 통치", "강제징용", "강제 징용", "강제동원", "강제 동원",
    "위안부", "강제 연행",
    "6·25", "6.25", "한국전쟁", "한국 전쟁",
    "분단", "휴전선", "DMZ", "동북공정"
]

# -----------------------------------
# 1. OpenAI 임베딩 함수
# -----------------------------------

def get_embedding(text: str) -> np.ndarray:
    """주어진 텍스트에 대해 OpenAI 임베딩 벡터를 반환하는 함수"""
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,  # 사용할 임베딩 모델 지정
        input=text,             # 임베딩할 텍스트 입력
    )
    vec = np.array(resp.data[0].embedding, dtype=np.float32)  # 임베딩 벡터를 numpy 배열로 변환
    return vec  # 임베딩 벡터 반환


def embed_texts(texts: List[str]) -> np.ndarray:
    """여러 개의 텍스트 리스트를 받아 (N, d) 임베딩 배열로 변환하는 함수"""
    embs = []  # 임베딩 벡터를 누적할 리스트
    total = len(texts)  # 전체 텍스트 개수
    for i, t in enumerate(texts):
        if (i + 1) % 50 == 0 or i == total - 1:
            print(f"[임베딩] {i+1}/{total} 개 처리 완료")
        e = get_embedding(t)  # 개별 텍스트 임베딩 계산
        embs.append(e)
    embs_arr = np.vstack(embs)  # 리스트를 (N, d) 배열로 변환
    return embs_arr

# -----------------------------------
# 2. augmented_dataset 로드
# -----------------------------------

def load_augmented_dataset(path: Path = DATASET_PATH) -> List[Dict]:
    """augmented_dataset.json 파일을 로드하여 리스트로 반환하는 함수"""
    if not path.exists():
        raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data

# -----------------------------------
# 3. NLI 학습을 위한 (문장+팩트) 페어 생성
# -----------------------------------

def build_nli_pairs(dataset: List[Dict]) -> Tuple[List[str], np.ndarray]:
    """
    NLI 학습을 위해 (문장+팩트 페어, 라벨) 리스트를 생성
    - harmful_prompt + ideal_answer -> CONTRADICT (1)
    - benign_prompt  + ideal_answer -> NOT_CONTRADICT (0)
    """
    pair_texts: List[str] = []
    labels: List[int] = []

    for item in dataset:
        harmful = item["harmful_prompt"]
        benign = item["benign_prompt"]
        fact = item["ideal_answer"]

        # harmful → CONTRADICT
        pair_h = f"[문장]: {harmful}\n[기준 설명]: {fact}"
        pair_texts.append(pair_h)
        labels.append(1)

        # benign → NOT_CONTRADICT
        pair_b = f"[문장]: {benign}\n[기준 설명]: {fact}"
        pair_texts.append(pair_b)
        labels.append(0)

    labels_arr = np.array(labels, dtype=int)
    return pair_texts, labels_arr

# -----------------------------------
# 4. NLI 임베딩/라벨 캐시 생성/로드
# -----------------------------------

def prepare_nli_embeddings_and_labels(
    dataset_path: Path = DATASET_PATH,
    emb_cache_path: Path = NLI_EMB_CACHE_PATH,
    label_cache_path: Path = NLI_LABEL_CACHE_PATH,
) -> Tuple[np.ndarray, np.ndarray]:
    """NLI 학습용 페어 임베딩과 라벨을 준비하고 캐시를 활용하는 함수"""

    dataset = load_augmented_dataset(dataset_path)

    if emb_cache_path.exists() and label_cache_path.exists():
        print("[NLI] 캐시 존재, 바로 로드합니다.")
        X_all = np.load(emb_cache_path)
        y_all = np.load(label_cache_path)

        # 데이터셋 크기와 캐시가 맞는지 확인
        pair_texts, labels = build_nli_pairs(dataset)
        if X_all.shape[0] != len(pair_texts) or y_all.shape[0] != len(pair_texts):
            print("⚠️ NLI 캐시 크기가 데이터셋과 불일치. 캐시를 재생성합니다.")
            X_all = embed_texts(pair_texts)
            y_all = labels
            np.save(emb_cache_path, X_all)
            np.save(label_cache_path, y_all)
    else:
        print("[NLI] 캐시가 없어 새로 임베딩을 계산합니다.")
        pair_texts, labels = build_nli_pairs(dataset)
        X_all = embed_texts(pair_texts)
        y_all = labels
        np.save(emb_cache_path, X_all)
        np.save(label_cache_path, y_all)
        print(f"[NLI] 임베딩 캐시 저장: {emb_cache_path}")
        print(f"[NLI] 라벨 캐시 저장: {label_cache_path}")

    return X_all, y_all

# -----------------------------------
# 5. NLI 분류기 학습 (한 번만 실행)
# -----------------------------------

def train_nli_classifier(
    dataset_path: Path = DATASET_PATH,
    model_output_path: Path = NLI_MODEL_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """NLI 분류기(문장+팩트 → CONTRADICT 여부)를 학습하는 함수 (한 번만 실행)"""

    X_all, y_all = prepare_nli_embeddings_and_labels(
        dataset_path=dataset_path,
        emb_cache_path=NLI_EMB_CACHE_PATH,
        label_cache_path=NLI_LABEL_CACHE_PATH,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_all,
        y_all,
        test_size=test_size,
        random_state=random_state,
        stratify=y_all,
    )

    print(f"[NLI] Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("=== NLI Classification Report (CONTRADICT vs NOT) ===")
    print(classification_report(y_test, y_pred, digits=4))
    print("=== NLI Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    dump(clf, model_output_path)
    print(f"[NLI] 분류기 저장 완료: {model_output_path}")

# -----------------------------------
# 6. 대한민국 입장 팩트 인덱스 구축 (korea_fact_positions.json → fact_texts.json + 임베딩)
# -----------------------------------

def build_fact_index_from_positions(
    fact_positions_path: Path = FACT_POSITIONS_PATH,
    fact_texts_path: Path = FACT_TEXTS_PATH,
    fact_emb_path: Path = FACT_EMB_CACHE_PATH,
):
    """
    korea_fact_positions.json 에서 fact_text만 뽑아서
    fact_texts.json + fact_embeddings.npy 를 생성하는 함수 (한 번만 실행)
    """
    if not fact_positions_path.exists():
        raise FileNotFoundError(f"팩트 데이터셋 파일을 찾을 수 없습니다: {fact_positions_path}")

    data = json.loads(fact_positions_path.read_text(encoding="utf-8"))
    fact_texts = [item["fact_text"] for item in data]

    fact_texts_path.write_text(
        json.dumps(fact_texts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[FACT] fact_texts.json 저장: {fact_texts_path}")

    fact_embs = embed_texts(fact_texts)
    np.save(fact_emb_path, fact_embs)
    print(f"[FACT] fact_embeddings.npy 저장: {fact_emb_path}")

# -----------------------------------
# 7. Intent 분류기 (규칙 기반)
# -----------------------------------

def classify_intent(prompt: str) -> str:
    text = prompt.strip()

    # 1) 왜곡 주장을 인용/질문 → ASK_ABOUT_DISTORTION (그대로 유지)
    distortion_keywords = [
        "일부에서는", "주장이 있다", "라는 주장이 있다",
        "라고 주장", "주장의 근거", "이 주장의 근거",
        "이 주장이 왜", "이 주장이 맞는지", "이 주장은",
    ]
    if any(kw in text for kw in distortion_keywords) and "?" in text:
        return "ASK_ABOUT_DISTORTION"

    # 2) 전제형 의문 → ASSERTION (이미 추가해둔 패턴 유지)
    assertion_question_patterns = [
        "아니었어?", "아니야?", "맞지?", "맞는 거 아냐?", "맞는거 아냐?",
        "거 아냐?", "거 아니야?", "된 거 아냐?", "된거 아냐?",
        "아닌가?", "아닌 거 아냐?", "아닌거 아냐?", "않나?",
        "맞는 거지?", "맞는거지?",
        "잖아?", "잖아요?", "잖습니까?",
    ]
    if any(pat in text for pat in assertion_question_patterns):
        return "ASSERTION"

    # 3) 서술형 단정(문어체 + 구어체) → ASSERTION
    declarative_suffixes = [
        "다", "이다", "였다", "했다", "하였다", "이었다",
        "없었다", "있었다",
        "야", "이야", "거야", "거다", "것이다",
        "입니다", "입니다.", "라고 본다", "라고 생각한다", "라고 할 수 있다",
    ]

    if "?" not in text:
        # 끝이 위 접미사들 중 하나면 평서문으로 간주
        if any(text.endswith(suf) for suf in declarative_suffixes):
            return "ASSERTION"
        # 온점 포함 버전도 한번 더 체크
        if any(text.endswith(suf + ".") for suf in declarative_suffixes):
            return "ASSERTION"

        # 물음표도 없고 길이가 어느 정도 이상이면 기본적으로 단정문 취급
        if len(text) >= 8:
            return "ASSERTION"

    # 4) 일반 질문 → ASK_INFO
    if text.endswith("?") or "왜" in text or "어떻게" in text:
        return "ASK_INFO"

    # 5) 그 외
    return "OTHER"


# -----------------------------------
# 8. 도메인 필터
# -----------------------------------

def is_in_domain(prompt: str) -> bool:
    """프롬프트가 우리가 다루는 도메인(영토/고대사/전쟁/일제 인권)에 속하는지 여부를 판정"""
    text = prompt.strip()
    return any(kw in text for kw in DOMAIN_KEYWORDS)

# -----------------------------------
# 9. NLI 분류기 로드 및 예측
# -----------------------------------

def load_nli_classifier(model_path: Path = NLI_MODEL_PATH):
    """저장된 NLI 분류기 모델을 로드"""
    if not model_path.exists():
        raise FileNotFoundError(f"NLI 모델 파일을 찾을 수 없습니다: {model_path}")
    clf = load(model_path)
    return clf

def predict_contradict_prob(clf, prompt: str, fact_text: str) -> float:
    """
    (프롬프트, 기준 설명) 페어에 대해 CONTRADICT(1)일 확률을 반환
    """
    pair_text = f"[문장]: {prompt}\n[기준 설명]: {fact_text}"
    emb = get_embedding(pair_text).reshape(1, -1)
    prob_contradict = clf.predict_proba(emb)[0, 1]
    return float(prob_contradict)

# -----------------------------------
# 10. 기준 팩트 로드 + 검색 (RAG)
# -----------------------------------

def load_fact_index(
    fact_texts_path: Path = FACT_TEXTS_PATH,
    fact_emb_path: Path = FACT_EMB_CACHE_PATH,
) -> Tuple[List[str], np.ndarray]:
    """fact_texts.json + fact_embeddings.npy 로드"""
    if not fact_texts_path.exists():
        raise FileNotFoundError(f"fact_texts.json 을 찾을 수 없습니다: {fact_texts_path}")
    if not fact_emb_path.exists():
        raise FileNotFoundError(f"fact_embeddings.npy 를 찾을 수 없습니다: {fact_emb_path}")

    fact_texts = json.loads(fact_texts_path.read_text(encoding="utf-8"))
    fact_embs = np.load(fact_emb_path)
    return fact_texts, fact_embs

def search_related_facts(
    prompt: str,
    fact_texts: List[str],
    fact_embs: np.ndarray,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """
    프롬프트와 가장 관련 있는 기준 팩트 top_k개를 코사인 유사도로 검색
    반환: [(fact_text, similarity), ...]
    """
    q_emb = get_embedding(prompt)
    q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-8)
    f_norm = fact_embs / (np.linalg.norm(fact_embs, axis=1, keepdims=True) + 1e-8)
    sims = np.dot(f_norm, q_norm)  # (N,)

    top_idx = np.argsort(-sims)[:top_k]
    results: List[Tuple[str, float]] = []
    for idx in top_idx:
        results.append((fact_texts[idx], float(sims[idx])))
    return results

# -----------------------------------
# 11. 최종 판정 (CONTRADICT ∧ ASSERTION → HARMFUL)
# -----------------------------------

def classify_prompt(prompt: str,
                    nli_clf=None,
                    fact_texts: List[str] = None,
                    fact_embs: np.ndarray = None,
                    contradict_threshold: float = 0.6,
                    top_k_facts: int = 5) -> Dict:
    """
    최종적으로 프롬프트가 '막아야 할 왜곡 발화인지'를 판정하는 함수

    로직:
    1) 도메인 밖이면 → SAFE (OUT_OF_DOMAIN)
    2) intent 분류
    3) 관련 팩트 top-k 검색
    4) 각 (prompt, fact)에 대해 CONTRADICT 확률 계산
    5) max(CONTRADICT 확률) >= threshold 이고, intent == ASSERTION 이면 → HARMFUL
       그 외 → SAFE
    """

    result = {
        "prompt": prompt,
        "in_domain": False,
        "intent": None,
        "max_contradict_prob": 0.0,
        "harmful": False,
        "reason": "",
    }

    # 1) 도메인 판정
    if not is_in_domain(prompt):
        result["in_domain"] = False
        result["harmful"] = False
        result["reason"] = "도메인(영토/고대사/전쟁/일제 인권) 밖의 발화로 판단됨."
        return result

    result["in_domain"] = True

    # 2) Intent 분류
    intent = classify_intent(prompt)
    result["intent"] = intent

    # 3) NLI 분류기 & 팩트 인덱스 준비
    if nli_clf is None:
        nli_clf = load_nli_classifier()
    if fact_texts is None or fact_embs is None:
        fact_texts, fact_embs = load_fact_index()

    # 4) 관련 팩트 검색
    related_facts = search_related_facts(
        prompt,
        fact_texts,
        fact_embs,
        top_k=top_k_facts,
    )

    # 5) CONTRADICT 확률 최대값 계산
    max_prob = 0.0
    for fact_text, sim in related_facts:
        prob = predict_contradict_prob(nli_clf, prompt, fact_text)
        if prob > max_prob:
            max_prob = prob

    result["max_contradict_prob"] = max_prob

    # 6) 최종 규칙: CONTRADICT ∧ ASSERTION → HARMFUL
    if max_prob >= contradict_threshold and intent == "ASSERTION":
        result["harmful"] = True
        result["reason"] = (
            "도메인 내 발화이며, 기준 설명과 모순될 가능성이 크고 "
            "발화 의도가 ASSERTION(왜곡 주장을 사실처럼 단정/전제)으로 판단됨."
        )
    else:
        result["harmful"] = False
        if max_prob < contradict_threshold:
            result["reason"] = (
                "도메인 내 발화지만, 기준 설명과의 모순(CONTRADICT) 확률이 임계값보다 낮음."
            )
        else:
            result["reason"] = (
                "기준 설명과 모순될 여지는 있으나, 발화 의도가 ASSERTION이 아니므로 "
                "왜곡 주장을 사실처럼 단정한다고 보기는 어려움."
            )

    return result

# -----------------------------------
# 12. 실행 시에 쓸 편의 함수 (캐시 로드)
# -----------------------------------

_nli_clf = None
_fact_texts = None
_fact_embs = None

def get_guard_resources():
    """NLI 모델과 팩트 인덱스를 최초 1회만 로드하고 캐시로 재사용"""
    global _nli_clf, _fact_texts, _fact_embs

    if _nli_clf is None:
        _nli_clf = load_nli_classifier()
    if _fact_texts is None or _fact_embs is None:
        _fact_texts, _fact_embs = load_fact_index()

    return _nli_clf, _fact_texts, _fact_embs

def guard_classify(prompt: str) -> Dict:
    """외부에서 사용할 최종 가드 함수"""
    nli_clf, fact_texts, fact_embs = get_guard_resources()
    res = classify_prompt(
        prompt,
        nli_clf=nli_clf,
        fact_texts=fact_texts,
        fact_embs=fact_embs,
    )
    return res

# -----------------------------------
# 13. 단독 실행 시 간단 테스트 (학습 X, 분류만)
# -----------------------------------

if __name__ == "__main__":
    # ⚠️ 여기서는 학습하지 않고, 이미 학습/임베딩된 파일을 사용해서 분류만 테스트
    test_prompts = [
        "고구려는 중국의 소수민족이 세운 지방 정권 아니었어?",
        "일부에서는 고구려를 중국 지방 정권으로 보자는 주장이 있다는데, 이 주장의 근거는 뭐야?",
        "대한민국에 대해 소개해줘.",
    ]

    for q in test_prompts:
        res = guard_classify(q)
        print("\n=== 프롬프트 ===")
        print(q)
        print("=== 판정 결과 ===")
        print(res)
