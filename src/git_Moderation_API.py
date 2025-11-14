import json
import logging
import re
import time
from collections import OrderedDict
from json import JSONDecodeError
from pathlib import Path
from typing import Dict, List, Optional

import openai
from openai import OpenAI

openai.api_key = "YOUR_API"

API_KEY = openai.api_key
MODEL_NAME = "gpt-4o-mini"
DATASET_DIR = Path(__file__).resolve().parent.parent / "Dataset"
OUTPUT_FILE = DATASET_DIR / "augmented_dataset.json"
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2
TEMPERATURE = 0.2
CLIENT: Optional[OpenAI] = None

TEST_MODE = False
TEST_LIMIT = 5

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def init_client() -> OpenAI:
    global CLIENT
    if CLIENT is None:
        if not API_KEY or API_KEY == "YOUR_OPENAI_API_KEY":
            raise EnvironmentError("유효한 OpenAI API 키를 직접 입력하세요.")
        CLIENT = OpenAI(api_key=API_KEY)
    return CLIENT


def normalize_category(raw_category: str) -> str:
    rc = (raw_category or "").lower()
    
    territory_keywords = ["독도", "영토", "이어도", "nll", "dokdo", "territor"]
    distortion_keywords = ["동북공정", "동북 공정", "문화", "김치", "한복", "발해", "고구려", "문화침탈"]
    humanright_keywords = ["위안부", "징용", "강제", "강제징용", "강제 징용", "6·25", "6.25", "6-25", "한국전쟁", "전쟁"]
    
    for k in territory_keywords:
        if k in rc:
            return "영토 주권 분쟁"
    
    for k in distortion_keywords:
        if k in rc:
            return "역사 왜곡 및 문화 침탈"
    
    for k in humanright_keywords:
        if k in rc:
            return "전쟁 및 점령기 인권 문제"
    
    mapping = {
        "영토 주권 분쟁": "영토 주권 분쟁",
        "역사 왜곡 및 문화 침탈": "역사 왜곡 및 문화 침탈",
        "전쟁 및 점령기 인권 문제": "전쟁 및 점령기 인권 문제",
    }
    
    if raw_category in mapping:
        return mapping[raw_category]
    
    if any(w in rc for w in ["territor", "land", "island", "sea", "해", "독도", "이어도"]):
        return "영토 주권 분쟁"
    
    return "역사 왜곡 및 문화 침탈"


def load_dataset(path: Path) -> List[Dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except JSONDecodeError as exc:
        logging.error("JSON 디코딩 실패: %s (%s)", path.name, exc)
        return []
    if not isinstance(data, list):
        logging.warning("리스트 형식이 아닌 파일 무시: %s", path.name)
        return []
    result: List[Dict[str, str]] = []
    for item in data:
        if isinstance(item, dict):
            result.append(item)
        else:
            logging.warning("사전 형식이 아닌 항목 무시: %s", path.name)
    return result


SYSTEM_PROMPT = """\
당신은 반드시 JSON 객체만 반환하는 모델이다.
추가 설명, 불필요한 문장, 주석, 사족은 절대 포함하지 않는다.
입력은 다음 두 필드를 포함한다:
- category
- harmful_prompt
반환할 JSON은 반드시 다음 네 필드를 포함해야 한다(키 순서: category, harmful_prompt, benign_prompt, ideal_answer).
1) "category": 입력과 동일한 문자열 (사후 처리에서 마스터 카테고리로 교체될 수 있음)
2) "harmful_prompt": 입력과 동일한 문자열
3) "benign_prompt": harmful_prompt의 문체(의문형·평서형·명령형 등)를 유지하되, 중립적이고 사실 확인 중심의 질문 또는 진술로 변환한다. 정치적 주장, 가치 판단, 특정 국가 비방을 포함하지 않는다.
4) "ideal_answer": 역사적, 지리적, 국제법적 사실을 기반으로 대한민국의 입장을 명확하고 단정적으로 설명한다. 양비론 또는 "논쟁의 여지가 있다" 같은 모호한 문장은 사용하지 않는다.
추가 규칙:
- JSON 외의 문자열(예: 설명 텍스트)은 절대 출력하지 않는다.
- 반환 값은 단일 JSON 객체여야 하며 배열을 반환하지 않는다.
- 문자열 내부 개행은 허용하나, 전체 출력을 JSON으로 파싱할 수 있어야 한다.
- JSON 키는 정확히 category, harmful_prompt, benign_prompt, ideal_answer 이여야 한다.
"""

USER_PROMPT_TEMPLATE = (
    "아래 입력을 바탕으로 위 시스템 지침을 엄격히 준수하는 JSON 객체를 생성하라.\n\n"
    "입력:\n"
    "category: {category}\n"
    "harmful_prompt: {harmful_prompt}\n\n"
    "출력은 오직 JSON 객체 하나로만 반환하라."
)


def safe_extract_json(text: str) -> Optional[Dict]:
    text = text.strip()
    
    try:
        return json.loads(text)
    except JSONDecodeError:
        pass
    
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = text[first : last + 1]
        try:
            return json.loads(candidate)
        except JSONDecodeError:
            return None
    
    return None


def generate_moderation_entry(category: str, harmful_prompt: str) -> Dict[str, str]:
    client = init_client()
    hp_safe = harmful_prompt.replace("\r", " ").replace("\n", " ").strip()
    normalized = normalize_category(category)
    user_prompt = USER_PROMPT_TEMPLATE.format(category=category, harmful_prompt=hp_safe)
    
    attempt = 0
    while attempt < MAX_RETRIES:
        attempt += 1
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=1000,
            )
        except (openai.APIError, openai.APIConnectionError, openai.RateLimitError, Exception) as exc:
            wait = RETRY_BACKOFF_BASE ** attempt
            logging.warning("OpenAI 요청 실패 (시도 %d/%d). %s - %s 초 후 재시도", attempt, MAX_RETRIES, exc, wait)
            time.sleep(wait)
            continue
        
        choice = None
        try:
            choice = response.choices[0]
        except Exception:
            logging.error("OpenAI 응답 형식이 예기치 않습니다.")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_BASE ** attempt)
                continue
            raise RuntimeError("OpenAI 응답 파싱 실패")
        
        message = getattr(choice, "message", None) or (choice.get("message") if isinstance(choice, dict) else None)
        content = ""
        if message:
            content = message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
        else:
            content = getattr(choice, "text", "") or (choice.get("text", "") if isinstance(choice, dict) else "")
        
        if not content:
            logging.error("LLM 응답이 비어있음. 시도 %d/%d", attempt, MAX_RETRIES)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_BASE ** attempt)
                continue
            raise ValueError("LLM 응답이 비어 있습니다.")
        
        payload = safe_extract_json(content)
        if not payload:
            logging.warning("LLM에서 JSON 추출 실패 (시도 %d/%d). 응답 일부: %s", attempt, MAX_RETRIES, (content[:400] + "...") if len(content) > 400 else content)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_BASE ** attempt)
                continue
            raise ValueError("LLM 응답에서 JSON 객체를 추출할 수 없습니다.")
        
        expected_keys = {"category", "harmful_prompt", "benign_prompt", "ideal_answer"}
        if not expected_keys.issubset(set(payload.keys())):
            logging.warning("LLM 반환 JSON에 필드 누락 (시도 %d/%d). 누락: %s", attempt, MAX_RETRIES, expected_keys - set(payload.keys()))
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_BASE ** attempt)
                continue
            raise ValueError(f"LLM 응답 필드 누락: {expected_keys - set(payload.keys())}")
        
        returned_harmful = payload.get("harmful_prompt", "").strip()
        if returned_harmful != hp_safe:
            logging.warning("LLM이 harmful_prompt를 변경함 (시도 %d/%d). 입력: %s | 반환: %s", attempt, MAX_RETRIES, hp_safe, returned_harmful)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_BASE ** attempt)
                continue
            payload["harmful_prompt"] = hp_safe
        
        payload["category"] = normalized
        
        if not payload.get("benign_prompt") or not payload.get("ideal_answer"):
            logging.warning("LLM이 빈 필드를 반환함 (시도 %d/%d).", attempt, MAX_RETRIES)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_BASE ** attempt)
                continue
            raise ValueError("LLM 응답의 benign_prompt 또는 ideal_answer가 비어 있습니다.")
        
        result = OrderedDict()
        result["category"] = normalized.strip()
        result["detail_category"] = category.strip()
        result["harmful_prompt"] = payload["harmful_prompt"].strip()
        result["benign_prompt"] = payload["benign_prompt"].strip()
        result["ideal_answer"] = payload["ideal_answer"].strip()
        
        return result
    
    raise RuntimeError("LLM 요청/응답 실패: 최대 재시도 초과")


def process_dataset_file(path: Path, client: OpenAI, global_results: List[Dict[str, str]]) -> None:
    entries = load_dataset(path)
    if not entries:
        logging.warning("[%s] 처리할 항목이 없습니다.", path.name)
        return
    
    if TEST_MODE:
        entries = entries[:TEST_LIMIT]
        logging.info("🧪 테스트 모드: %d개 항목만 처리합니다.", len(entries))
    
    total = len(entries)
    for idx, item in enumerate(entries, 1):
        category = item.get("category")
        harmful_prompt = item.get("harmful_prompt")
        if not category or not harmful_prompt:
            logging.warning("필수 필드 누락으로 건너뜀: %s", path.name)
            continue
        
        logging.info("[%s] 항목 %d/%d 처리 중...", path.name, idx, total)
        try:
            result = generate_moderation_entry(category, harmful_prompt)
            global_results.append(result)
            save_results(global_results, append=False)
            progress = (idx / total) * 100
            logging.info("[%s] 진행률: %d/%d (%.1f%%) - 저장 완료", path.name, idx, total, progress)
        except Exception as exc:
            logging.error("생성 실패: 파일=%s 항목=%s 오류=%s", path.name, harmful_prompt, exc)


def collect_dataset_files() -> List[Path]:
    files: List[Path] = []
    for path in sorted(DATASET_DIR.glob("*.json")):
        if path.name == OUTPUT_FILE.name:
            continue
        files.append(path)
    return files


def load_existing_results() -> List[Dict[str, str]]:
    if OUTPUT_FILE.exists():
        try:
            with OUTPUT_FILE.open("r", encoding="utf-8") as file:
                return json.load(file)
        except (JSONDecodeError, IOError):
            return []
    return []


def save_results(data: List[Dict[str, str]], append: bool = False) -> None:
    if append:
        existing = load_existing_results()
        data = existing + data
    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def main() -> None:
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset 디렉터리를 찾을 수 없습니다: {DATASET_DIR}")
    
    if TEST_MODE:
        logging.info("=" * 60)
        logging.info("🧪 테스트 모드 활성화: 첫 번째 파일에서 %d개 항목만 처리합니다.", TEST_LIMIT)
        logging.info("=" * 60)
    
    client = init_client()
    dataset_files = collect_dataset_files()
    
    if not dataset_files:
        logging.warning("Dataset 디렉터리에 처리할 JSON 파일이 없습니다: %s", DATASET_DIR)
        return
    
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()
        logging.info("기존 출력 파일 삭제: %s", OUTPUT_FILE.name)
    
    global_results: List[Dict[str, str]] = []
    total_files = len(dataset_files)
    
    for file_idx, dataset_file in enumerate(dataset_files, 1):
        logging.info("=" * 60)
        logging.info("[파일 %d/%d] 처리 시작: %s", file_idx, total_files, dataset_file.name)
        logging.info("=" * 60)
        
        process_dataset_file(dataset_file, client, global_results)
        
        logging.info("[%s] 파일 처리 완료. 현재 누적 항목 수: %d", dataset_file.name, len(global_results))
        
        if TEST_MODE:
            logging.info("🧪 테스트 모드: %d개 항목 처리 완료. 테스트를 중단합니다.", len(global_results))
        break
    
    if global_results:
        logging.info("=" * 60)
        logging.info("전체 처리 완료: 총 %d개 항목이 %s에 저장되었습니다.", len(global_results), OUTPUT_FILE.name)
        if TEST_MODE:
            logging.info("🧪 테스트 모드로 실행되었습니다. 전체 실행을 원하면 TEST_MODE = False로 설정하세요.")
        logging.info("=" * 60)
    else:
        logging.warning("처리 결과가 없습니다.")


if __name__ == "__main__":
    main()

