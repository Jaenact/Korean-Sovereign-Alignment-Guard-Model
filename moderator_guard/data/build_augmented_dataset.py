import json
import logging
import time
from collections import OrderedDict
from json import JSONDecodeError
from pathlib import Path
from typing import Dict, List, Optional

import openai

# Import from the new modular structure
from moderator_guard.llm.client import client as API_CLIENT
from moderator_guard.config import (
    DATASET_DIR,
    EMBEDDING_MODEL as MODEL_NAME, # Or a new config variable for generation model
)

# --- Configuration ---
OUTPUT_FILE = DATASET_DIR / "augmented_dataset.json"
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2
TEMPERATURE = 0.2

TEST_MODE = False
TEST_LIMIT = 5

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


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
        logging.error("JSON decoding failed: %s (%s)", path.name, exc)
        return []
    if not isinstance(data, list):
        logging.warning("Ignoring file that is not a list: %s", path.name)
        return []
    result: List[Dict[str, str]] = []
    for item in data:
        if isinstance(item, dict):
            result.append(item)
        else:
            logging.warning("Ignoring non-dict item in %s", path.name)
    return result


SYSTEM_PROMPT = """\
You are a model that MUST return only a JSON object.
Never include additional explanations, unnecessary sentences, comments, or footnotes.
The input contains the following two fields:
- category
- harmful_prompt
The returned JSON MUST include the following four fields (in order: category, harmful_prompt, benign_prompt, ideal_answer).
1) "category": The same string as the input (may be replaced by a master category in post-processing).
2) "harmful_prompt": The same string as the input.
3) "benign_prompt": Maintain the style of the harmful_prompt (e.g., question, statement, command) but transform it into a neutral, fact-checking-oriented question or statement. Do not include political arguments, value judgments, or slander against specific countries.
4) "ideal_answer": Clearly and definitively explain the position of the Republic of Korea based on historical, geographical, and international legal facts. Do not use ambiguous sentences like "there is room for debate."
Additional rules:
- NEVER output any string other than the JSON (e.g., explanatory text).
- The return value must be a single JSON object, not an array.
- Newlines within strings are allowed, but the entire output must be parsable as JSON.
- The JSON keys must be exactly category, harmful_prompt, benign_prompt, and ideal_answer.
"""

USER_PROMPT_TEMPLATE = (
    "Based on the input below, generate a JSON object that strictly adheres to the system instructions above.\n\n"
    "Input:\n"
    "category: {category}\n"
    "harmful_prompt: {harmful_prompt}\n\n"
    "Output must be a single JSON object only."
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
    hp_safe = harmful_prompt.replace("\r", " ").replace("\n", " ").strip()
    normalized = normalize_category(category)
    user_prompt = USER_PROMPT_TEMPLATE.format(category=category, harmful_prompt=hp_safe)
    
    attempt = 0
    while attempt < MAX_RETRIES:
        attempt += 1
        try:
            response = API_CLIENT.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=1000,
            )
        except (openai.APIError, openai.APIConnectionError, openai.RateLimitError) as exc:
            wait = RETRY_BACKOFF_BASE ** attempt
            logging.warning("OpenAI request failed (attempt %d/%d). %s - Retrying in %s seconds", attempt, MAX_RETRIES, exc, wait)
            time.sleep(wait)
            continue
        except Exception as exc:
            logging.error("An unexpected error occurred during OpenAI API call: %s", exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_BASE ** attempt)
                continue
            raise RuntimeError("Failed to get response from LLM after multiple retries.") from exc

        
        try:
            content = response.choices[0].message.content
        except (IndexError, AttributeError, TypeError) as e:
            logging.error("Unexpected OpenAI response format: %s. Response: %s", e, response)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_BASE ** attempt)
                continue
            raise RuntimeError("Failed to parse LLM response.")
        
        if not content:
            logging.error("LLM response is empty. Attempt %d/%d", attempt, MAX_RETRIES)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_BASE ** attempt)
                continue
            raise ValueError("LLM response is empty.")
        
        payload = safe_extract_json(content)
        if not payload:
            logging.warning("Failed to extract JSON from LLM (attempt %d/%d). Response snippet: %s", attempt, MAX_RETRIES, (content[:400] + "...") if len(content) > 400 else content)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_BASE ** attempt)
                continue
            raise ValueError("Could not extract JSON object from LLM response.")
        
        expected_keys = {"category", "harmful_prompt", "benign_prompt", "ideal_answer"}
        if not expected_keys.issubset(set(payload.keys())):
            logging.warning("LLM returned JSON with missing fields (attempt %d/%d). Missing: %s", attempt, MAX_RETRIES, expected_keys - set(payload.keys()))
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_BASE ** attempt)
                continue
            raise ValueError(f"LLM response missing fields: {expected_keys - set(payload.keys())}")
        
        returned_harmful = payload.get("harmful_prompt", "").strip()
        if returned_harmful != hp_safe:
            logging.warning("LLM modified the harmful_prompt (attempt %d/%d). Input: %s | Returned: %s", attempt, MAX_RETRIES, hp_safe, returned_harmful)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_BASE ** attempt)
                continue
            payload["harmful_prompt"] = hp_safe
        
        payload["category"] = normalized
        
        if not payload.get("benign_prompt") or not payload.get("ideal_answer"):
            logging.warning("LLM returned empty fields (attempt %d/%d).", attempt, MAX_RETRIES)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_BASE ** attempt)
                continue
            raise ValueError("LLM response has empty benign_prompt or ideal_answer.")
        
        result = OrderedDict()
        result["category"] = normalized.strip()
        result["detail_category"] = category.strip()
        result["harmful_prompt"] = payload["harmful_prompt"].strip()
        result["benign_prompt"] = payload["benign_prompt"].strip()
        result["ideal_answer"] = payload["ideal_answer"].strip()
        
        return result
    
    raise RuntimeError("Failed to get valid LLM response after maximum retries.")


def process_dataset_file(path: Path, global_results: List[Dict[str, str]]) -> None:
    entries = load_dataset(path)
    if not entries:
        logging.warning("[%s] No items to process.", path.name)
        return
    
    if TEST_MODE:
        entries = entries[:TEST_LIMIT]
        logging.info("🧪 Test Mode: Processing only %d items.", len(entries))
    
    total = len(entries)
    for idx, item in enumerate(entries, 1):
        category = item.get("category")
        harmful_prompt = item.get("harmful_prompt")
        if not category or not harmful_prompt:
            logging.warning("Skipping item with missing required fields: %s", path.name)
            continue
        
        logging.info("[%s] Processing item %d/%d...", path.name, idx, total)
        try:
            result = generate_moderation_entry(category, harmful_prompt)
            global_results.append(result)
            save_results(global_results, append=False)
            progress = (idx / total) * 100
            logging.info("[%s] Progress: %d/%d (%.1f%%) - Saved", path.name, idx, total, progress)
        except Exception as exc:
            logging.error("Generation failed: file=%s item=%s error=%s", path.name, harmful_prompt, exc)


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
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")
    
    if TEST_MODE:
        logging.info("=" * 60)
        logging.info("🧪 TEST MODE ACTIVATED: Processing only %d items from the first file.", TEST_LIMIT)
        logging.info("=" * 60)
    
    dataset_files = collect_dataset_files()
    
    if not dataset_files:
        logging.warning("No JSON files to process in Dataset directory: %s", DATASET_DIR)
        return
    
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()
        logging.info("Deleted existing output file: %s", OUTPUT_FILE.name)
    
    global_results: List[Dict[str, str]] = []
    total_files = len(dataset_files)
    
    for file_idx, dataset_file in enumerate(dataset_files, 1):
        logging.info("=" * 60)
        logging.info("[File %d/%d] Starting processing: %s", file_idx, total_files, dataset_file.name)
        logging.info("=" * 60)
        
        process_dataset_file(dataset_file, global_results)
        
        logging.info("[%s] File processing complete. Current total items: %d", dataset_file.name, len(global_results))
        
        if TEST_MODE:
            logging.info("🧪 Test Mode: Processed %d items. Halting test.", len(global_results))
        break
    
    if global_results:
        logging.info("=" * 60)
        logging.info("Total processing complete: %d items saved to %s.", len(global_results), OUTPUT_FILE.name)
        if TEST_MODE:
            logging.info("🧪 Ran in Test Mode. To run on all data, set TEST_MODE = False.")
        logging.info("=" * 60)
    else:
        logging.warning("No results were processed.")


if __name__ == "__main__":
    main()
