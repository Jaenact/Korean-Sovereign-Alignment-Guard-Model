# config.py

from pathlib import Path

# --- Paths ---
# All paths are relative to the project root (moderator_guard)
# Use BASE_DIR.parent to get the absolute path to the project root if needed.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Dataset"
CLASSIFIER_DIR = BASE_DIR / "moderator_guard" / "classifier"

# Input Data
DATASET_PATH = DATA_DIR / "augmented_dataset.json"
FACT_POSITIONS_PATH = DATA_DIR / "korea_fact_positions.json" # This file seems to be missing, but referenced.
FACT_TEXTS_PATH = CLASSIFIER_DIR / "fact.json"

# Cached/Generated Files
NLI_EMB_CACHE_PATH = CLASSIFIER_DIR / "nli_pair_embeddings.npy"
NLI_LABEL_CACHE_PATH = CLASSIFIER_DIR / "nli_pair_labels.npy"
FACT_EMB_CACHE_PATH = CLASSIFIER_DIR / "fact_embeddings.npy"

# Model Files
NLI_MODEL_PATH = CLASSIFIER_DIR / "nli_contradict_classifier.joblib"

# --- Models ---
EMBEDDING_MODEL = "text-embedding-3-small"

# --- Keywords ---
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
