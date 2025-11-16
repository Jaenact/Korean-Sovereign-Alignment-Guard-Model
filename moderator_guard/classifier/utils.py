# moderator_guard/classifier/utils.py

import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

from moderator_guard.config import (
    DATASET_PATH,
    NLI_EMB_CACHE_PATH,
    NLI_LABEL_CACHE_PATH,
    FACT_POSITIONS_PATH,
    FACT_TEXTS_PATH,
    FACT_EMB_CACHE_PATH,
)
from moderator_guard.llm.client import embed_texts, get_embedding


def load_augmented_dataset(path: Path = DATASET_PATH) -> List[Dict]:
    """Loads the augmented_dataset.json file and returns it as a list."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data


def build_nli_pairs(dataset: List[Dict]) -> Tuple[List[str], np.ndarray]:
    """
    Creates (sentence + fact pair, label) lists for NLI training.
    - harmful_prompt + ideal_answer -> CONTRADICT (1)
    - benign_prompt  + ideal_answer -> NOT_CONTRADICT (0)
    """
    pair_texts: List[str] = []
    labels: List[int] = []

    for item in dataset:
        harmful = item["harmful_prompt"]
        benign = item["benign_prompt"]
        fact = item["ideal_answer"]

        # harmful -> CONTRADICT
        pair_h = f"[Sentence]: {harmful}\n[Reference]: {fact}"
        pair_texts.append(pair_h)
        labels.append(1)

        # benign -> NOT_CONTRADICT
        pair_b = f"[Sentence]: {benign}\n[Reference]: {fact}"
        pair_texts.append(pair_b)
        labels.append(0)

    labels_arr = np.array(labels, dtype=int)
    return pair_texts, labels_arr


def prepare_nli_embeddings_and_labels(
    dataset_path: Path = DATASET_PATH,
    emb_cache_path: Path = NLI_EMB_CACHE_PATH,
    label_cache_path: Path = NLI_LABEL_CACHE_PATH,
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepares NLI training pair embeddings and labels, using cache if available."""
    dataset = load_augmented_dataset(dataset_path)

    if emb_cache_path.exists() and label_cache_path.exists():
        print("[NLI] Cache exists, loading directly.")
        X_all = np.load(emb_cache_path)
        y_all = np.load(label_cache_path)

        pair_texts, labels = build_nli_pairs(dataset)
        if X_all.shape[0] != len(pair_texts) or y_all.shape[0] != len(pair_texts):
            print("⚠️ NLI cache size mismatch. Regenerating cache.")
            X_all = embed_texts(pair_texts)
            y_all = labels
            np.save(emb_cache_path, X_all)
            np.save(label_cache_path, y_all)
    else:
        print("[NLI] No cache found, calculating new embeddings.")
        pair_texts, labels = build_nli_pairs(dataset)
        X_all = embed_texts(pair_texts)
        y_all = labels
        np.save(emb_cache_path, X_all)
        np.save(label_cache_path, y_all)
        print(f"[NLI] Embedding cache saved: {emb_cache_path}")
        print(f"[NLI] Label cache saved: {label_cache_path}")

    return X_all, y_all


def build_fact_index_from_positions(
    fact_positions_path: Path = FACT_POSITIONS_PATH,
    fact_texts_path: Path = FACT_TEXTS_PATH,
    fact_emb_path: Path = FACT_EMB_CACHE_PATH,
):
    """
    Extracts fact_text from korea_fact_positions.json to create
    fact_texts.json + fact_embeddings.npy. (Run once)
    """
    if not fact_positions_path.exists():
        raise FileNotFoundError(f"Fact dataset file not found: {fact_positions_path}")

    data = json.loads(fact_positions_path.read_text(encoding="utf-8"))
    fact_texts = [item["fact_text"] for item in data]

    fact_texts_path.write_text(
        json.dumps(fact_texts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[FACT] fact_texts.json saved: {fact_texts_path}")

    fact_embs = embed_texts(fact_texts)
    np.save(fact_emb_path, fact_embs)
    print(f"[FACT] fact_embeddings.npy saved: {fact_emb_path}")


def load_fact_index(
    fact_texts_path: Path = FACT_TEXTS_PATH,
    fact_emb_path: Path = FACT_EMB_CACHE_PATH,
) -> Tuple[List[str], np.ndarray]:
    """Loads fact_texts.json + fact_embeddings.npy."""
    if not fact_texts_path.exists():
        raise FileNotFoundError(f"fact_texts.json not found: {fact_texts_path}")
    if not fact_emb_path.exists():
        raise FileNotFoundError(f"fact_embeddings.npy not found: {fact_emb_path}")

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
    Searches for the top_k most relevant facts to the prompt using cosine similarity.
    Returns: [(fact_text, similarity), ...]
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
