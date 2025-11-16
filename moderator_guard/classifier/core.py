# moderator_guard/classifier/core.py

from typing import List, Dict, Tuple
import numpy as np
from joblib import load

from moderator_guard.config import DOMAIN_KEYWORDS, NLI_MODEL_PATH
from moderator_guard.llm.client import get_embedding
from moderator_guard.classifier.utils import load_fact_index, search_related_facts, load_nli_classifier

# ----------------------------------- 
# 7. Intent Classifier (Rule-based)
# ----------------------------------- 

def classify_intent(prompt: str) -> str:
    text = prompt.strip()

    # 1) Citing/questioning a distorted claim -> ASK_ABOUT_DISTORTION (maintain as is)
    distortion_keywords = [
        "some say", "there is a claim", "the claim that",
        "claim", "basis of the claim", "why is this claim",
        "is this claim correct", "this claim is",
    ]
    if any(kw in text for kw in distortion_keywords) and "?" in text:
        return "ASK_ABOUT_DISTORTION"

    # 2) Premise-based question -> ASSERTION (maintain already added patterns)
    assertion_question_patterns = [
        "wasn't it?", "isn't it?", "right?", "isn't that right?",
        "isnt that right?", "isnt it?", "isnt it?", "wasnt it?",
        "isnt it?", "isnt that right?", "isnt that right?", "isnt it?",
        "is that right?", "is that right?",
        "you know?", "you know?", "dont you?",
    ]
    if any(pat in text for pat in assertion_question_patterns):
        return "ASSERTION"

    # 3) Declarative assertion (formal + informal) -> ASSERTION
    declarative_suffixes = [
        "is", "was", "did", "has",
        "was not", "was",
        "is", "is", "is", "is", "is",
        "is", "is.", "I think", "I believe", "it can be said",
    ]

    if "?" not in text:
        # If it ends with one of the above suffixes, consider it a statement
        if any(text.endswith(suf) for suf in declarative_suffixes):
            return "ASSERTION"
        # Check again with a period
        if any(text.endswith(suf + ".") for suf in declarative_suffixes):
            return "ASSERTION"

        # If there's no question mark and the length is sufficient, treat as an assertion
        if len(text) >= 8:
            return "ASSERTION"

    # 4) General question -> ASK_INFO
    if text.endswith("?") or "why" in text or "how" in text:
        return "ASK_INFO"

    # 5) Other
    return "OTHER"


# ----------------------------------- 
# 8. Domain Filter
# ----------------------------------- 

def is_in_domain(prompt: str) -> bool:
    """Determines if the prompt belongs to our covered domains (territory/ancient history/war/Japanese colonial human rights)."""
    text = prompt.strip()
    return any(kw in text for kw in DOMAIN_KEYWORDS)

# ----------------------------------- 
# 9. NLI Classifier Prediction
# ----------------------------------- 

def predict_contradict_prob(clf, prompt: str, fact_text: str) -> float:
    """
    Returns the probability of CONTRADICT(1) for a (prompt, reference) pair.
    """
    pair_text = f"[Sentence]: {prompt}\n[Reference]: {fact_text}"
    emb = get_embedding(pair_text).reshape(1, -1)
    prob_contradict = clf.predict_proba(emb)[0, 1]
    return float(prob_contradict)

# ----------------------------------- 
# 11. Final Judgment (CONTRADICT ∧ ASSERTION → HARMFUL)
# ----------------------------------- 

def classify_prompt(prompt: str,
                    nli_clf=None,
                    fact_texts: List[str] = None,
                    fact_embs: np.ndarray = None,
                    contradict_threshold: float = 0.6,
                    top_k_facts: int = 5) -> Dict:
    """
    Finally determines if the prompt is a harmful distorted statement.

    This function implements the core 3-stage classification logic:
    1. Domain Filter: Checks if the prompt is related to the topics we cover.
    2. Intent Classifier: Determines the user's intent (e.g., asking a question vs. asserting a belief).
    3. NLI Classifier: Checks if the prompt contradicts established facts.

    A prompt is judged as HARMFUL only if it meets all three conditions:
    - Is within the domain.
    - Has an 'ASSERTION' intent.
    - Contradicts a known fact with high probability.
    """

    result = {
        "prompt": prompt,
        "in_domain": False,
        "intent": None,
        "max_contradict_prob": 0.0,
        "harmful": False,
        "reason": "",
    }

    # Stage 1: Domain Filter
    # If the prompt is not about our target topics, we can immediately classify it as safe.
    if not is_in_domain(prompt):
        result["in_domain"] = False
        result["harmful"] = False
        result["reason"] = "Determined to be outside the covered domains (territory/ancient history/war/Japanese colonial human rights)."
        return result

    result["in_domain"] = True

    # Stage 2: Intent Classifier
    # We determine if the user is asking a question or asserting a potentially distorted belief as fact.
    intent = classify_intent(prompt)
    result["intent"] = intent

    # Prepare resources for the final stage.
    if nli_clf is None:
        nli_clf = load_nli_classifier()
    if fact_texts is None or fact_embs is None:
        fact_texts, fact_embs = load_fact_index()

    # Retrieve the most relevant facts to the user's prompt.
    related_facts = search_related_facts(
        prompt,
        fact_texts,
        fact_embs,
        top_k=top_k_facts,
    )

    # Stage 3: NLI (Contradiction) Classifier
    # We check if the prompt contradicts any of the retrieved facts. We take the highest contradiction
    # probability among the top-k facts.
    max_prob = 0.0
    for fact_text, sim in related_facts:
        prob = predict_contradict_prob(nli_clf, prompt, fact_text)
        if prob > max_prob:
            max_prob = prob

    result["max_contradict_prob"] = max_prob

    # Final Judgment: A prompt is harmful if it's an assertion that contradicts the facts.
    if max_prob >= contradict_threshold and intent == "ASSERTION":
        result["harmful"] = True
        result["reason"] = (
            "The statement is within the domain, has a high probability of contradicting the reference, "
            "and the intent is judged as an ASSERTION (asserting a distorted claim as fact)."
        )
    else:
        result["harmful"] = False
        if max_prob < contradict_threshold:
            result["reason"] = (
                "The statement is within the domain, but the contradiction probability is below the threshold."
            )
        else:
            result["reason"] = (
                "There is a possibility of contradiction, but the intent is not an ASSERTION, "
                "so it is not considered to be asserting a distorted claim as fact."
            )

    return result

# -----------------------------------
# 12. Convenience Functions for Execution
# -----------------------------------

_nli_clf = None
_fact_texts = None
_fact_embs = None

def get_guard_resources():
    """
    Loads the NLI model and fact index only once and reuses them from a global cache.
    This prevents reloading these large files from disk on every single classification call,
    significantly improving performance for interactive use.
    """
    global _nli_clf, _fact_texts, _fact_embs

    if _nli_clf is None:
        _nli_clf = load_nli_classifier()
    if _fact_texts is None or _fact_embs is None:
        _fact_texts, _fact_embs = load_fact_index()

    return _nli_clf, _fact_texts, _fact_embs
def guard_classify(prompt: str) -> Dict:
    """The final guard function to be used externally."""
    nli_clf, fact_texts, fact_embs = get_guard_resources()
    res = classify_prompt(
        prompt,
        nli_clf=nli_clf,
        fact_texts=fact_texts,
        fact_embs=fact_embs,
    )
    return res

# ----------------------------------- 
# 13. Simple Test on Standalone Execution
# ----------------------------------- 

if __name__ == "__main__":
    # ⚠️ This does not train, only tests classification using pre-trained/embedded files.
    test_prompts = [
        "Wasn't Goguryeo a minority regime established by a minority group in China?",
        "Some claim that Goguryeo should be seen as a local government of China. What is the basis for this claim?",
        "Tell me about the Republic of Korea.",
    ]

    for q in test_prompts:
        res = guard_classify(q)
        print("\n=== Prompt ===")
        print(q)
        print("=== Judgment ===")
        print(res)