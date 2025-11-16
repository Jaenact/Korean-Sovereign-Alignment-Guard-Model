# moderator_guard/training/train_classifier.py

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from joblib import dump

from moderator_guard.config import DATASET_PATH, NLI_MODEL_PATH, NLI_EMB_CACHE_PATH, NLI_LABEL_CACHE_PATH
from moderator_guard.classifier.utils import prepare_nli_embeddings_and_labels

def train_nli_classifier(
    dataset_path: Path = DATASET_PATH,
    model_output_path: Path = NLI_MODEL_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Trains and saves the NLI (Natural Language Inference) classifier.

    This classifier is trained to determine if a given sentence (a user's prompt)
    contradicts a known fact (a reference statement).

    The process involves:
    1. Preparing the data: Loading the augmented dataset, creating sentence-fact pairs,
       and generating embeddings for them. This is handled by `prepare_nli_embeddings_and_labels`,
       which uses caching for efficiency.
    2. Splitting the data: The dataset is split into training and testing sets.
    3. Training the model: A Logistic Regression model is trained on the training data.
       `class_weight="balanced"` is used to handle potential class imbalance.
    4. Evaluating the model: The trained model is evaluated on the test set, and a
       classification report and confusion matrix are printed.
    5. Saving the model: The final trained classifier is saved to a file using joblib.
    """

    # 1. Prepare embeddings and labels, using cache if available.
    X_all, y_all = prepare_nli_embeddings_and_labels(
        dataset_path=dataset_path,
        emb_cache_path=NLI_EMB_CACHE_PATH,
        label_cache_path=NLI_LABEL_CACHE_PATH,
    )

    # 2. Split data into training and testing sets.
    # `stratify=y_all` ensures that the class distribution is the same in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X_all,
        y_all,
        test_size=test_size,
        random_state=random_state,
        stratify=y_all,
    )

    print(f"[NLI] Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # 3. Train a simple but effective Logistic Regression classifier.
    clf = LogisticRegression(
        class_weight="balanced",  # Helps with imbalanced datasets.
        max_iter=1000,
        n_jobs=-1,  # Use all available CPU cores.
    )
    clf.fit(X_train, y_train)

    # 4. Evaluate the classifier on the held-out test set.
    y_pred = clf.predict(X_test)
    print("=== NLI Classification Report (CONTRADICT vs NOT) ===")
    print(classification_report(y_test, y_pred, digits=4))
    print("=== NLI Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    # 5. Save the trained model for later use in the main application.
    dump(clf, model_output_path)
    print(f"[NLI] Classifier saved: {model_output_path}")

if __name__ == '__main__':
    print("Starting NLI classifier training...")
    train_nli_classifier()
    print("Training complete.")