from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

from utils import normalize_text


# =========================================================
# PATHS
# =========================================================

MODEL_DIR: Path = Path("../models")


# =========================================================
# CONFIG
# =========================================================

CONFIDENCE_THRESHOLD: float = 0.40


# =========================================================
# LOAD MODELS
# =========================================================

print("Loading embedding model...")

embedding_model: SentenceTransformer = (
    SentenceTransformer(
        str(MODEL_DIR / "embedding_model")
    )
)

print("Loading classifier...")

classifier = joblib.load(
    MODEL_DIR / "classifier.joblib"
)

print("System Ready!")


# =========================================================
# PREDICTION
# =========================================================

def predict_command(
    text: str,
) -> Tuple[str, float]:
    """
    Predict semantic command.
    """

    normalized_text: str = normalize_text(text)

    embedding = embedding_model.encode(
        [normalized_text],
        convert_to_numpy=True,
    )

    probabilities = classifier.predict_proba(
        embedding
    )[0]

    max_index: int = int(
        np.argmax(probabilities)
    )

    confidence: float = float(
        probabilities[max_index]
    )

    predicted_label: str = (
        classifier.classes_[max_index]
    )

    if confidence < CONFIDENCE_THRESHOLD:
        return "REJECTED_OOS", confidence

    return predicted_label, confidence


# =========================================================
# MAIN LOOP
# =========================================================

def main() -> None:
    """
    Interactive demo.
    """

    print("\nSemantic Command Classifier")
    print("Type 'exit' to quit.\n")

    while True:

        user_input: str = input(
            "Enter command: "
        )

        if user_input.lower() == "exit":
            break

        label, confidence = predict_command(
            user_input
        )

        print(f"\nPrediction : {label}")
        print(f"Confidence : {confidence:.4f}")
        print("-" * 40)


if __name__ == "__main__":
    main()
