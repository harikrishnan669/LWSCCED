from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from utils import normalize_text


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path("..")

ENCODER_DIR = BASE_DIR / "onnx_encoder"
CLASSIFIER_PATH = (
    BASE_DIR
    / "onnx_classifier"
    / "classifier.onnx"
)

CONFIDENCE_THRESHOLD = 0.40
MAX_LENGTH = 32


# =========================================================
# LOAD MODELS
# =========================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    str(ENCODER_DIR)
)

print("Loading encoder...")

encoder_session = ort.InferenceSession(
    str(ENCODER_DIR / "model_int8.onnx"),
    providers=["CPUExecutionProvider"]
)

print("Loading classifier...")

classifier_session = ort.InferenceSession(
    str(CLASSIFIER_PATH),
    providers=["CPUExecutionProvider"]
)

print("System Ready!")


# =========================================================
# EMBEDDING
# =========================================================

def get_embedding(text: str) -> np.ndarray:

    tokens = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="np"
    )

    encoder_inputs = {}

    for inp in encoder_session.get_inputs():

        if inp.name == "input_ids":
            encoder_inputs[inp.name] = tokens["input_ids"]

        elif inp.name == "attention_mask":
            encoder_inputs[inp.name] = (
                tokens["attention_mask"]
            )

        elif (
            inp.name == "token_type_ids"
            and "token_type_ids" in tokens
        ):
            encoder_inputs[inp.name] = (
                tokens["token_type_ids"]
            )

    outputs = encoder_session.run(
        None,
        encoder_inputs
    )

    embedding = outputs[0]

    # Mean Pooling
    if len(embedding.shape) == 3:
        embedding = embedding.mean(axis=1)

    return embedding.astype(np.float32)


# =========================================================
# PREDICTION
# =========================================================

def predict_command(
    text: str
) -> Tuple[str, float]:

    normalized_text = normalize_text(text)

    embedding = get_embedding(
        normalized_text
    )

    classifier_input_name = (
        classifier_session
        .get_inputs()[0]
        .name
    )

    outputs = classifier_session.run(
        None,
        {
            classifier_input_name:
            embedding
        }
    )

    # --------------------------------------------------
    # Case 1:
    # sklearn-onnx output:
    # [label_array, [{class1: prob1, class2: prob2}]]
    # --------------------------------------------------

    if (
        len(outputs) >= 2
        and isinstance(outputs[1], list)
        and len(outputs[1]) > 0
        and isinstance(outputs[1][0], dict)
    ):

        prob_dict = outputs[1][0]

        predicted_label = max(
            prob_dict,
            key=prob_dict.get
        )

        confidence = float(
            prob_dict[predicted_label]
        )

    # --------------------------------------------------
    # Case 2:
    # Probability array output
    # --------------------------------------------------

    else:

        probabilities = np.asarray(
            outputs[-1]
        )

        probabilities = probabilities.squeeze()

        max_index = int(
            np.argmax(probabilities)
        )

        confidence = float(
            probabilities[max_index]
        )

        predicted_label = (
            f"CLASS_{max_index}"
        )

    if confidence < CONFIDENCE_THRESHOLD:
        return (
            "REJECTED_OOS",
            confidence
        )

    return (
        predicted_label,
        confidence
    )

# =========================================================
# MAIN
# =========================================================

def main() -> None:

    print("\nSemantic Command Classifier")
    print("Type 'exit' to quit.\n")

    while True:

        text = input(
            "Enter command: "
        )

        if text.lower() == "exit":
            break

        label, confidence = (
            predict_command(text)
        )

        print(
            f"\nPrediction : {label}"
        )

        print(
            f"Confidence : {confidence:.4f}"
        )

        print("-" * 40)


if __name__ == "__main__":
    main()