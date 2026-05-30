# Script used to convert trained logistic regression classifier from sckit-learn format to ONNX format to run with windows,linux and embedded devices
from __future__ import annotations

from pathlib import Path

import joblib # load the pretrained classifier
from skl2onnx import convert_sklearn # convert Sckit-learn model to ONNX model
from skl2onnx.common.data_types import FloatTensorType


MODEL_DIR: Path = Path("../models")
OUTPUT_DIR: Path = Path("../onnx_classifier")


def main() -> None:
    """
    Export sklearn classifier to ONNX.
    """

    OUTPUT_DIR.mkdir(exist_ok=True)

    classifier = joblib.load(
        MODEL_DIR / "classifier.joblib" # loads trained classifier
    )

    initial_type = [
        (
            "float_input",
            FloatTensorType([None, 384]), # MiniLM create a total of 384 embeddings
        )
    ]

    onnx_model = convert_sklearn( # converts to ONNX
        classifier,
        initial_types=initial_type,
    )

    output_path: Path = (
        OUTPUT_DIR / "classifier.onnx"
    )

    with output_path.open("wb") as file:
        file.write(onnx_model.SerializeToString())

    print("\nClassifier exported successfully!")
    print(f"Saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
