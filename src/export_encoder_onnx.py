#Scipt used to conver the Embedding model (all-MiniLM-L6-v2) to ONNX
from __future__ import annotations

from pathlib import Path

from optimum.onnxruntime import ORTModelForFeatureExtraction # Optimum run time for feature extraction
from transformers import AutoTokenizer # Loads the tokenizer associated with MiniLM


MODEL_NAME: str = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

OUTPUT_DIR: Path = Path("../onnx_encoder")

# Export sentence transformer encoder to ONNX.
def main() -> None:

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("Exporting encoder to ONNX...")

    model = ORTModelForFeatureExtraction.from_pretrained(
        MODEL_NAME,
        export=True,
    )

    model.save_pretrained(OUTPUT_DIR)

    tokenizer.save_pretrained(OUTPUT_DIR) #save the tokenizer to load the model

    print("\nONNX encoder exported successfully!")
    print(f"Saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
