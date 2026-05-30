#Convert the ONNX MiniLM model from FP32 (32-bit floating point) weights to INT8 (8-bit integer) weights to reduce model size and improve inference speed.
from __future__ import annotations

from pathlib import Path

from onnxruntime.quantization import (
    QuantType,
    quantize_dynamic,
)


INPUT_MODEL: Path = Path(
    "../onnx_encoder/model.onnx" # input the model obtained from encoder converted to ONNX
)

OUTPUT_MODEL: Path = Path(
    "../onnx_encoder/model_int8.onnx" # output the 8-bit model so that size reduces
)


def main() -> None:

    quantize_dynamic(
        model_input=str(INPUT_MODEL),
        model_output=str(OUTPUT_MODEL),
        weight_type=QuantType.QInt8, # Convert to Signed 8-bit Integer
    )

    print("\nINT8 quantization completed!")
    print(f"Saved to: {OUTPUT_MODEL.resolve()}")

if __name__ == "__main__":
    main()
