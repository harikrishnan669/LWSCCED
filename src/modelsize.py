from pathlib import Path

fp32 = Path("../onnx_encoder/model.onnx")
int8 = Path("../onnx_encoder/model_int8.onnx")

fp32_mb = fp32.stat().st_size / (1024 * 1024)
int8_mb = int8.stat().st_size / (1024 * 1024)

reduction = ((fp32_mb - int8_mb) / fp32_mb) * 100

print(f"FP32 Model : {fp32_mb:.2f} MB")
print(f"INT8 Model : {int8_mb:.2f} MB")
print(f"Reduction  : {reduction:.2f}%")