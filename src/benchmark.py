# Used to measure how fast the quantized ONNX encoder runs interface on the CPU
# It helps in mainly converting text to embedded vector
from __future__ import annotations

import time
from statistics import mean

import onnxruntime as ort # Much faster
from transformers import AutoTokenizer


MODEL_PATH: str = (
    "../onnx_encoder/model_int8.onnx"
)

TOKENIZER_PATH: str = (
    "../onnx_encoder"
)


TEST_SENTENCES: list[str] = [
    "turn it down",
    "play the next song",
    "increase brightness",
    "what is the weather today",
]


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(  # Loads MiniLM tokenizer
        TOKENIZER_PATH
    )

    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"],  # Loads ONNX model into the memory
    )

    timings: list[float] = []

    for sentence in TEST_SENTENCES:

        encoded = tokenizer(
            sentence,
            return_tensors="np",
            padding=True,
            truncation=True,
        )

        start_time = time.perf_counter()  # Measure the execution time

        session.run(
            None,
            dict(encoded),
        )

        end_time = time.perf_counter()

        latency_ms: float = (
            (end_time - start_time) * 1000
        )

        timings.append(latency_ms)

        print(
            f"{sentence:<35}"
            f" {latency_ms:.2f} ms"
        )

    avg_latency: float = mean(timings)

    print("\n" + "=" * 50)
    print(f"Average Latency: {avg_latency:.2f} ms")


if __name__ == "__main__":
    main()
