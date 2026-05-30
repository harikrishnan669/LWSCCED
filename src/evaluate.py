from __future__ import annotations

from pathlib import Path
from typing import Any
import joblib # loads saved classifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer # loads MiniLM embedding model
from sklearn.metrics import ( # used for evaluation
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)
from utils import normalize_text # clean ASR mistakes before prediction

# paths for accessing files

DATA_DIR: Path = Path("data")
MODEL_DIR: Path = Path("../models") # contains the classifier and embedding model
OUTPUT_DIR: Path = Path("../evaluation")

OUTPUT_DIR.mkdir(exist_ok=True)

CONFIDENCE_THRESHOLD: float = 0.40

# loads MiniLM
embedding_model: SentenceTransformer = SentenceTransformer(
    str(MODEL_DIR / "embedding_model")
)

# loads the classifier
classifier: Any = joblib.load(
    MODEL_DIR / "classifier.joblib"
)

# load test data
test_df: pd.DataFrame = pd.read_csv(
    DATA_DIR / "test.csv"
)

texts: list[str] = test_df["text"].tolist()
true_labels: list[str] = test_df["label"].tolist()


# Predict command label with OOS rejection.

def predict(text: str) -> str:

    normalized_text: str = normalize_text(text)

    embedding = embedding_model.encode(
        [normalized_text],
        convert_to_numpy=True,
    )

    probabilities = classifier.predict_proba(embedding)[0]

    max_index: int = int(np.argmax(probabilities))

    confidence: float = float(probabilities[max_index])

    predicted_label: str = classifier.classes_[max_index]

    if confidence < CONFIDENCE_THRESHOLD:
        return "out_of_scope"

    return predicted_label


# predict label for every test sample
predictions: list[str] = [
    predict(text)
    for text in texts
]

#contains precision,recall,F1 score,support
print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report: str = classification_report(
    true_labels,
    predictions,
    digits=4,
)

print(report)

accuracy: float = accuracy_score(
    true_labels,
    predictions,
)

print(f"\nOverall Accuracy: {accuracy:.4f}")

# evaluate each clean,paraphrase,asr_noise,char_noise,filler
print("\n" + "=" * 60)
print("NOISE ROBUSTNESS ANALYSIS")
print("=" * 60)

if "source" in test_df.columns:

    noise_groups: dict[str, pd.DataFrame] = {
        source_name: subset
        for source_name, subset in test_df.groupby("source")
    }

    for source_name, subset in noise_groups.items():

        subset_texts: list[str] = subset["text"].tolist()
        subset_labels: list[str] = subset["label"].tolist()

        subset_predictions: list[str] = [
            predict(text)
            for text in subset_texts
        ]

        subset_accuracy: float = accuracy_score(
            subset_labels,
            subset_predictions,
        )

        print(
            f"{source_name:<20}"
            f" Accuracy: {subset_accuracy:.4f}"
            f" | Samples: {len(subset)}"
        )

# Failure analysis
print("\n" + "=" * 60)
print("FAILURE ANALYSIS")
print("=" * 60)

failure_count: int = 0

for text, true_label, predicted_label in zip(
    texts,
    true_labels,
    predictions,
):

    if true_label != predicted_label:

        print(f"\nInput      : {text}")
        print(f"Expected   : {true_label}")
        print(f"Predicted  : {predicted_label}")

        failure_count += 1

    if failure_count >= 20:
        break

# choose best rejection threshold
print("\n" + "=" * 60)
print("THRESHOLD ANALYSIS")
print("=" * 60)

thresholds: list[float] = [
    0.30,
    0.40,
    0.50,
]

for threshold in thresholds:

    threshold_predictions: list[str] = []

    for text in texts:

        normalized_text: str = normalize_text(text)

        embedding = embedding_model.encode(
            [normalized_text],
            convert_to_numpy=True,
        )

        probabilities = classifier.predict_proba(embedding)[0]

        max_index: int = int(np.argmax(probabilities))

        confidence: float = float(
            probabilities[max_index]
        )

        predicted_label: str = (
            classifier.classes_[max_index]
        )

        if confidence < threshold:
            predicted_label = "out_of_scope"

        threshold_predictions.append(
            predicted_label
        )

    threshold_accuracy: float = accuracy_score(
        true_labels,
        threshold_predictions,
    )

    print(
        f"Threshold {threshold:.2f}"
        f" -> Accuracy: {threshold_accuracy:.4f}"
    )

# create confusion matrix
labels: list[str] = sorted(
    list(set(true_labels))
)

cm = confusion_matrix(
    true_labels,
    predictions,
    labels=labels,
)

fig, ax = plt.subplots(figsize=(14, 14))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels,
)

disp.plot(
    ax=ax,
    xticks_rotation=90,
)

plt.title("Command Classification Confusion Matrix")

plt.tight_layout()

confusion_matrix_path: Path = (
    OUTPUT_DIR / "confusion_matrix.png"
)

plt.savefig(confusion_matrix_path)

print(f"\nConfusion matrix saved to:")
print(confusion_matrix_path.resolve())


# Measures how well unknown commands are rejected.
oos_true: int = 0
oos_correct: int = 0

valid_true: int = 0
valid_rejected: int = 0

for true_label, predicted_label in zip(
    true_labels,
    predictions,
):
    if true_label == "out_of_scope":
        oos_true += 1

        if predicted_label == "out_of_scope":
            oos_correct += 1

    else:
        valid_true += 1

        if predicted_label == "out_of_scope":
            valid_rejected += 1


oos_rejection_rate: float = (
    oos_correct / oos_true
    if oos_true > 0
    else 0.0
)

false_rejection_rate: float = (
    valid_rejected / valid_true
    if valid_true > 0
    else 0.0
)


print("\n" + "=" * 60)
print("OOS EVALUATION")
print("=" * 60)

print(f"OOS Rejection Rate     : {oos_rejection_rate:.4f}")
print(f"False Rejection Rate   : {false_rejection_rate:.4f}")


# Newly added intents
print("\n" + "=" * 60)
print("EXTENSION COMMAND ANALYSIS")
print("=" * 60)

extension_commands: list[str] = [
    "increase_brightness",
    "decrease_brightness",
    "start_vehicle",
    "stop_vehicle",
]

for command in extension_commands:

    indices: list[int] = [
        index
        for index, label in enumerate(true_labels)
        if label == command
    ]

    command_true: list[str] = [
        true_labels[index]
        for index in indices
    ]

    command_pred: list[str] = [
        predictions[index]
        for index in indices
    ]

    command_accuracy: float = accuracy_score(
        command_true,
        command_pred,
    )

    print(
        f"{command:<25}"
        f" Accuracy: {command_accuracy:.4f}"
    )

# Save the report
report_path: Path = OUTPUT_DIR / "evaluation_report.txt"

with report_path.open(
    "w",
    encoding="utf-8",
) as file:

    file.write("CLASSIFICATION REPORT\n")
    file.write("=" * 60 + "\n")
    file.write(report)

    file.write("\n")
    file.write(f"Overall Accuracy: {accuracy:.4f}\n")

    file.write("\n")
    file.write("OOS EVALUATION\n")
    file.write("=" * 60 + "\n")

    file.write(
        f"OOS Rejection Rate: "
        f"{oos_rejection_rate:.4f}\n"
    )

    file.write(
        f"False Rejection Rate: "
        f"{false_rejection_rate:.4f}\n"
    )
    file.write("\n")
    file.write("=" * 60 + "\n")
    file.write("NOISE ROBUSTNESS ANALYSIS\n")
    file.write("=" * 60 + "\n")

    if "source" in test_df.columns:

        for source_name, subset in noise_groups.items():
            subset_texts = subset["text"].tolist()
            subset_labels = subset["label"].tolist()

            subset_predictions = [
                predict(text)
                for text in subset_texts
            ]

            subset_accuracy = accuracy_score(
                subset_labels,
                subset_predictions,
            )

            file.write(
                f"{source_name:<20}"
                f" Accuracy: {subset_accuracy:.4f}"
                f" | Samples: {len(subset)}\n"
            )
print(f"\nEvaluation report saved to:")
print(report_path.resolve())
