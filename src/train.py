from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib #used for saving the trained model
import pandas as pd
from sentence_transformers import SentenceTransformer #loading pretrained transformer model
from sklearn.linear_model import LogisticRegression #classifier for intent prediction
from sklearn.metrics import classification_report # used for evaluation of precision, recall, F1 score, Accuracy

BASE_DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR.parent / "models"

MODEL_DIR.mkdir(exist_ok=True)

# loading train and test data set
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df: pd.DataFrame = pd.read_csv(DATA_DIR / "train.csv")
    test_df: pd.DataFrame = pd.read_csv(DATA_DIR / "test.csv")

    return train_df, test_df

# convert text to embeddings
def generate_embeddings(
    model: SentenceTransformer,
    texts: list[str],
) -> list[list[float]]:
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    return embeddings.tolist()

# training of the model
def main() -> None:

    print("Loading dataset...")

    train_df, test_df = load_data()

    print("Loading embedding model...")

    embedding_model: SentenceTransformer = SentenceTransformer( # MiniLM is a light weight and fast test embedding model
        "sentence-transformers/all-MiniLM-L6-v2" # generate upto 384 dimensional vectors
    )

    print("Generating embeddings...")

    x_train = generate_embeddings(   # Generating train embedding
        embedding_model,
        train_df["text"].tolist(),
    )

    x_test = generate_embeddings(    # Generating test embedding
        embedding_model,
        test_df["text"].tolist(),
    )

    y_train = train_df["label"].tolist()  # Extracting test and train labels
    y_test = test_df["label"].tolist()

    print("Training classifier...")

    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced", # balancing weight prevents large class dominating
    )

    classifier.fit(x_train, y_train) #learning happens here embedding to intent labels

    print("Evaluating model...")

    predictions = classifier.predict(x_test)

    print(
        classification_report(  # classification report
            y_test,
            predictions,
        )
    )

    print("Saving models...")

    joblib.dump(
        classifier,
        MODEL_DIR / "classifier.joblib", # Save trained Logistic Regression model (Classifier)
    )

    embedding_model.save(
        str(MODEL_DIR / "embedding_model") # Save the MiniLM embedding model
    )

    print("Training completed!")


if __name__ == "__main__":
    main()