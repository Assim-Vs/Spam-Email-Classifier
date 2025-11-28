# train_spam_model.py

import os
import sys
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

DATA_PATH = os.path.join("data", "spam.csv")


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load a CSV file with columns: label, text
    Labels must be 'spam' or 'ham'.
    """
    if not os.path.exists(csv_path):
        print(f"ERROR: Dataset file not found at: {csv_path}")
        sys.exit(1)

    try:
        # Simple CSV with comma separator
        df = pd.read_csv(csv_path, encoding="latin-1")
    except Exception as e:
        print("ERROR reading CSV:", e)
        sys.exit(1)

    print("[DEBUG] Columns:", df.columns.tolist())
    print("[DEBUG] Head:\n", df.head())

    # Check columns
    if "label" not in df.columns or "text" not in df.columns:
        print("ERROR: CSV must have 'label' and 'text' columns.")
        sys.exit(1)

    df = df[["label", "text"]].copy()
    df.dropna(subset=["label", "text"], inplace=True)

    # Normalize labels
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    print("[DEBUG] Unique labels:", df["label"].unique())

    # Keep only spam/ham
    df = df[df["label"].isin(["spam", "ham"])]

    if len(df) == 0:
        print("ERROR: No valid rows with labels 'spam' or 'ham'.")
        sys.exit(1)

    return df


def main():
    print("Loading dataset...")
    df = load_dataset(DATA_PATH)
    print(f"Rows in dataset: {len(df)}")

    # Encode labels: spam = 1, ham = 0
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

    X = df["text"].values
    y = df["label_num"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train Naive Bayes
    model_nb = MultinomialNB()
    model_nb.fit(X_train_tfidf, y_train)

    # ---------- Evaluation ----------
    y_pred = model_nb.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)

    print("\n===== Evaluation on Test Set =====")
    print(f"Accuracy: {acc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix (rows = true, cols = predicted):")
    print("          Pred HAM   Pred SPAM")
    print(f"True HAM    {cm[0, 0]:>5}       {cm[0, 1]:>5}")
    print(f"True SPAM   {cm[1, 0]:>5}       {cm[1, 1]:>5}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    # ---------- Save model + vectorizer ----------
    joblib.dump(vectorizer, "vectorizer.joblib")
    joblib.dump(model_nb, "spam_nb_model.joblib")
    print("\nSaved files:")
    print("  vectorizer.joblib")
    print("  spam_nb_model.joblib")


if __name__ == "__main__":
    main()
