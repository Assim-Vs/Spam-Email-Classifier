"""
spam_classifier.py

Train and evaluate spam classifiers using Naive Bayes and SVM.
Handles various CSV formats (like SMS Spam Collection) and prints debug info.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

# ============================================================
# 1. Load Dataset (robust with debug info)
# ============================================================

def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Loads the dataset from csv_path and tries to detect
    which columns are labels and which are text.
    Works with common spam datasets like SMS Spam Collection.
    """

    if not os.path.exists(csv_path):
        print(f"ERROR: Dataset file not found at: {csv_path}")
        sys.exit(1)

    # Try reading the CSV
    df = pd.read_csv(csv_path, encoding="latin-1")

    print("\n[DEBUG] Raw columns:", df.columns.tolist())
    print("[DEBUG] First 5 rows:\n", df.head())

    # Case 1: Already has 'label' and 'text'
    if "label" in df.columns and "text" in df.columns:
        pass

    # Case 2: SMS Spam Collection format: v1 (label), v2 (text)
    elif "v1" in df.columns and "v2" in df.columns:
        df = df[["v1", "v2"]]
        df.columns = ["label", "text"]

    else:
        # Try to guess label and text columns
        possible_label_cols = [
            c for c in df.columns
            if c.lower() in ["label", "class", "category", "spam", "target", "is_spam"]
        ]
        possible_text_cols = [
            c for c in df.columns
            if c.lower() in ["text", "message", "email", "sms", "content", "body"]
        ]

        if possible_label_cols and possible_text_cols:
            df = df[[possible_label_cols[0], possible_text_cols[0]]]
            df.columns = ["label", "text"]
        else:
            print("\nERROR: Could not automatically detect label/text columns.")
            print("Columns found:", df.columns.tolist())
            print("Hint: make sure your CSV has label and text-like columns "
                  "(e.g., 'v1'/'v2', 'label'/'text', 'class'/'message').")
            sys.exit(1)

    # Now we should definitely have 'label' and 'text'
    df = df[["label", "text"]].copy()

    # Drop missing rows
    df.dropna(subset=["label", "text"], inplace=True)

    # Normalize labels to lowercase strings
    df["label"] = df["label"].astype(str).str.lower().str.strip()

    print("\n[DEBUG] Unique labels BEFORE mapping:", df["label"].unique())

    # Map various label styles to 'spam' / 'ham'
    spam_like = {"spam", "1", "yes", "true", "junk", "spam message"}
    ham_like = {"ham", "0", "no", "false", "normal", "legit", "not spam"}

    def normalize_label(x: str) -> str:
        if x in spam_like:
            return "spam"
        if x in ham_like:
            return "ham"
        return x  # unknown labels kept as is for now

    df["label"] = df["label"].apply(normalize_label)

    print("[DEBUG] Unique labels AFTER mapping:", df["label"].unique())

    # Keep only spam/ham
    df = df[df["label"].isin(["spam", "ham"])]

    print(f"[DEBUG] Rows after filtering spam/ham: {len(df)}")

    if len(df) == 0:
        print("\nERROR: After filtering, dataset has 0 rows.")
        print("Check your CSV file: label values may not match 'spam'/ 'ham' or known variants.")
        print("Try opening spam.csv and see exactly what is inside the label column.")
        sys.exit(1)

    return df


# ============================================================
# 2. Prepare Data (split into train/test)
# ============================================================

def prepare_data(df: pd.DataFrame):
    """
    Encodes labels and splits into train/test.
    """

    # Encode labels: spam=1, ham=0
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

    if df["label_num"].isna().any():
        print("\n[DEBUG] Some labels could not be mapped to 0/1:")
        print(df[df["label_num"].isna()]["label"].value_counts())
        print("Exiting because of unmapped labels.")
        sys.exit(1)

    X = df["text"].values
    y = df["label_num"].values

    if len(X) == 0:
        print("ERROR: No samples available after preprocessing.")
        sys.exit(1)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


# ============================================================
# 3. Feature Extraction (TF-IDF)
# ============================================================

def build_vectorizer():
    """
    Create a TF-IDF vectorizer with reasonable defaults for spam detection.
    """
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",   # remove common English words
        ngram_range=(1, 2),     # unigrams + bigrams
        min_df=2,               # ignore very rare terms
    )
    return vectorizer


# ============================================================
# 4. Train Models
# ============================================================

def train_naive_bayes(X_train_tfidf, y_train):
    model_nb = MultinomialNB()
    model_nb.fit(X_train_tfidf, y_train)
    return model_nb


def train_svm(X_train_tfidf, y_train):
    # Linear SVM - good for text classification
    model_svm = LinearSVC()
    model_svm.fit(X_train_tfidf, y_train)
    return model_svm


# ============================================================
# 5. Evaluation Helpers
# ============================================================

def evaluate_model(name: str, model, X_test_tfidf, y_test):
    y_pred = model.predict(X_test_tfidf)

    print(f"\n===== {name} Evaluation =====")
    print("Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred)))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))


# ============================================================
# 6. Interactive Prediction
# ============================================================

def predict_single_message_nb(vectorizer, model_nb, message: str):
    vec = vectorizer.transform([message])
    pred = model_nb.predict(vec)[0]

    # Naive Bayes supports predict_proba
    prob_spam = model_nb.predict_proba(vec)[0][1]

    label = "SPAM" if pred == 1 else "HAM"
    print(f"\n[Naive Bayes] Prediction: {label} (spam probability = {prob_spam:.3f})")


def predict_single_message_svm(vectorizer, model_svm, message: str):
    vec = vectorizer.transform([message])
    pred = model_svm.predict(vec)[0]

    label = "SPAM" if pred == 1 else "HAM"
    print(f"[SVM] Prediction: {label}")


# ============================================================
# 7. Save & Load Models
# ============================================================

def save_models(vectorizer, model_nb, model_svm,
                vec_path="vectorizer.joblib",
                nb_path="naive_bayes_model.joblib",
                svm_path="svm_model.joblib"):

    joblib.dump(vectorizer, vec_path)
    joblib.dump(model_nb, nb_path)
    joblib.dump(model_svm, svm_path)

    print(f"\nModels saved to:\n  {vec_path}\n  {nb_path}\n  {svm_path}")


def load_models(vec_path, nb_path, svm_path):
    vectorizer = joblib.load(vec_path)
    model_nb = joblib.load(nb_path)
    model_svm = joblib.load(svm_path)
    return vectorizer, model_nb, model_svm


# ============================================================
# 8. Main Script
# ============================================================

def main():
    # Path to dataset (adjust if your path is different)
    data_path = os.path.join("data", "spam.csv")

    print("Loading dataset...")
    df = load_dataset(data_path)
    print(f"Dataset loaded: {len(df)} rows\n")

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Vectorizer
    vectorizer = build_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train models
    print("\nTraining Naive Bayes...")
    model_nb = train_naive_bayes(X_train_tfidf, y_train)

    print("Training SVM...")
    model_svm = train_svm(X_train_tfidf, y_train)

    # Evaluate models
    evaluate_model("Naive Bayes", model_nb, X_test_tfidf, y_test)
    evaluate_model("SVM", model_svm, X_test_tfidf, y_test)

    # Test some sample messages
    sample_spam = "Congratulations! You won a free iPhone. Click here to claim your prize now!"
    sample_ham = "Hi John, are we still on for the meeting tomorrow at 10 AM?"

    print("\n--- Sample Predictions ---")
    print("Message 1:", sample_spam)
    predict_single_message_nb(vectorizer, model_nb, sample_spam)
    predict_single_message_svm(vectorizer, model_svm, sample_spam)

    print("\nMessage 2:", sample_ham)
    predict_single_message_nb(vectorizer, model_nb, sample_ham)
    predict_single_message_svm(vectorizer, model_svm, sample_ham)

    # Save models
    save_models(vectorizer, model_nb, model_svm)


if __name__ == "__main__":
    main()

