# spam_gui.py

import joblib
import tkinter as tk
from tkinter import messagebox

# -------- Load trained model and vectorizer --------

try:
    vectorizer = joblib.load("vectorizer.joblib")
    model_nb = joblib.load("spam_nb_model.joblib")
except Exception as e:
    raise RuntimeError(
        "Error loading model files. Make sure 'vectorizer.joblib' and "
        "'spam_nb_model.joblib' exist in this folder.\n"
        "You must run train_spam_model.py first."
    ) from e

def predict_spam(message: str):
    """
    Returns: (label, probability_of_spam)
    label: 'SPAM' or 'HAM'
    """
    vec = vectorizer.transform([message])
    pred = model_nb.predict(vec)[0]
    prob_spam = model_nb.predict_proba(vec)[0][1]
    label = "SPAM" if pred == 1 else "HAM"
    return label, prob_spam

# -------- GUI code --------

def on_check():
    text = text_box.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Empty input", "Please type or paste a message first.")
        return

    label, prob = predict_spam(text)
    result_var.set(f"{label} (spam probability: {prob:.3f})")

def create_window():
    window = tk.Tk()
    window.title("Spam Email Classifier")

    # Set window size (optional)
    window.geometry("750x450")

    title_label = tk.Label(
        window,
        text="Spam Email Classifier",
        font=("Arial", 18, "bold")
    )
    title_label.pack(pady=10)

    instruction_label = tk.Label(
        window,
        text="Type or paste an email/message below and click 'Check Spam':",
        font=("Arial", 11)
    )
    instruction_label.pack()

    global text_box
    text_box = tk.Text(window, height=12, width=80, font=("Consolas", 10))
    text_box.pack(pady=10)

    check_button = tk.Button(
        window,
        text="Check Spam",
        font=("Arial", 12, "bold"),
        command=on_check
    )
    check_button.pack(pady=5)

    global result_var
    result_var = tk.StringVar()
    result_label = tk.Label(
        window,
        textvariable=result_var,
        font=("Arial", 14, "bold")
    )
    result_label.pack(pady=10)

    window.mainloop()

if __name__ == "__main__":
    create_window()

