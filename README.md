# Spam Email Classifier 📨

A simple machine learning project in Python to classify messages as **spam** or **ham** (not spam).  
It uses **TF–IDF** for text features and a **Naive Bayes** classifier from scikit-learn, plus a small **Tkinter GUI**.

---

## Features

- Train a spam classifier on a CSV dataset (`label`, `text`)
- View evaluation metrics in the console:
  - Accuracy
  - Confusion matrix
  - Classification report (precision, recall, F1-score)
- Simple GUI app:
  - Type or paste a message
  - Click **“Check Spam”**
  - See prediction + spam probability

---

## Project Structure

```text
Spam-Email-Classifier/
│
├─ data/
│   └─ spam.csv           # sample dataset (label,text)
│
├─ train_spam_model.py    # trains model, prints metrics, saves .joblib files
├─ spam_gui.py            # Tkinter GUI that uses the trained model
├─ requirements.txt       # Python dependencies
└─ .gitignore

------------------------------------------------------------------------------
Dataset Format
label,text
ham,"Hey, are we still meeting tonight?"
spam,"Limited time offer!!! Click here for a 70% discount now!"

label must be either spam or ham
text is the message/email content

----------------------------------------------------------------------------
Running the GUI
After training:
python spam_gui.py

You’ll see a Spam Email Classifier window:
Type or paste a message
Click Check Spam
The app shows:
--------------------------------------------------------------------------------
Requirements
Main Python packages:
numpy
pandas
scikit-learn
joblib
-----------------------------------------------------------------------------

Future Improvements
Use a larger real-world dataset (e.g., SMS Spam Collection)
Add support for SVM and compare with Naive Bayes
Visualize metrics (ROC curve, precision–recall curve)
Improve GUI (colors, history of predictions, logs)
