import pandas as pd
import install scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def explain_text(text):
    text_lower = text.lower()

    urgency_words = ["now", "today", "limited", "last chance", "hurry", "instantly"]
    reward_words = ["win big", "huge rewards", "easy money", "massive jackpots", "double your winnings", "cash prizes"]
    responsibility_words = ["gamble responsibly", "terms apply", "18+ only", "responsible gambling", "terms and conditions"]

    reasons = []

    for word in urgency_words:
        if word in text_lower:
            reasons.append(f"urgency cue detected: {word}")

    for word in reward_words:
        if word in text_lower:
            reasons.append(f"reward exaggeration detected: {word}")

    has_responsibility = any(word in text_lower for word in responsibility_words)
    if not has_responsibility:
        reasons.append("no responsibility cue detected")

    if not reasons:
        reasons.append("neutral or informational language")

    return reasons


# Load dataset
df = pd.read_csv("ads_dataset.csv")

# Clean text
df["clean_text"] = df["text"].apply(clean_text)

# Features and labels
X = df["clean_text"]
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test, raw_train, raw_test = train_test_split(
    X, y, df["text"], test_size=0.3, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Baseline model
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
nb_preds = nb_model.predict(X_test_vec)

# Proposed model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_vec, y_train)
lr_preds = lr_model.predict(X_test_vec)

# Results
print("===== Naive Bayes =====")
print("Accuracy:", accuracy_score(y_test, nb_preds))
print("F1 Score:", f1_score(y_test, nb_preds, pos_label="manipulative"))

print("\n===== Logistic Regression =====")
print("Accuracy:", accuracy_score(y_test, lr_preds))
print("F1 Score:", f1_score(y_test, lr_preds, pos_label="manipulative"))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, lr_preds))

print("\nSample Predictions:\n")

for text, pred in zip(raw_test.iloc[:5], lr_preds[:5]):
    print("Text:", text)
    print("Prediction:", pred)
    print("Explanation:")
    for r in explain_text(text):
        print("-", r)
    print()