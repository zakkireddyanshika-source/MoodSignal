import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
import os

df = pd.read_csv("data/processed/clean.csv")
print(f"loaded {len(df)} rows")

texts = df["full_text"]
labels = df["label"]

text_train, text_test, labels_train, labels_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)
print(f"training: {len(text_train)} posts, test: {len(text_test)} posts")

vectorizer = TfidfVectorizer(stop_words="english")
train_numbers = vectorizer.fit_transform(text_train)
test_numbers = vectorizer.transform(text_test)
print("converted text to numbers")

smote = SMOTE(random_state=42)
train_balanced, labels_balanced = smote.fit_resample(train_numbers, labels_train)
print(f"after smote: {pd.Series(labels_balanced).value_counts().to_dict()}")

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(train_balanced, labels_balanced)
print("model trained")

predictions = model.predict(test_numbers)
accuracy = accuracy_score(labels_test, predictions)
print(f"overall accuracy: {accuracy:.1%}")

print(classification_report(labels_test, predictions))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/baseline_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
print("saved model and vectorizer")
