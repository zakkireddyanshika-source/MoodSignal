import joblib
import numpy as np
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

os.makedirs("outputs", exist_ok=True)

data = pd.read_csv("data/processed/clean.csv")
classes = sorted(data["label"].unique())
labels = data["label"].map({label: i for i, label in enumerate(classes)})
x_train, x_test, y_train, y_test = train_test_split(
    data["full_text"], labels, test_size=0.2, random_state=42, stratify=labels
)

plt.figure(figsize=(6, 4))
data["label"].value_counts().sort_index().plot(kind="bar", color=["steelblue", "orange", "forestgreen"])
plt.title("Posts per Condition")
plt.ylabel("count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/class_distribution.png", dpi=150)
print("saved class_distribution.png")

baseline = joblib.load("models/baseline_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
label_to_number = {label: i for i, label in enumerate(classes)}
baseline_preds = [label_to_number[p] for p in baseline.predict(vectorizer.transform(x_test))]

plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, baseline_preds), annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.title("Baseline Confusion Matrix")
plt.ylabel("actual")
plt.xlabel("predicted")
plt.tight_layout()
plt.savefig("outputs/confusion_baseline.png", dpi=150)
print("saved confusion_baseline.png")

from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("models/mentalbert_model")
bert_model = AutoModelForSequenceClassification.from_pretrained("models/mentalbert_model")
bert_model.eval()
tokens = tokenizer(list(x_test), max_length=256, truncation=True, padding="max_length", return_tensors="pt")
with torch.no_grad():
    bert_preds = bert_model(**tokens).logits.argmax(1).numpy()

plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, bert_preds), annot=True, fmt="d", cmap="Greens",
            xticklabels=classes, yticklabels=classes)
plt.title("MentalBERT Confusion Matrix")
plt.ylabel("actual")
plt.xlabel("predicted")
plt.tight_layout()
plt.savefig("outputs/confusion_mentalbert.png", dpi=150)
print("saved confusion_mentalbert.png")

plt.figure(figsize=(5, 3))
epochs = [1, 2, 3]
losses = [0.5565, 0.2615, 0.1321]
plt.plot(epochs, losses, marker="o", color="forestgreen", linewidth=2)
plt.title("MentalBERT Training Loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.xticks(epochs)
plt.tight_layout()
plt.savefig("outputs/mentalbert_loss.png", dpi=150)
print("saved mentalbert_loss.png")

features = pd.read_csv("data/processed/features.csv")
feature_cols = ["i_rate", "absolutist_rate", "grandiosity_rate", "energy_rate",
                "anxiety_rate", "death_rate", "external_rate", "negative_rate"]
averages = features.groupby("label")[feature_cols].mean()

plt.figure(figsize=(10, 5))
averages.T.plot(kind="bar", figsize=(10, 5))
plt.title("Average Feature Values by Condition")
plt.ylabel("rate per word")
plt.xticks(rotation=45, ha="right")
plt.legend(title="condition")
plt.tight_layout()
plt.savefig("outputs/feature_comparison.png", dpi=150)
print("saved feature_comparison.png")

print("\nall charts saved to outputs/")