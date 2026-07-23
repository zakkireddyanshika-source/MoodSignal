import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# load the same test set every model was evaluated on
data = pd.read_csv("data/processed/clean.csv")
classes = sorted(data["label"].unique())
labels = data["label"].map({label: i for i, label in enumerate(classes)})
x_train, x_test, y_train, y_test = train_test_split(
    data["full_text"], labels, test_size=0.2, random_state=42, stratify=labels
)

results = {}

# baseline - tfidf + logistic regression
baseline_model = joblib.load("models/baseline_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
preds = baseline_model.predict(vectorizer.transform(x_test))
results["Baseline"] = {
    "accuracy": accuracy_score(y_test, preds),
    "f1_per_class": f1_score(y_test, preds, average=None)
}
print("baseline done")

# mentalbert
tokenizer = AutoTokenizer.from_pretrained("models/mentalbert_model")
bert_model = AutoModelForSequenceClassification.from_pretrained("models/mentalbert_model")
bert_model.eval()
tokens = tokenizer(list(x_test), max_length=256, truncation=True, padding="max_length", return_tensors="pt")
with torch.no_grad():
    logits = bert_model(**tokens).logits
bert_preds = logits.argmax(1).numpy()
results["MentalBERT"] = {
    "accuracy": accuracy_score(y_test, bert_preds),
    "f1_per_class": f1_score(y_test, bert_preds, average=None)
}
print("mentalbert done")

# print the table
print("\nmodel comparison:")
for name, r in results.items():
    print(f"{name}: accuracy {r['accuracy']:.1%}, f1 per class {r['f1_per_class'].round(2)}")

# chart 1 - overall accuracy bar chart
plt.figure(figsize=(6, 4))
names = list(results.keys())
accs = [results[n]["accuracy"] * 100 for n in names]
plt.bar(names, accs, color=["#6b7280", "#10b981"])
plt.ylabel("accuracy (%)")
plt.title("model comparison - overall accuracy")
plt.ylim(0, 100)
for i, v in enumerate(accs):
    plt.text(i, v + 1, f"{v:.1f}%", ha="center")
plt.tight_layout()
plt.savefig("outputs/accuracy_comparison.png", dpi=150)
print("saved outputs/accuracy_comparison.png")

# chart 2 - f1 per class grouped bar chart
plt.figure(figsize=(7, 4))
x = np.arange(len(classes))
width = 0.35
for i, (name, r) in enumerate(results.items()):
    plt.bar(x + i * width, r["f1_per_class"], width, label=name)
plt.xticks(x + width / 2, classes)
plt.ylabel("f1 score")
plt.title("f1 score per class")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/f1_per_class.png", dpi=150)
print("saved outputs/f1_per_class.png")