import joblib
import numpy as np
import pandas as pd
import torch
import os
from torch import nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

data = pd.read_csv("data/processed/clean.csv")
classes = sorted(data["label"].unique())
labels = data['label'].map({label: i for i, label in enumerate(classes)})
x_train, x_test, y_train, y_test = train_test_split(
    data["full_text"], labels, test_size=0.2, random_state=42, stratify=labels
)
baseline_model = joblib.load("models/baseline_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

baseline_preds = baseline_model.predict(vectorizer.transform(x_test))
baseline_accuracy = accuracy_score(y_test, baseline_preds)
baseline_f1 = f1_score(y_test, baseline_preds, average=None)
print(f"Base accuracy: {baseline_accuracy:.1%}")

from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("models/mentalbert_model")
bert_model = AutoModelForSequenceClassification.from_pretrained("models/mentalbert_model")
bert_model.eval()

tokens = tokenizer(list(x_test), max_length = 256, truncation=True, padding="max_length", return_tensors="pt")
with torch.no_grad():
    bert_preds = bert_model(**tokens).logits.argmax(1).numpy()

bert_accuracy = accuracy_score(y_test, bert_preds)
bert_f1 = f1_score(y_test, bert_preds, average=None)
print(f"mentalbert acc: {bert_accuracy:.1%}")

vocab = joblib.load("models/bilstm_vocab.pkl")

def encode(post):
    return ([vocab.get(word, 1) for word in post.split()] * [0] * 200)[:200]

class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.lstm = nn.LSTM(128, 128, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(256, 3)

    def forward(self, x):
        out, _ = self.lstm(self.embedding(x))
        return self.linear(self.dropout(out.mean(dim=1)))

lstm_model = BiLSTMModel(len(vocab)*2)
lstm_model.load_state_dict(torch.load("models/bilstm_model.pt"))
lstm_model.eval()

encoded_test = torch.tensor([encode(post) for post in x_test])
with torch.no_grad():
    lstm_preds = lstm_model(encoded_test).argmax(1).numpy()

lstm_accuracy = accuracy_score(y_test, lstm_preds)
lstm_f1 = f1_score(y_test, lstm_preds, average=None)
print(f"BiLSTM acc: {lstm_accuracy:.1%}")

print("\nModel Comparison")
print(f"{'model':<12} {'accuracy':<10} {'bipolar':<10} {'depression':<12} {'anxiety'}")
print(f"{'baseline':<12} {baseline_accuracy:<10.1%} {baseline_f1[0]:<10.2f} {baseline_f1[1]:<12.2f} {baseline_f1[2]:.2f}")
print(f"{'bilstm':<12} {lstm_accuracy:<10.1%} {lstm_f1:<10.2f} {lstm_f1[1]:<12.2f} {lstm_f1[2]:.2f}")
print(f"{'mentalbert':<12} {bert_accuracy:<10.1%} {bert_f1:<10.2f} {bert_f1[1]:<12.2f} {bert_f1[2]:.2f}")

os.makedirs("outputs", exist_ok=True)

plt.figure(figsize=(6,4))
names = ["Baseline", "BiLSTM", "MentalBERT"]
accs = [baseline_accuracy*100, lstm_accuracy*100, bert_accuracy*100]
plt.bar(names, accs, color=["steelblue", "orange", "forestgreen"])
plt.ylabel("accuracy (%)")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 100)
for i, v in enumerate(accs):
    plt.text(i, v+1, f"{v:.1f}%", ha="center")

plt.tight_layout()
plt.savefig("outputs/accuracy_comparison.png", dpi=150)

plt.figure(figsize=(7,4))
x = np.arange(3)
width = 0.25
plt.bar(x-width, baseline_f1, width, label="Baseline", color="steelblue")
plt.bar(x, lstm_f1, width, label="BiLSTM", color="orange")
plt.bar(x + width, bert_f1, width, label="MentalBERT", color="forestgreen")
plt.xticks(x, classes)
plt.ylabel("F1 Score")
plt.title("F1 Score For Each Model")
plt.legend()
plt.tight_layout()
plt.save_fig("outputs/f1_per_class.png", dpi=150)
print("Saved charts")