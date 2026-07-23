import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer

data = pd.read_csv("data/processed/features.csv")
classes = sorted(data["label"].unique())
labels = data["label"].map({label: i for i, label in enumerate(classes)})
features = ["word_count", "i_rate", "absolutist_rate", "grandiosity_rate", "energy_rate", "anxiety_rate", "death_rate", "external_rate", "negative_rate"]
tokenizer = AutoTokenizer.from_pretrained("models/mentalbert_model")
bert = AutoModel.from_pretrained("models/mentalbert_model")
bert.eval()

def embed(text):
    tokens = tokenizer(str(text), max_length=256, truncation=True, padding="max_length", return_tensors="pt")
    with torch.no_grad():
        return bert(**tokens).last_hidden_state[:, 0].numpy()[0]

vectors = np.array([np.concatenate([embed(row["full_text"]), row[features].to_numpy(dtype=float)]) for _, row in data.iterrows()])
x_train, x_test, y_train, y_test = train_test_split(vectors, labels, test_size=0.2, random_state=42, stratify=labels)
model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
model.fit(x_train, y_train)
print(classification_report(y_test, model.predict(x_test), target_names=classes))
joblib.dump(model, "models/hybrid_model.pkl")
