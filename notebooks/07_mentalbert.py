import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

data = pd.read_csv("data/processed/clean.csv")
classes = sorted(data["label"].unique())
labels = data["label"].map({label: i for i, label in enumerate(classes)})
x_train, x_test, y_train, y_test = train_test_split(data["full_text"], labels, test_size=0.2, random_state=42, stratify=labels)
tokenizer = AutoTokenizer.from_pretrained("mental/mental-bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("mental/mental-bert-base-uncased", num_labels=3)

class BertPostDataset(Dataset):
    def __init__(self, posts, labels):
        self.posts = list(posts)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        batch = tokenizer(str(self.posts[index]), max_length=256, truncation=True, padding="max_length", return_tensors="pt")
        batch = {key: value.squeeze(0) for key, value in batch.items()}
        batch["labels"] = torch.tensor(self.labels[index])
        return batch

train_loader = DataLoader(BertPostDataset(x_train, y_train), batch_size=16, shuffle=True)
test_loader = DataLoader(BertPostDataset(x_test, y_test), batch_size=32)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

for epoch in range(3):
    model.train()
    loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        value = model(**batch).loss
        value.backward()
        optimizer.step()
        loss += value.item()
    print(f"Epoch {epoch + 1}: {loss / len(train_loader):.4f}")

model.eval()
predictions = []
targets = []
with torch.no_grad():
    for batch in test_loader:
        targets.extend(batch.pop("labels").tolist())
        predictions.extend(model(**batch).logits.argmax(1).tolist())
print(classification_report(targets, predictions, target_names=classes))
model.save_pretrained("models/mentalbert_model")
tokenizer.save_pretrained("models/mentalbert_model")
