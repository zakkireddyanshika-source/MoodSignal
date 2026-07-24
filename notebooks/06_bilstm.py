import joblib
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data = pd.read_csv("data/processed/clean.csv")
text = data["full_text"]
labels = pd.Categorical(data["label"]).codes

words = pd.Series(" ".join(text).split()).value_counts().head(19998)
vocab = {word: i + 2 for i, word in enumerate(words.index)}

def encode(post):
    return ([vocab.get(word, 1) for word in post.split()] + [0] * 200)[:200]

class PostDataset(Dataset):
    def __init__(self, posts, labels):
        self.posts = [encode(post) for post in posts]
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.tensor(self.posts[index]), torch.tensor(self.labels[index])

class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.lstm = nn.LSTM(128, 128, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(256, 3)

    def forward(self, x):
        out, _ = self.lstm(self.embedding(x))
        # average across all words, not just the last one which is usually padding
        return self.linear(self.dropout(out.mean(dim=1)))

x_train, x_test, y_train, y_test = train_test_split(text, labels, test_size=0.2, random_state=42, stratify=labels)
train_loader = DataLoader(PostDataset(x_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(PostDataset(x_test, y_test), batch_size=32)

model = BiLSTMModel(len(vocab) + 2)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    total = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item()
    print(f"epoch {epoch + 1}: {total / len(train_loader):.4f}")

model.eval()
predictions = []
with torch.no_grad():
    for x, _ in test_loader:
        predictions.extend(model(x).argmax(1).tolist())

print(classification_report(y_test, predictions))
torch.save(model.state_dict(), "models/bilstm_model.pt")
joblib.dump(vocab, "models/bilstm_vocab.pkl")