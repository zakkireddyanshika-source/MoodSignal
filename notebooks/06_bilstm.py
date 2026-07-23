# step 6 - bilstm model, reads each post forwards and backwards to understand word order

# pandas for loading and working with the data table
import pandas as pd
# torch is the main deep learning library we build the model with
import torch
# tool to check how good the models guesses are after training
from sklearn.metrics import classification_report
# splits data into train and test sets
from sklearn.model_selection import train_test_split
# nn has the building blocks for neural networks (embedding, lstm, linear layers)
from torch import nn
# dataloader batches up our data, dataset is the format pytorch wants our data in
from torch.utils.data import DataLoader, Dataset

# load the cleaned data, just post text and label, nothing fancy
data = pd.read_csv("data/processed/clean.csv")
# grab the text column, fill blanks so nothing breaks, make sure its all strings
text = data["full_text"].fillna("").astype(str)
# turn the label words (bipolar, depression, anxiety) into numbers 0, 1, 2
labels = pd.Categorical(data["label"]).codes

# build one big list of every word used across every single post
all_words = []
for post in text:
    for word in post.split():
        all_words.append(word)

# count how many times each word shows up, keep only the top 19998 most common
word_counts = pd.Series(all_words).value_counts()
top_words = word_counts.head(19998)

# build the vocabulary, a dictionary that turns each word into a number
# start at 2 because 0 means padding and 1 means unknown word
vocab = {}
next_number = 2
for word in top_words.index:
    vocab[word] = next_number
    next_number += 1

# turns one post into a list of exactly 200 numbers so every post is the same length
def encode(post):
    # look up each words number, use 1 if we have never seen the word before
    numbers = []
    for word in post.split():
        numbers.append(vocab.get(word, 1))
    # pad the end with zeros in case the post is shorter than 200 words
    numbers = numbers + [0] * 200
    # cut it down to 200 numbers, this also handles posts that were too short
    return numbers[:200]

# wraps our data up in the format pytorch wants for training
class PostDataset(Dataset):
    # runs once when we build the dataset, encodes every post up front
    def __init__(self, posts, labels):
        self.posts = [encode(post) for post in posts]
        self.labels = list(labels)

    # tells pytorch how many posts we have
    def __len__(self):
        return len(self.labels)

    # gives pytorch one post and its label when it asks for a given index
    def __getitem__(self, index):
        return torch.tensor(self.posts[index]), torch.tensor(self.labels[index], dtype=torch.long)

# the actual bilstm model
class BiLSTMModel(nn.Module):
    # sets up all the layers of the model
    def __init__(self, vocab_size):
        super().__init__()
        # turns each word number into a 128 number vector that captures meaning
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        # reads the post forwards and backwards, 128 numbers each direction, 256 total
        self.lstm = nn.LSTM(128, 128, bidirectional=True, batch_first=True)
        # randomly turns off 30% of neurons while training so it doesnt just memorize
        self.dropout = nn.Dropout(0.3)
        # turns the 256 lstm numbers into 3 scores, one per class
        self.linear = nn.Linear(256, 3)

    # runs the model on a batch of posts
    def forward(self, x):
        # turn word numbers into embeddings, then read them with the lstm
        out, _ = self.lstm(self.embedding(x))
        # average across every word in the post instead of just grabbing the last one
        # (grabbing the last one was our old padding bug, it read zeros instead of real words)
        averaged = out.mean(dim=1)
        # drop some neurons then turn what is left into 3 class scores
        return self.linear(self.dropout(averaged))

# split into 80% training, 20% test, same random_state as every other file so its the same split
x_train, x_test, y_train, y_test = train_test_split(text, labels, test_size=0.2, random_state=42, stratify=labels)

# wrap train and test sets in our dataset class, feed the model 32 posts at a time
train_loader = DataLoader(PostDataset(x_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(PostDataset(x_test, y_test), batch_size=32)

# build the model, +2 leaves room for the padding and unknown word slots
model = BiLSTMModel(len(vocab) + 2)
# cross entropy is the standard loss function for classifying into multiple categories
loss_fn = nn.CrossEntropyLoss()
# adam is the optimizer that updates the model weights as it learns
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train for 10 rounds through the whole training set
for epoch in range(10):
    # put the model in training mode, this turns dropout on
    model.train()
    total_loss = 0
    # go through the training data 32 posts at a time
    for x, y in train_loader:
        # clear out gradients left over from the last batch
        optimizer.zero_grad()
        # run the model and see how wrong its guesses were
        value = loss_fn(model(x), y)
        # figure out how to adjust the weights to do better next time
        value.backward()
        # actually update the weights
        optimizer.step()
        total_loss += value.item()
    # print how training is going after each epoch
    print(f"epoch {epoch + 1} done, average loss was {total_loss / len(train_loader):.4f}")

# put the model in eval mode, this turns dropout off since we are done training
model.eval()
predictions = []
# no need to track gradients here, we are just testing not training
with torch.no_grad():
    for x, _ in test_loader:
        predictions.extend(model(x).argmax(1).tolist())

# print precision, recall and f1 for each class on the test set
print(classification_report(y_test, predictions))

# save the trained model so we can load it again later without retraining
torch.save(model.state_dict(), "models/bilstm_model.pt")
