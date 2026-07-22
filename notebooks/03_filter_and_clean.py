# step 3 - keep only bipolar depression and anxiety posts, clean the text, save it

import pandas as pd
import re
import sys
import os

# fix printing for windows
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# load the raw data
df = pd.read_csv("data/raw/data_to_be_cleansed.csv")
print(f"loaded {len(df)} rows")

# keep only bipolar depression and anxiety — throw away stress and avpd
df = df[df["target"].isin([1, 2, 4])].copy()
print(f"kept {len(df)} rows after filtering")

# convert target numbers to actual words
df["label"] = df["target"].map({1: "depression", 2: "bipolar", 4: "anxiety"})

# combine title and text into one column
df["full_text"] = df["title"].fillna("") + " " + df["text"].fillna("")

# lowercase everything
df["full_text"] = df["full_text"].str.lower()

# remove web links that start with http
df["full_text"] = df["full_text"].str.replace(r"http\S+", "", regex=True)

# remove web links that start with www but not http
df["full_text"] = df["full_text"].str.replace(r"www\S+", "", regex=True)

# remove reddit usernames
df["full_text"] = df["full_text"].str.replace(r"/?u/\S+", "", regex=True)

# remove subreddit names
df["full_text"] = df["full_text"].str.replace(r"/?r/\S+", "", regex=True)

# remove punctuation and special characters
df["full_text"] = df["full_text"].str.replace(r"[^a-z0-9\s]", "", regex=True)

# collapse extra spaces
df["full_text"] = df["full_text"].str.replace(r"\s+", " ", regex=True).str.strip()

# remove posts with fewer than 20 words
before = len(df)
df = df[df["full_text"].str.split().str.len() >= 20].copy()
print(f"removed {before - len(df)} short posts")

# how many posts per label
print("\nposts per label:")
print(df["label"].value_counts())

# make sure the folder exists
os.makedirs("data/processed", exist_ok=True)

# save only the columns we need
df[["full_text", "label"]].to_csv("data/processed/clean.csv", index=False)
print(f"\nsaved {len(df)} posts to data/processed/clean.csv")
