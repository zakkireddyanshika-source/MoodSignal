# step 3 - keep only bipolar depression and anxiety posts, clean the text, save it

import pandas as pd
import re
import sys
import os

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

df = pd.read_csv("data/raw/data_to_be_cleansed.csv")
print(f"loaded {len(df)} rows")

df = df[df["target"].isin([1, 2, 4])].copy()
print(f"kept {len(df)} rows after filtering")

df["label"] = df["target"].map({1: "depression", 2: "bipolar", 4: "anxiety"})

df["full_text"] = df["title"].fillna("") + " " + df["text"].fillna("")

df["full_text"] = df["full_text"].str.lower()

df["full_text"] = df["full_text"].str.replace(r"http\S+", "", regex=True)

df["full_text"] = df["full_text"].str.replace(r"www\S+", "", regex=True)

df["full_text"] = df["full_text"].str.replace(r"/?u/\S+", "", regex=True)

df["full_text"] = df["full_text"].str.replace(r"/?r/\S+", "", regex=True)

df["full_text"] = df["full_text"].str.replace(r"[^a-z0-9\s]", "", regex=True)

df["full_text"] = df["full_text"].str.replace(r"\s+", " ", regex=True).str.strip()

before = len(df)
df = df[df["full_text"].str.split().str.len() >= 20].copy()
print(f"removed {before - len(df)} short posts")

print("\nposts per label:")
print(df["label"].value_counts())

os.makedirs("data/processed", exist_ok=True)

df[["full_text", "label"]].to_csv("data/processed/clean.csv", index=False)
print(f"\nsaved {len(df)} posts to data/processed/clean.csv")
