import pandas as pd

df = pd.read_csv("data/processed/clean.csv")
print(f"loaded {len(df)} rows")

df["word_count"] = df["full_text"].str.split().str.len()

df["i_rate"] = df["full_text"].str.count(r"\b(i|me|my)\b") / df["word_count"]

df["absolutist_rate"] = df["full_text"].str.count(r"\b(always|never|nothing|everything|everyone|nobody|forever|completely)\b") / df["word_count"]

df["grandiosity_rate"] = df["full_text"].str.count(r"\b(special|genius|chosen|destiny|extraordinary|unlimited|unstoppable)\b") / df["word_count"]

df["energy_rate"] = df["full_text"].str.count(r"\b(cant stop|no sleep|dont need sleep|racing thoughts|unstoppable)\b") / df["word_count"]

df["anxiety_rate"] = df["full_text"].str.count(r"\b(anxious|panic|worry|fear|dread|scared|terrified|overwhelmed)\b") / df["word_count"]

df["death_rate"] = df["full_text"].str.count(r"\b(die|dead|suicide|kill|end it)\b") / df["word_count"]

df["external_rate"] = df["full_text"].str.count(r"\b(they|he|she|them)\b") / df["word_count"]

df["negative_rate"] = df["full_text"].str.count(r"\b(hate|awful|terrible|horrible|worthless|useless)\b") / df["word_count"]

feature_cols = ["word_count", "i_rate", "absolutist_rate", "grandiosity_rate", "energy_rate", "anxiety_rate", "death_rate", "external_rate", "negative_rate"]
print(df.groupby("label")[feature_cols].mean())

df.to_csv("data/processed/features.csv", index=False)
print(f"saved {len(df)} rows to data/processed/features.csv")
