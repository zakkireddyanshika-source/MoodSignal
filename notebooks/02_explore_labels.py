
import pandas as pd
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
df = pd.read_csv("data/raw/data_to_be_cleansed.csv")
print("how many posts per target:")
print(df["target"].value_counts().sort_index())

combined = (df["title"].fillna("") + " " + df["text"].fillna("")).str.lower()

keywords = ["bipolar", "manic", "depress", "anxiety", "stress", "avpd", "avoidant"]

print("\nkeyword frequency per target:")
for target in sorted(df["target"].unique()):
    group = combined[df["target"] == target]
    counts = {kw: float(round(100 * group.str.contains(kw, regex=False).sum() / len(group), 1)) for kw in keywords}
    print(f"target {target}: {counts}")

print("\nexample posts per target:")
for target in sorted(df["target"].unique()):
    print(f"\ntarget {target}:")
    examples = df[df["target"] == target].sample(3, random_state=42)
    for _, row in examples.iterrows():
        print(f"  title: {row['title']}")
        print(f"  text: {str(row['text'])[:200]}")
        print()
