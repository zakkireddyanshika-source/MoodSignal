# step 2 - figure out what the target numbers 0-4 actually mean

import pandas as pd
import sys

# fix printing for windows
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# load the raw data
df = pd.read_csv("data/raw/data_to_be_cleansed.csv")

# how many posts per target number
print("how many posts per target:")
print(df["target"].value_counts().sort_index())

# combine title and text into one thing to search through
combined = (df["title"].fillna("") + " " + df["text"].fillna("")).str.lower()

# keywords that hint at each condition
keywords = ["bipolar", "manic", "depress", "anxiety", "stress", "avpd", "avoidant"]

# for each target group check how often each keyword appears
print("\nkeyword frequency per target:")
for target in sorted(df["target"].unique()):
    # get just the posts for this target
    group = combined[df["target"] == target]
    # count how many posts mention each keyword (regex=False so keywords are treated as plain text, not patterns)
    counts = {kw: float(round(100 * group.str.contains(kw, regex=False).sum() / len(group), 1)) for kw in keywords}
    print(f"target {target}: {counts}")

# print 3 example posts per target so we can read them
print("\nexample posts per target:")
for target in sorted(df["target"].unique()):
    print(f"\ntarget {target}:")
    examples = df[df["target"] == target].sample(3, random_state=42)
    for _, row in examples.iterrows():
        print(f"  title: {row['title']}")
        print(f"  text: {str(row['text'])[:200]}")
        print()
