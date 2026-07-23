import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

os.makedirs("outputs", exist_ok=True)
model = joblib.load("models/baseline_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

data = pd.read_csv("data/processed/clean.csv")
classes = sorted(data["label"].unique())
labels = data["label"].map({label: i for i, label in enumerate(classes)}).values

x_train, x_test, y_train, y_test = train_test_split(data["full_text"], labels, test_size=0.2, random_state=42, stratify=labels)

sample = x_test.sample(n=100, random_state=42)

sample_numbers = vectorizer.transform(sample)

explainer = shap.LinearExplainer(model, sample_numbers)
shap_values = explainer.shap_values(sample_numbers)

words = vectorizer.get_feature_names_out()

for i, class_names in enumerate(classes):
    class_values = shap_values[:, :, i]
    average_impact = np.abs(class_values).mean(axis=0)
    top_indices = average_impact.argsort()[-10:][::-1]
    print(f"\ntop words for {class_names}:")
    for index in top_indices:
        print(f"{words[index]}: {average_impact[index]:.4f}")

print(np.array(shap_values).shape)

plt.figure(figsize=(10, 6))
overall_impact = np.abs(shap_values).mean(axis=(0, 2))
top_20 = overall_impact.argsort()[-20:]
plt.barh(range(20), overall_impact[top_20])
plt.yticks(range(20), [words[i] for i in top_20])
plt.xlabel("average impact on prediction")
plt.title("most important words overall")
plt.tight_layout()
plt.savefig("outputs/shap_summary.png")
print("saved plot to outputs/shap_summary.png")
