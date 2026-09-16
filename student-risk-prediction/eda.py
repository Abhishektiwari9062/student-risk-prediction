import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/data.csv", sep=";")

print("Shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nTarget distribution:\n", df["Target"].value_counts())
print("\nTarget distribution (%):\n", df["Target"].value_counts(normalize=True) * 100)
print("\nMissing values total:", df.isnull().sum().sum())
print("\nData types:\n", df.dtypes.value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Target", order=df["Target"].value_counts().index)
plt.title("Class Distribution: Dropout / Enrolled / Graduate")
plt.tight_layout()
plt.savefig("outputs/class_distribution.png", dpi=150)
plt.close()

numeric_df = df.select_dtypes(include="number")
plt.figure(figsize=(14, 12))
sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0, annot=False)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png", dpi=150)
plt.close()

for col in ["Age at enrollment", "Curricular units 1st sem (grade)", "Scholarship holder", "Gender"]:
    if col in df.columns:
        print(f"\n{col} summary:\n", df[col].describe() if df[col].dtype != object else df[col].value_counts())

df.to_csv("outputs/raw_snapshot.csv", index=False)
print("\nEDA complete. Charts saved to outputs/.")