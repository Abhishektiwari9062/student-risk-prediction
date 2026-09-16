import pandas as pd
import joblib
from sklearn.metrics import f1_score, recall_score

X_val = pd.read_csv("outputs/X_val.csv")
y_val = pd.read_csv("outputs/y_val.csv").values.ravel()
model = joblib.load("outputs/model_xgboost_best.pkl")
target_encoder = joblib.load("outputs/target_encoder.pkl")

preds = model.predict(X_val)

def readable_label(series, high_label, low_label):
    high_val = series.max()
    low_val = series.min()
    return {high_val: high_label, low_val: low_label}

gender_labels = readable_label(X_val["Gender"], "Gender=1", "Gender=0")
scholarship_labels = readable_label(
    X_val["Scholarship holder"], "Scholarship holder", "No scholarship"
)

dropout_class_idx = list(target_encoder.classes_).index("Dropout")

print("=== Fairness check: performance by Gender ===")
for gender_value in X_val["Gender"].unique():
    mask = X_val["Gender"] == gender_value
    subset_f1 = f1_score(y_val[mask], preds[mask], average="macro")
    subset_recall_dropout = recall_score(
        y_val[mask], preds[mask], average=None, labels=[dropout_class_idx]
    )
    label = gender_labels[gender_value]
    print(f"{label} (n={mask.sum()}): Macro F1={subset_f1:.4f}, Dropout Recall={subset_recall_dropout[0]:.4f}")

print("\n=== Fairness check: performance by Scholarship status ===")
for scholarship_value in X_val["Scholarship holder"].unique():
    mask = X_val["Scholarship holder"] == scholarship_value
    subset_f1 = f1_score(y_val[mask], preds[mask], average="macro")
    label = scholarship_labels[scholarship_value]
    print(f"{label} (n={mask.sum()}): Macro F1={subset_f1:.4f}")

print("\nWrite in your report: if performance gaps exist across groups, discuss WHY they might exist and what mitigation would look like (this section alone signals research maturity to reviewers).")