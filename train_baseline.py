import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

X_train = pd.read_csv("outputs/X_train.csv")
X_val = pd.read_csv("outputs/X_val.csv")
y_train = pd.read_csv("outputs/y_train.csv").values.ravel()
y_val = pd.read_csv("outputs/y_val.csv").values.ravel()
target_encoder = joblib.load("outputs/target_encoder.pkl")
class_names = target_encoder.classes_

def evaluate(model, name):
    preds = model.predict(X_val)
    print(f"\n{'='*50}\n{name}\n{'='*50}")
    print(classification_report(y_val, preds, target_names=class_names))
    macro_f1 = f1_score(y_val, preds, average="macro")
    print(f"Macro F1 (treats all classes equally, good for imbalance): {macro_f1:.4f}")

    cm = confusion_matrix(y_val, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix — {name}")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"outputs/confusion_matrix_{name.replace(' ', '_')}.png", dpi=150)
    plt.close()
    return macro_f1

# Baseline 1: Logistic Regression — the "floor" every fancier model must beat
logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
logreg.fit(X_train, y_train)
f1_logreg = evaluate(logreg, "Logistic Regression")
joblib.dump(logreg, "outputs/model_logreg.pkl")

# Baseline 2: Random Forest — a reasonable non-linear baseline
rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)
f1_rf = evaluate(rf, "Random Forest")
joblib.dump(rf, "outputs/model_rf.pkl")

print(f"\n\nSummary — Macro F1: LogReg={f1_logreg:.4f} | RandomForest={f1_rf:.4f}")