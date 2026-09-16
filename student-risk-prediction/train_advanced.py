import pandas as pd
import joblib
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV

X_train = pd.read_csv("outputs/X_train.csv")
X_val = pd.read_csv("outputs/X_val.csv")
y_train = pd.read_csv("outputs/y_train.csv").values.ravel()
y_val = pd.read_csv("outputs/y_val.csv").values.ravel()
target_encoder = joblib.load("outputs/target_encoder.pkl")
class_names = target_encoder.classes_

# SMOTE: synthetically generates minority-class examples, applied ONLY to training data
# (never touch val/test with synthetic data — that would corrupt the evaluation)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("Before SMOTE:", pd.Series(y_train).value_counts().to_dict())
print("After SMOTE:", pd.Series(y_train_res).value_counts().to_dict())

# Small grid search — shows you tuned deliberately, not just used defaults
param_grid = {
    "max_depth": [4, 6, 8],
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
}
xgb = XGBClassifier(random_state=42, eval_metric="mlogloss")
grid = GridSearchCV(xgb, param_grid, cv=3, scoring="f1_macro", n_jobs=-1)
grid.fit(X_train_res, y_train_res)

print("\nBest params:", grid.best_params_)
best_model = grid.best_estimator_

preds = best_model.predict(X_val)
print("\n" + classification_report(y_val, preds, target_names=class_names))
macro_f1 = f1_score(y_val, preds, average="macro")
print(f"XGBoost (tuned + SMOTE) Macro F1: {macro_f1:.4f}")

joblib.dump(best_model, "outputs/model_xgboost_best.pkl")