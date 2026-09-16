import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

X_train = pd.read_csv("outputs/X_train.csv")
X_val = pd.read_csv("outputs/X_val.csv")
model = joblib.load("outputs/model_xgboost_best.pkl")
target_encoder = joblib.load("outputs/target_encoder.pkl")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val.iloc[:200])  
dropout_class_idx = list(target_encoder.classes_).index("Dropout")
plt.figure()
shap.summary_plot(
    shap_values[:, :, dropout_class_idx] if len(shap_values.shape) == 3 else shap_values,
    X_val.iloc[:200],
    show=False
)
plt.tight_layout()
plt.savefig("outputs/shap_summary_dropout.png", dpi=150)
plt.close()

print("SHAP summary plot saved. Open outputs/shap_summary_dropout.png")
print("\nWrite in your report: which features push predictions toward 'Dropout,' and whether that matches existing education-research literature (e.g., first-semester grades, age at enrollment, tuition payment status).")