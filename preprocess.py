import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/data.csv", sep=";")
df.columns = df.columns.str.strip()

# Target encoding — keep the mapping, you'll need it for readable results later
target_encoder = LabelEncoder()
df["Target_encoded"] = target_encoder.fit_transform(df["Target"])
print("Target classes:", dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_))))

X = df.drop(columns=["Target", "Target_encoded"])
y = df["Target_encoded"]

# Stratified split — CRITICAL given the class imbalance, keeps class ratios consistent
# across train/val/test so evaluation isn't accidentally lucky or unlucky
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"\nTrain: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

# Scale numeric features — fit ONLY on train, to avoid leaking test-set statistics
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

# Save everything so later scripts don't have to redo this
joblib.dump(scaler, "outputs/scaler.pkl")
joblib.dump(target_encoder, "outputs/target_encoder.pkl")
X_train_scaled.to_csv("outputs/X_train.csv", index=False)
X_val_scaled.to_csv("outputs/X_val.csv", index=False)
X_test_scaled.to_csv("outputs/X_test.csv", index=False)
y_train.to_csv("outputs/y_train.csv", index=False)
y_val.to_csv("outputs/y_val.csv", index=False)
y_test.to_csv("outputs/y_test.csv", index=False)

print("\nPreprocessing complete. Splits saved to outputs/.")