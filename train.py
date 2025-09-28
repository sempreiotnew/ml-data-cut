import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# ----------------------------
# Load CSV data
# ----------------------------
df = pd.read_csv("data.csv")

# ----------------------------
# Features and target
# ----------------------------
feature_cols = [
    "baseline_mean",
    "min_val",
    "drop_magnitude",
    "drop_duration_s",
    "recovery_duration_s",
    "total_response_time_s",
    "drop_rate",
    "recovery_rate",
    "drop_area",
    "recovery_area",
    "skewness",
    "kurtosis"
]

X = df[feature_cols].values
y = df["label"].values  # target classes: alcool, cigarro, ar

# ----------------------------
# Train/test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# Train Random Forest
# ----------------------------
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    class_weight="balanced"  # useful if classes are unbalanced
)
clf.fit(X_train, y_train)

# ----------------------------
# Evaluate model
# ----------------------------
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ----------------------------
# Save trained model
# ----------------------------
joblib.dump(clf, "rf_sensor_model.pkl")
print("Model saved as rf_sensor_model.pkl")
