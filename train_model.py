import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

DATA_FILE = "student_data.csv"
FEATURES = ["CGPA", "AptitudeScore", "TechnicalScore", "Projects", "Internship"]
TARGET = "Placed"

df = pd.read_csv(DATA_FILE)
required = FEATURES + [TARGET]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

X = df[FEATURES]
y = df[TARGET].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced",
        min_samples_leaf=2
    ))
])
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("Balanced accuracy:", round(balanced_accuracy_score(y_test, pred), 4))
print("Classification report:")
print(classification_report(y_test, pred, target_names=["Not Placed", "Placed"], zero_division=0))
print("Confusion matrix:")
print(confusion_matrix(y_test, pred))

joblib.dump({
    "model": model,
    "features": FEATURES,
    "target": TARGET,
    "class_labels": {0: "Not Placed", 1: "Placed"}
}, "placement_model.pkl")
print("Saved: placement_model.pkl")
