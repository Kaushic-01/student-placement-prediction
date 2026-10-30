
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# 1. File and target column
DATA_FILE = "student_placement_career_success_dataset.csv"
TARGET = "placement_status"

# 2. Columns to exclude
EXCLUDED = [
    "student_id",
    TARGET,
    "company_type",
    "work_mode",
    "salary_lpa",
    "interview_rounds_cleared",
    "offer_count",
    "joining_delay_months",
    "layoffs_risk_score"
]

# 3. Load dataset
df = pd.read_csv(DATA_FILE)

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' was not found.")

X = df.drop(columns=EXCLUDED)
y = df[TARGET].astype(str).str.strip()

# 4. Identify numerical and categorical columns
numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

# 5. Handle missing numerical values
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

# 6. Handle categorical values
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# 7. Combine preprocessing
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# 8. Create Machine Learning pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2,
        n_jobs=-1
    ))
])

# 9. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 10. Train model
pipeline.fit(X_train, y_train)

# 11. Make predictions
predictions = pipeline.predict(X_test)

# 12. Evaluate model
print("Balanced accuracy:", round(
    balanced_accuracy_score(y_test, predictions), 4
))

print("\nClassification report:")
print(classification_report(
    y_test, predictions, zero_division=0
))

print("\nConfusion matrix (label order):", pipeline.classes_)

print(confusion_matrix(
    y_test,
    predictions,
    labels=pipeline.classes_
))

# 13. Store model and input information
bundle = {
    "model": pipeline,
    "feature_columns": X.columns.tolist(),
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,

    "categories": {
        col: sorted(
            X[col].dropna().astype(str).unique().tolist()
        )
        for col in categorical_features
    },

    "defaults": {
        col: (
            float(X[col].median())
            if col in numeric_features
            else str(X[col].mode(dropna=True).iloc[0])
        )
        for col in X.columns
    },

    "numeric_bounds": {
        col: [
            float(X[col].min()),
            float(X[col].max())
        ]
        for col in numeric_features
    },

    "classes": pipeline.classes_.tolist()
}

# 14. Save trained model
joblib.dump(bundle, "placement_model.pkl")

print("\nModel pipeline saved as placement_model.pkl")
