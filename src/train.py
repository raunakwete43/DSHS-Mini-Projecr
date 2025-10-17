import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier  # <-- Import XGBoost
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import joblib
import warnings
import kagglehub
import os

path = kagglehub.dataset_download("redwankarimsony/heart-disease-data")
path = os.path.join(path, os.listdir(path)[0])

warnings.filterwarnings("ignore")

print("Loading and preprocessing data...")
df = pd.read_csv(path)
df = df.drop(columns=["id", "dataset"])
df["target"] = df["num"].apply(lambda x: 1 if x > 0 else 0)
df = df.drop("num", axis=1)
X = df.drop("target", axis=1)
y = df["target"]
numerical_cols = ["age", "trestbps", "chol", "thalch", "oldpeak"]
categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

numeric_imputer = SimpleImputer(strategy="median")
categorical_imputer = SimpleImputer(strategy="most_frequent")
X[numerical_cols] = numeric_imputer.fit_transform(X[numerical_cols])
X[categorical_cols] = categorical_imputer.fit_transform(X[categorical_cols])
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Data ready for tuning.")


print("\nStarting Hyperparameter Tuning...")
param_grid = {
    "LogisticRegression": {
        "model": LogisticRegression(solver="liblinear"),
        "params": {"C": [0.1, 1, 2, 3, 4, 10], "penalty": ["l1", "l2"]},
    },
    "RandomForestClassifier": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100, 150, 200],
            "max_depth": [10, 20, 30, 40, 50],
            "min_samples_split": [2, 3, 4, 5, 6, 7, 8, 9],
        },
    },
    "SVC": {
        "model": SVC(random_state=42),
        "params": {"C": [0.1, 1, 2, 3, 4, 10], "kernel": ["linear", "rbf"]},
    },
    "XGBClassifier": {  # <-- XGBoost entry
        "model": XGBClassifier(
            use_label_encoder=False, eval_metric="logloss", random_state=42
        ),
        "params": {
            "n_estimators": [50, 100, 150, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5, 7, 10],
        },
    },
}

scores = []
for model_name, config in param_grid.items():
    print(f"\n--- Tuning {model_name} ---")
    grid_search = GridSearchCV(
        config["model"], config["params"], cv=5, scoring="accuracy", n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    scores.append(
        {
            "model": model_name,
            "best_score": grid_search.best_score_,
            "best_params": grid_search.best_params_,
            "best_estimator": grid_search.best_estimator_,
        }
    )
    print(f"Best Score: {grid_search.best_score_:.4f}")
    print(f"Best Params: {grid_search.best_params_}")

best_model_config = max(scores, key=lambda x: x["best_score"])
best_model = best_model_config["best_estimator"]
print(
    f"\n🏆 Overall Best Model: {best_model_config['model']} with accuracy {best_model_config['best_score']:.4f}"
)

y_pred = best_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred)
print(f"Final accuracy on test set: {final_accuracy:.4f}")

print("Saving best model and artifacts...")
artifacts = {
    "model": best_model,
    "numeric_imputer": numeric_imputer,
    "categorical_imputer": categorical_imputer,
    "encoded_columns": X.columns.tolist(),
}
joblib.dump(artifacts, "model_artifacts.joblib")

print("\n✅ Best model and artifacts saved successfully as 'model_artifacts.joblib'")
