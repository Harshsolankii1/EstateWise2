from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# 1. Project aur dataset ka path
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "real_estate.xlsx"

# 2. Excel dataset load karo
df = pd.read_excel(DATA_PATH)

# 3. Column names simple karo
df = df.rename(columns={
    "X1 transaction date": "transaction_date",
    "X2 house age": "house_age",
    "X3 distance to the nearest MRT station": "mrt_distance",
    "X4 number of convenience stores": "stores",
    "X5 latitude": "latitude",
    "X6 longitude": "longitude",
    "Y house price of unit area": "price_per_unit_area"
})

# 4. Sirf record number wala column hatao
df = df.drop(columns=["No"])

# 5. Input columns select karo
feature_columns = [
    "transaction_date",
    "house_age",
    "mrt_distance",
    "stores",
    "latitude",
    "longitude"
]

X = df[feature_columns]

# 6. Jo value predict karni hai
y = df["price_per_unit_area"]

# 7. Training aur final testing data separate karo
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# 8. Preparation ka result dikhao
print("\n--- DATA PREPARATION COMPLETE ---")
print("Total properties:", len(df))
print("Input features:", len(feature_columns))
print("Training properties:", len(X_train))
print("Testing properties:", len(X_test))

print("\n--- INPUT COLUMN NAMES ---")
for column in feature_columns:
    print(column)

print("\n--- TARGET ---")
print(y.name)

print("\nNext step: compare machine-learning models.")

# 9. Models aur evaluation tools import karo
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_validate


# 10. Teen model pipelines banao
models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "Decision Tree": Pipeline([
        ("model", DecisionTreeRegressor(
            random_state=42
        ))
    ]),

    "Random Forest": Pipeline([
        ("model", RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=1
        ))
    ])
}


# 11. Sabhi models ke liye same 5 folds
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# 12. Training data par models compare karo
results = []

print("\n--- MODEL COMPARISON STARTED ---", flush=True)

for name, pipeline in models.items():
    print(f"\nEvaluating: {name}...", flush=True)

    scores = cross_validate(
        estimator=pipeline,
        X=X_train,
        y=y_train,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2"
        },
        n_jobs=1,
        error_score="raise"
    )

    results.append({
        "Model": name,
        "CV_MAE": -scores["test_mae"].mean(),
        "CV_RMSE": -scores["test_rmse"].mean(),
        "RMSE_Std": scores["test_rmse"].std(),
        "CV_R2": scores["test_r2"].mean()
    })

    print(f"Completed: {name}", flush=True)


# 13. Result table banao
comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    by="CV_RMSE"
).reset_index(drop=True)

print("\n--- MODEL COMPARISON RESULTS ---")
print(comparison.round(3).to_string(index=False))


# 14. Result CSV file mein save karo
results_folder = PROJECT_DIR / "results"
results_folder.mkdir(parents=True, exist_ok=True)

output_path = results_folder / "model_comparison.csv"

comparison.to_csv(output_path, index=False)

print("\nComparison saved to:", output_path)
print("\nFinal test data has NOT been used.")
print("Next step: review results and tune the selected model.")

# 15. Random Forest ki settings tune karo
from sklearn.model_selection import GridSearchCV

print("\n--- RANDOM FOREST TUNING STARTED ---", flush=True)

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 5, 10],
    "model__min_samples_leaf": [1, 2, 4]
}

grid = GridSearchCV(
    estimator=models["Random Forest"],
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    n_jobs=1,
    verbose=1,
    refit=True,
    error_score="raise"
)

grid.fit(X_train, y_train)


# 16. Selected settings aur CV score dikhao
best_pipeline = grid.best_estimator_

print("\n--- TUNING COMPLETE ---")
print("\nSelected settings:")

for parameter, value in grid.best_params_.items():
    print(f"{parameter}: {value}")

print(
    "\nSelected model CV RMSE:",
    round(-grid.best_score_, 3)
)


# 17. Original Random Forest se comparison
original_rf_rmse = comparison.loc[
    comparison["Model"] == "Random Forest",
    "CV_RMSE"
].iloc[0]

print(
    "Original Random Forest CV RMSE:",
    round(original_rf_rmse, 3)
)


# 18. Saari tested settings ki table save karo
tuning_results = pd.DataFrame(grid.cv_results_)

tuning_results["CV_RMSE"] = (
    -tuning_results["mean_test_score"]
)

tuning_summary = tuning_results[[
    "param_model__n_estimators",
    "param_model__max_depth",
    "param_model__min_samples_leaf",
    "CV_RMSE",
    "std_test_score",
    "rank_test_score"
]].sort_values("rank_test_score")

tuning_summary.to_csv(
    results_folder / "tuning_results.csv",
    index=False
)

print("\nTuning results saved in results/tuning_results.csv")
print("Final test data has still NOT been used.")
print("Next step: final test evaluation and model saving.")

# 19. Final test evaluation
import json
import pickle
import sklearn

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("\n--- FINAL TEST EVALUATION ---", flush=True)

y_pred = best_pipeline.predict(X_test)

test_mae = mean_absolute_error(y_test, y_pred)
test_rmse = mean_squared_error(y_test, y_pred) ** 0.5
test_r2 = r2_score(y_test, y_pred)

print("Test properties:", len(X_test))
print("Test MAE:", round(test_mae, 3))
print("Test RMSE:", round(test_rmse, 3))
print("Test R2:", round(test_r2, 3))


# 20. Final metrics save karo
metrics = {
    "test_properties": int(len(X_test)),
    "test_mae": float(test_mae),
    "test_rmse": float(test_rmse),
    "test_r2": float(test_r2),
    "selected_cv_rmse": float(-grid.best_score_),
    "best_parameters": grid.best_params_
}

metrics_path = results_folder / "final_metrics.json"

with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)


# 21. Actual aur predicted values save karo
test_predictions = X_test.copy()

test_predictions["actual_price"] = y_test
test_predictions["predicted_price"] = y_pred

test_predictions["error_actual_minus_predicted"] = (
    test_predictions["actual_price"]
    - test_predictions["predicted_price"]
)

test_predictions.to_csv(
    results_folder / "test_predictions.csv",
    index=False
)


# 22. App ke liye pipeline aur input details save karo
artifact = {
    "pipeline": best_pipeline,
    "features": feature_columns,
    "target_unit": "10,000 TWD per ping",
    "input_min": X_train.min().to_dict(),
    "input_max": X_train.max().to_dict(),
    "defaults": X_train.median().to_dict(),
    "example_input": X_train.iloc[0].to_dict(),
    "metrics": metrics,
    "sklearn_version": sklearn.__version__
}

model_path = PROJECT_DIR / "model.pkl"

with open(model_path, "wb") as f:
    pickle.dump(
        artifact,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )

print("\n--- MODEL SAVED ---")
print("Model file:", model_path)
print("Model size in bytes:", model_path.stat().st_size)


# 23. Saved file dobara load karke check karo
with open(model_path, "rb") as f:
    loaded_artifact = pickle.load(f)

sample_input = X_test.iloc[[0]]

original_prediction = float(
    best_pipeline.predict(sample_input)[0]
)

loaded_prediction = float(
    loaded_artifact["pipeline"].predict(sample_input)[0]
)

if abs(original_prediction - loaded_prediction) > 1e-9:
    raise RuntimeError("Saved model prediction verification failed.")

print("Saved model verification: PASSED")
print("\nMetrics saved in results/final_metrics.json")
print("Predictions saved in results/test_predictions.csv")
print("\nNext step: create the Streamlit application.")