from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_validate,
    train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


DATA_PATH = Path("india_real_estate_valuation.xlsx")
ARTIFACT_PATH = Path("model.pkl")

TARGET = "Price_per_sqft_INR"

NUMERIC_FEATURES = [
    "Property_Age_Years",
    "Distance_to_Metro_km",
    "Nearby_Convenience_Stores",
    "Latitude",
    "Longitude",
    "Property_Area_sqft"
]

CATEGORICAL_FEATURES = [
    "City",
    "State",
    "Locality"
]


def prepare_dataframe(data):
    data = data.copy()

    data["Transaction_Year"] = data["Transaction_Date"].dt.year
    data["Transaction_Month"] = data["Transaction_Date"].dt.month

    data["Transaction_Seasonal_Value"] = np.sin(
        2 * np.pi * data["Transaction_Month"] / 12
    )

    data["Transaction_Seasonal_Cosine"] = np.cos(
        2 * np.pi * data["Transaction_Month"] / 12
    )

    return data


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "india_real_estate_valuation.xlsx not found. "
            "Place it in the same folder as this script."
        )

    raw_data = pd.read_excel(
        DATA_PATH,
        sheet_name="India_Property_Data"
    )

    data = prepare_dataframe(raw_data)

    features = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + [
            "Transaction_Year",
            "Transaction_Month",
            "Transaction_Seasonal_Value",
            "Transaction_Seasonal_Cosine"
        ]
    )

    X = data[features]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        shuffle=True
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES + [
                    "Transaction_Year",
                    "Transaction_Month",
                    "Transaction_Seasonal_Value",
                    "Transaction_Seasonal_Cosine"
                ]
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES
            )
        ]
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(
            random_state=42,
            n_jobs=-1
        )
    }

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    comparison_results = []

    for model_name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessing", preprocessor),
                ("model", estimator)
            ]
        )

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
                "r2": "r2"
            },
            n_jobs=-1
        )

        comparison_results.append({
            "Model": model_name,
            "CV MAE": -scores["test_mae"].mean(),
            "CV RMSE": -scores["test_rmse"].mean(),
            "CV R2": scores["test_r2"].mean()
        })

    comparison_table = pd.DataFrame(comparison_results)

    print("\nModel Comparison (5-fold Cross-Validation)")
    print(comparison_table.to_string(index=False))

    tuned_pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    parameter_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_leaf": [1, 2, 4]
    }

    grid_search = GridSearchCV(
        tuned_pipeline,
        param_grid=parameter_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("\nBest Parameters")
    print(grid_search.best_params_)

    print(
        "\nBest CV RMSE:",
        -grid_search.best_score_
    )

    best_pipeline = grid_search.best_estimator_
    test_prediction = best_pipeline.predict(X_test)

    test_mae = mean_absolute_error(y_test, test_prediction)
    test_rmse = root_mean_squared_error(y_test, test_prediction)
    test_r2 = r2_score(y_test, test_prediction)

    print("\nHeld-Out Test Performance")
    print(f"MAE: {test_mae:.2f} INR per sq ft")
    print(f"RMSE: {test_rmse:.2f} INR per sq ft")
    print(f"R2: {test_r2:.3f}")

    input_min = X_train.min().to_dict()
    input_max = X_train.max().to_dict()

    example_row = X_train.iloc[0].to_dict()

    artifact = {
        "pipeline": best_pipeline,
        "features": features,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "input_min": input_min,
        "input_max": input_max,
        "example_input": example_row,
        "cities": sorted(data["City"].unique().tolist()),
        "states": sorted(data["State"].unique().tolist()),
        "localities": sorted(data["Locality"].unique().tolist()),
        "metrics": {
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
            "test_properties": len(X_test),
            "best_parameters": grid_search.best_params_,
            "comparison": comparison_table.to_dict(
                orient="records"
            )
        }
    }

    with open(ARTIFACT_PATH, "wb") as file:
        pickle.dump(artifact, file)

    print(f"\nSaved model artifact: {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()