from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["axes.titleweight"] = "bold"

PRICE_LABEL = "Price per Unit Area (10,000 TWD/ping)"


def save_chart(fig, filename):
    fig.tight_layout()
    output_path = FIGURES_DIR / filename
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filename}", flush=True)


# Load and prepare the dataset
df = pd.read_excel(
    PROJECT_DIR / "data" / "real_estate.xlsx"
)

df = df.rename(columns={
    "X1 transaction date": "transaction_date",
    "X2 house age": "house_age",
    "X3 distance to the nearest MRT station": "mrt_distance",
    "X4 number of convenience stores": "stores",
    "X5 latitude": "latitude",
    "X6 longitude": "longitude",
    "Y house price of unit area": "price_per_unit_area"
})

feature_columns = [
    "transaction_date",
    "house_age",
    "mrt_distance",
    "stores",
    "latitude",
    "longitude"
]

X = df[feature_columns]
y = df["price_per_unit_area"]

# Reproduce the original training split without retraining any model
X_train, _, y_train, _ = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

train_df = X_train.copy()
train_df["price_per_unit_area"] = y_train

print("\n--- GENERATING PROJECT CHARTS ---", flush=True)


# 1. Target distribution
fig, ax = plt.subplots(figsize=(9, 5))

sns.histplot(
    data=train_df,
    x="price_per_unit_area",
    bins=25,
    color="#2563eb",
    ax=ax
)

ax.set_title("Property Price Distribution — Training Data")
ax.set_xlabel(PRICE_LABEL)
ax.set_ylabel("Number of Properties")

save_chart(fig, "01_price_distribution.png")


# 2. Property age versus price
fig, ax = plt.subplots(figsize=(9, 5))

sns.scatterplot(
    data=train_df,
    x="house_age",
    y="price_per_unit_area",
    color="#2563eb",
    alpha=0.65,
    ax=ax
)

ax.set_title("Property Age vs Price — Training Data")
ax.set_xlabel("Property Age (Years)")
ax.set_ylabel(PRICE_LABEL)

save_chart(fig, "02_property_age_vs_price.png")


# 3. MRT distance versus price
fig, ax = plt.subplots(figsize=(9, 5))

sns.scatterplot(
    data=train_df,
    x="mrt_distance",
    y="price_per_unit_area",
    color="#0d9488",
    alpha=0.65,
    ax=ax
)

ax.set_title("MRT Accessibility vs Price — Training Data")
ax.set_xlabel("Distance to Nearest MRT Station (Metres)")
ax.set_ylabel(PRICE_LABEL)

save_chart(fig, "03_mrt_distance_vs_price.png")


# 4. Nearby stores versus price
fig, ax = plt.subplots(figsize=(10, 5))

sns.boxplot(
    data=train_df,
    x="stores",
    y="price_per_unit_area",
    color="#93c5fd",
    ax=ax
)

ax.set_title("Nearby Convenience Stores and Price — Training Data")
ax.set_xlabel("Number of Convenience Stores")
ax.set_ylabel(PRICE_LABEL)

save_chart(fig, "04_stores_vs_price.png")


# 5. Correlation heatmap
readable_names = {
    "transaction_date": "Transaction Date",
    "house_age": "Property Age",
    "mrt_distance": "MRT Distance",
    "stores": "Stores",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "price_per_unit_area": "Price per Unit Area"
}

correlation = train_df.rename(
    columns=readable_names
).corr()

fig, ax = plt.subplots(figsize=(11, 8))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    vmin=-1,
    vmax=1,
    ax=ax
)

ax.set_title("Feature Correlation Matrix — Training Data")

save_chart(fig, "05_correlation_heatmap.png")


# 6. Model comparison using saved CV results
comparison = pd.read_csv(
    RESULTS_DIR / "model_comparison.csv"
).sort_values("CV_RMSE")

fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.barh(
    comparison["Model"],
    comparison["CV_RMSE"],
    color="#2563eb"
)

ax.invert_yaxis()
ax.bar_label(bars, fmt="%.3f", padding=5)
ax.set_xlim(0, comparison["CV_RMSE"].max() * 1.18)

ax.set_title("Baseline Model Comparison — 5-Fold Cross-Validation")
ax.set_xlabel("Mean CV RMSE — Lower Is Better")
ax.set_ylabel("")

save_chart(fig, "06_model_comparison.png")


# 7. Actual versus predicted values on the held-out test set
predictions = pd.read_csv(
    RESULTS_DIR / "test_predictions.csv"
)

actual = predictions["actual_price"]
predicted = predictions["predicted_price"]

fig, ax = plt.subplots(figsize=(7, 7))

ax.scatter(
    actual,
    predicted,
    color="#2563eb",
    alpha=0.7
)

lower = min(actual.min(), predicted.min())
upper = max(actual.max(), predicted.max())

ax.plot(
    [lower, upper],
    [lower, upper],
    linestyle="--",
    color="#dc2626",
    label="Perfect Prediction"
)

ax.set_title("Actual vs Predicted Price — Held-Out Test Data")
ax.set_xlabel("Actual Price (10,000 TWD/ping)")
ax.set_ylabel("Predicted Price (10,000 TWD/ping)")
ax.legend()

save_chart(fig, "07_actual_vs_predicted.png")


# 8. Residual analysis
residuals = actual - predicted

fig, ax = plt.subplots(figsize=(9, 5))

ax.scatter(
    predicted,
    residuals,
    color="#0d9488",
    alpha=0.7
)

ax.axhline(
    y=0,
    color="#dc2626",
    linestyle="--"
)

ax.set_title("Residual Analysis — Held-Out Test Data")
ax.set_xlabel("Predicted Price (10,000 TWD/ping)")
ax.set_ylabel("Residual: Actual − Predicted (10,000 TWD/ping)")

save_chart(fig, "08_residual_analysis.png")


# Save training-data summaries for report preparation
train_df.describe().to_csv(
    RESULTS_DIR / "training_data_summary.csv"
)

train_df.corr().to_csv(
    RESULTS_DIR / "training_correlations.csv"
)

print("\n--- ALL 8 CHARTS SAVED SUCCESSFULLY ---")
print("Location:", FIGURES_DIR)
print("Training-data summary and correlations also saved.")
print("No model was retrained.")