
# EstateWise — Real Estate Valuation

An end-to-end machine-learning project for estimating historical property prices per unit area using property characteristics, transport accessibility, nearby amenities, and geographic location.

The project includes exploratory data analysis, comparison of three regression algorithms, cross-validation, hyperparameter tuning, held-out evaluation, and a Streamlit prediction interface.

## Project Status

- Data preparation: Complete
- Baseline model comparison: Complete
- Hyperparameter tuning: Complete
- Held-out test evaluation: Complete
- Saved-model verification: Passed
- Local Streamlit application: Working
- Analysis notebook: Created; final EDA observations under review
- Public deployment: Pending
- Final project report: Pending

## Problem Statement

Estimate historical house prices per unit area from six input features using the UCI Real Estate Valuation dataset.

This is a regression task. The output is a numerical estimate of price per unit area, not the total sale price of a property.

## Dataset

- Source: UCI Machine Learning Repository
- Dataset: Real Estate Valuation
- Records: 414
- Input features: 6
- Geographic coverage: Sindian District, New Taipei City, Taiwan
- Target: House price per unit area
- Target unit: 10,000 New Taiwan Dollars (TWD) per ping

**Dataset reference:**

Yeh, I. (2018). Real Estate Valuation [Dataset]. UCI Machine Learning Repository.

DOI: 10.24432/C5J30W.

Dataset page: [UCI Real Estate Valuation Dataset](https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set)

### Input Features

| Feature | Description |
|---|---|
| Transaction date | Historical transaction date in decimal-year format |
| House age | Property age in years |
| MRT distance | Distance to the nearest MRT station in metres |
| Convenience stores | Number of nearby convenience stores |
| Latitude | Property latitude |
| Longitude | Property longitude |

The original `No` column is a record identifier and is excluded from the predictive features.

## Technology Stack

- Python
- pandas
- scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Jupyter Notebook
- Pickle
- Git and GitHub for version control and publication

## Repository Structure

```text
EstateWise/
├── data/
│   └── real_estate.xlsx
├── notebooks/
│   └── analysis.ipynb
├── results/
│   ├── figures/
│   ├── model_comparison.csv
│   ├── tuning_results.csv
│   ├── final_metrics.json
│   └── test_predictions.csv
├── screenshots/
├── check_data.py
├── train_model.py
├── create_charts.py
├── app.py
├── model.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

The final `report.pdf` will be added when reporting is complete. Additional helper scripts may be included for chart generation.

## Methodology

1. Inspect the dataset structure and missing values.
2. Standardize column names and exclude the record identifier.
3. Separate the target from the six input features.
4. Create an 80/20 training and test split using `random_state=42`.
5. Explore the training subset.
6. Compare three regression pipelines using the same shuffled 5-fold cross-validation splits.
7. Tune the selected Random Forest pipeline with GridSearchCV.
8. Evaluate the selected pipeline on the held-out test subset.
9. Save the fitted pipeline and application metadata.
10. Load the saved artifact in a Streamlit application.

The notebook documents exploratory plots after the initial model-development run. These plots are not presented as the original basis for the modelling choices.

### Data Split

| Subset | Properties |
|---|---:|
| Training | 331 |
| Held-out test | 83 |

## Baseline Model Comparison

Results below are mean scores from 5-fold cross-validation on the training subset.

| Model | CV MAE | CV RMSE | CV R² |
|---|---:|---:|---:|
| Random Forest | 5.136 | 7.821 | 0.665 |
| Linear Regression | 6.545 | 9.111 | 0.548 |
| Decision Tree | 6.603 | 10.382 | 0.387 |

Random Forest was selected for tuning because it achieved the lowest mean MAE and RMSE and the highest mean R² in this comparison.

These results do not establish statistical significance or universal superiority over other algorithms.

## Hyperparameter Tuning

GridSearchCV evaluated the following search space:

| Parameter | Tested values |
|---|---|
| `n_estimators` | 100, 200 |
| `max_depth` | None, 5, 10 |
| `min_samples_leaf` | 1, 2, 4 |

The search used 18 combinations and 5 folds, producing 90 cross-validation fits.

### Selected Configuration

```text
n_estimators = 200
max_depth = None
min_samples_leaf = 4
```

The selected mean CV RMSE was 7.591, compared with 7.821 for the baseline Random Forest.

## Held-Out Test Results

| Metric | Result |
|---|---:|
| MAE | 3.654 |
| RMSE | 5.449 |
| R² | 0.823 |

MAE and RMSE are expressed in 10,000 TWD per ping.

R² is not an accuracy percentage.

The held-out results describe one test split. They do not establish performance in other geographic areas or future markets.

## Run Locally

Run the following commands from the project root.

### 1. Create a virtual environment

```powershell
python -m venv .venv
```

### 2. Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Verify the dataset

Ensure the Excel file exists at:

```text
data/real_estate.xlsx
```

Then run:

```powershell
.\.venv\Scripts\python.exe check_data.py
```

### 4. Launch the application

If the saved `model.pkl` is already present, training is not required.

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the Local URL printed in the terminal.

### Optional: Reproduce Training

```powershell
.\.venv\Scripts\python.exe -u train_model.py
```

This reruns model comparison, tuning, and evaluation, and overwrites the generated model and result files.

Do not use repeated inspection of test results to guide further model selection.

## Analysis Notebook

Open:

```text
notebooks/analysis.ipynb
```

Select the project's `.venv` Python kernel and run the cells in order.

The notebook reads the dataset and saved evaluation files. Run `train_model.py` first if those result files are absent.

## Application Features

- Professional English interface
- Two-column property input form
- Training-range input validation
- Pre-filled example from the training subset
- Saved-pipeline inference
- Price-per-unit-area estimate
- Held-out model-performance panel
- Submitted-input summary
- Clearly stated scope and limitations

## Saved Model

`model.pkl` contains:

- Fitted prediction pipeline
- Feature names and order
- Training-data input ranges
- Default and example inputs
- Evaluation metrics
- Training scikit-learn version

The application loads the saved artifact and does not retrain the model when generating predictions.

Only load Pickle artifacts from trusted sources. Use a compatible Python and dependency environment.

## Limitations

- The dataset is small, historical, and geographically specific.
- The model is not validated for current property prices in India.
- The prediction is price per unit area, not total property value.
- Property attributes absent from the dataset are not represented.
- Individually valid input ranges do not guarantee realistic combinations.
- The output is not a guaranteed sale price or a confidence interval.
- This project is an educational demonstration, not professional advice.

## Deployment

Public deployment is pending. The deployed application URL will be added after deployment and verification.

## Team Contributions

Add each member's actual name and contribution before submission.

All members should understand the complete workflow.
