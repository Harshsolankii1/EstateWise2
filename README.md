# EstateWise India — Real Estate Valuation

An end-to-end machine-learning project for estimating historical property prices per square foot using property characteristics, transport accessibility, nearby amenities, property area, and geographic location.

The project includes exploratory data analysis, comparison of three regression algorithms, cross-validation, hyperparameter tuning, held-out evaluation, and a Streamlit prediction interface.

## Project Status

- Data preparation: Complete
- Baseline model comparison: To be updated after training
- Hyperparameter tuning: To be updated after training
- Held-out test evaluation: To be updated after training
- Saved-model verification: To be updated after training
- Local Streamlit application: Working
- Analysis notebook: To be updated
- Public deployment: Pending
- Final project report: Pending

## Problem Statement

Estimate historical property prices per square foot from property, location, and transaction-related input features using an India real-estate valuation dataset.

This is a regression task. The primary output is a numerical estimate of price per square foot in Indian Rupees (INR), not a guaranteed total sale price.

The application also calculates an estimated total property value using the entered property area:

```text
Estimated Total Property Value = Estimated Price per Sq Ft × Property Area
```

## Dataset

- Dataset: India Real Estate Valuation
- File: `india_real_estate_valuation.xlsx`
- Worksheet: `India_Property_Data`
- Records: 414
- Target: `Price_per_sqft_INR`
- Target unit: Indian Rupees per square foot
- Geographic coverage: Multiple Indian cities included in the dataset

The dataset includes property age, distance to metro, nearby convenience stores, geographic coordinates, property area, transaction date, city, state, and locality.

### Input Features

| Feature | Description |
|---|---|
| Property Age | Property age in years |
| Distance to Metro | Distance to the nearest metro station in kilometres |
| Nearby Convenience Stores | Number of nearby convenience stores |
| Latitude | Geographic latitude |
| Longitude | Geographic longitude |
| Property Area | Property area in square feet |
| City | City where the property is located |
| State | State where the property is located |
| Locality | Locality or neighbourhood |
| Transaction Date | Historical transaction date |

The `Property_ID` column is a record identifier and is excluded from predictive features.

## Technology Stack

- Python
- pandas
- NumPy
- scikit-learn
- Streamlit
- Pickle
- Git and GitHub for version control and publication

## Repository Structure

```text
EstateWise-main/
├── app.py
├── train_model.py
├── india_real_estate_valuation.xlsx
├── model.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

The final `model.pkl` is generated after running the training script.

## Methodology

1. Load the India real-estate valuation dataset.
2. Inspect the dataset structure, columns, and missing values.
3. Separate the target variable from the input features.
4. Create an 80/20 training and test split using `random_state=42`.
5. Extract transaction year and month from the transaction date.
6. Create seasonal transaction features using sine and cosine transformations.
7. Compare Linear Regression, Decision Tree, and Random Forest using shuffled 5-fold cross-validation.
8. Tune the Random Forest pipeline using GridSearchCV.
9. Evaluate the selected pipeline on the held-out test subset.
10. Save the fitted pipeline and application metadata.
11. Load the saved artifact in a Streamlit application.

### Data Split

| Subset | Properties |
|---|---:|
| Training | 331 |
| Held-out test | 83 |

## Baseline Model Comparison

Results below are mean scores from 5-fold cross-validation on the training subset.

| Model | CV MAE | CV RMSE | CV R² |
|---|---:|---:|---:|
| Linear Regression | To be updated | To be updated | To be updated |
| Decision Tree | To be updated | To be updated | To be updated |
| Random Forest | To be updated | To be updated | To be updated |

The model with the lowest cross-validated RMSE and MAE, along with the highest R², will be selected for hyperparameter tuning.

These results do not establish statistical significance or universal superiority over other algorithms.

## Hyperparameter Tuning

GridSearchCV evaluates the following search space:

| Parameter | Tested values |
|---|---|
| `n_estimators` | 100, 200 |
| `max_depth` | None, 10, 20 |
| `min_samples_leaf` | 1, 2, 4 |

The search uses 18 parameter combinations and 5 cross-validation folds, producing 90 cross-validation fits.

### Selected Configuration

```text
n_estimators = To be updated after training
max_depth = To be updated after training
min_samples_leaf = To be updated after training
```

The selected mean CV RMSE will be updated after training.

## Held-Out Test Results

| Metric | Result |
|---|---:|
| MAE | To be updated |
| RMSE | To be updated |
| R² | To be updated |

MAE and RMSE are expressed in INR per square foot.

R² is not an accuracy percentage.

The held-out results describe one test split. They do not establish performance in other geographic areas or future markets.

## Run Locally

Run the following commands from the project root.

### 1. Create a virtual environment

```powershell
python -m venv .venv
```

### 2. Activate the virtual environment

```powershell
.\.venv\Scripts\activate
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

If `requirements.txt` is not available, install the required libraries using:

```powershell
pip install pandas numpy scikit-learn streamlit openpyxl
```

### 4. Verify the dataset

Ensure the Excel file exists at:

```text
india_real_estate_valuation.xlsx
```

### 5. Train the model

```powershell
python train_model.py
```

This reruns model comparison, tuning, and evaluation, and overwrites the generated model file.

### 6. Launch the application

```powershell
python -m streamlit run app.py
```

Open the Local URL printed in the terminal:

```text
http://localhost:8501
```

## Application Features

- Professional English interface
- City, state, and locality selection
- Property and transaction input form
- Property-area-based total value calculation
- Saved-pipeline inference
- Price-per-square-foot estimate
- Held-out model-performance panel
- Submitted-input summary
- Model comparison and hyperparameter details
- Clearly stated scope and limitations

## Saved Model

`model.pkl` contains:

- Fitted prediction pipeline
- Feature names and order
- Training-data input ranges
- Default and example inputs
- City, state, and locality options
- Evaluation metrics
- Selected hyperparameters

The application loads the saved artifact and does not retrain the model when generating predictions.

Only load Pickle artifacts from trusted sources. Use a compatible Python and dependency environment.

## Limitations

- The dataset is small, historical, and limited to the cities included in the dataset.
- The model is not validated for current property prices.
- The primary prediction is price per square foot, not a guaranteed total property value.
- Property attributes absent from the dataset are not represented.
- Individually valid input ranges do not guarantee realistic property combinations.
- Coordinates outside the training-data coverage may produce unreliable estimates.
- The output is not a guaranteed sale price or a confidence interval.
- This project is an educational demonstration, not professional real-estate valuation advice.

## Deployment

Public deployment is pending. The deployed application URL will be added after deployment and verification.

## Future Improvements

- Use a larger and more recent Indian property dataset.
- Add features such as BHK, floor number, furnishing status, property type, and amenities.
- Include uncertainty ranges or prediction intervals.
- Improve geographic validation using maps and location-based features.
- Deploy the application with a custom domain and monitoring.

## Team Contributions

Add each member's actual name and contribution before submission.

All members should understand the complete workflow.