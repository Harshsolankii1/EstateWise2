# EstateWise India — Property Valuation Workspace

EstateWise India is an educational machine-learning web application that estimates historical property prices per square foot in Indian Rupees (INR). It uses property characteristics, location details, transport accessibility, nearby amenities, and transaction information to generate an estimate.

The application uses a tuned Random Forest regression model and is built with Streamlit.

## Project Objective

The objective of this project is to demonstrate an end-to-end machine-learning workflow:

- Load and understand a real-estate dataset.
- Perform exploratory data analysis.
- Compare multiple regression models.
- Use cross-validation for reliable model comparison.
- Tune the selected model using GridSearchCV.
- Evaluate the final model on a held-out test set.
- Save the trained pipeline for inference.
- Deploy an interactive Streamlit application.

## Dataset

The project uses the dataset:

**`india_real_estate_valuation.xlsx`**

The main worksheet is `India_Property_Data`, containing 414 property records.

### Target Variable

The model predicts:

**Price per square foot in INR (`Price_per_sqft_INR`)**

The application also calculates an estimated total property value by multiplying the predicted price per square foot by the entered property area.

### Input Features

| Feature | Description |
|---|---|
| Property Age | Age of the property in years |
| Distance to Metro | Distance to the nearest metro station in kilometres |
| Nearby Convenience Stores | Number of nearby convenience stores |
| Latitude | Geographic latitude |
| Longitude | Geographic longitude |
| Property Area | Property area in square feet |
| City | City where the property is located |
| State | State where the property is located |
| Locality | Locality or neighbourhood |
| Transaction Date | Historical transaction date |

## Machine-Learning Workflow

### Models Compared

The following regression models were compared:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

### Model Evaluation

Model performance was compared using shuffled 5-fold cross-validation.

The evaluation metrics were:

| Metric | Meaning |
|---|---|
| CV MAE | Mean Absolute Error during cross-validation |
| CV RMSE | Root Mean Squared Error during cross-validation |
| CV R² | Coefficient of determination during cross-validation |

Lower MAE and RMSE values indicate better prediction accuracy. Higher R² indicates that the model explains more variation in the target variable.

### Hyperparameter Tuning

The Random Forest model was tuned using `GridSearchCV`.

The tuned hyperparameters include:

- `n_estimators`
- `max_depth`
- `min_samples_leaf`

The best-performing model was saved as `model.pkl`.

### Final Evaluation

The final tuned Random Forest model was evaluated on a held-out test subset.

The application displays:

- Held-out test R²
- Test MAE
- Test RMSE
- Number of test properties

MAE and RMSE are reported in INR per square foot. R² is not an accuracy percentage.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| pandas | Data loading, processing, and tabular analysis |
| NumPy | Numerical calculations and feature engineering |
| scikit-learn | Machine-learning models, preprocessing, cross-validation, and tuning |
| Streamlit | Interactive web application |
| pickle | Saving and loading the trained model artifact |
| pathlib | File and folder path handling |

## Project Structure

```text
EstateWise-main/
│
├── app.py
├── train_model.py
├── india_real_estate_valuation.xlsx
├── model.pkl
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone or Download the Project

Download or clone the project folder.

```bash
git clone <your-github-repository-link>
cd EstateWise-main
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the virtual environment:

```bash
.venv\Scripts\activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install the required libraries using:

```bash
pip install pandas numpy scikit-learn streamlit openpyxl
```

## How to Run the Project

### Step 1: Train the Model

Make sure `india_real_estate_valuation.xlsx` is in the project folder.

Run:

```bash
python train_model.py
```

This will:

- Load the India real-estate dataset.
- Prepare features.
- Compare Linear Regression, Decision Tree, and Random Forest.
- Perform 5-fold cross-validation.
- Tune the Random Forest model using GridSearchCV.
- Evaluate the final model on the test set.
- Save the trained pipeline as `model.pkl`.

### Step 2: Run the Streamlit Application

```bash
python -m streamlit run app.py
```

The application will open in the browser at:

```text
http://localhost:8501
```

## How to Use the Application

1. Open the EstateWise India application.
2. Select the property city, state, and locality.
3. Enter property age, metro distance, nearby convenience stores, property area, latitude, and longitude.
4. Select the transaction date.
5. Click **Generate Estimate**.
6. Review the estimated price per square foot and estimated total property value.

## Output

The application provides:

- Estimated price per square foot in INR.
- Estimated total property value in INR.
- Submitted property details.
- Model evaluation metrics.
- Methodology and limitations.

The total property value is calculated as:

```text
Estimated Total Value = Estimated Price per Sq Ft × Property Area
```

## Deployment

The project can be deployed using Streamlit Community Cloud.

1. Upload the project files to a GitHub repository.
2. Visit [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create a new app.
4. Connect the GitHub repository.
5. Select the main branch.
6. Set the main file path as `app.py`.
7. Deploy the application.

Streamlit Community Cloud allows users to deploy an app directly from a GitHub repository and provides a shareable application URL. [5][230]

## Limitations

- The dataset contains a limited number of historical property records.
- The model is not validated for current market prices.
- Predictions may be unreliable for locations or feature combinations not represented in the training data.
- The model does not include features such as floor number, number of bedrooms, furnishing status, property condition, or legal approvals.
- The application is an educational prototype and should not be used as professional real-estate valuation advice.

## Future Improvements

- Use a larger and more recent Indian property dataset.
- Add features such as BHK, floor number, furnishing, property type, and amenities.
- Include uncertainty ranges or prediction intervals.
- Improve geographic validation using maps and location-based features.
- Deploy the application with a custom domain and monitoring.

## Author

**Harsh**  
Jodhpur, Rajasthan, India

## License

This project is created for educational purposes.