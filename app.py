from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="EstateWise India | Property Valuation",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "model.pkl"


@st.cache_resource
def load_artifact(model_path, modified_time):
    with open(model_path, "rb") as file:
        return pickle.load(file)


if not MODEL_PATH.exists():
    st.error(
        "Model file not found. First run train_india_model.py, "
        "then place model.pkl in this folder."
    )
    st.stop()

try:
    artifact = load_artifact(
        str(MODEL_PATH),
        MODEL_PATH.stat().st_mtime_ns
    )
except Exception as exc:
    st.error("The saved model could not be loaded.")
    with st.expander("Technical details"):
        st.code(str(exc))
    st.stop()

metrics = artifact["metrics"]
example = artifact["example_input"]


st.markdown(
    """
    <style>
        .block-container {
            max-width: 1200px;
            padding-top: 2.5rem;
        }

        .hero-title {
            font-size: 2.7rem;
            font-weight: 750;
            letter-spacing: -0.04em;
            margin-bottom: 0.7rem;
        }

        .hero-description {
            color: #64748b;
            font-size: 1.05rem;
            line-height: 1.7;
        }
    </style>
    """,
    unsafe_allow_html=True
)


with st.sidebar:
    st.title("EstateWise India")
    st.caption("PROPERTY VALUATION WORKSPACE")

    st.divider()
    st.subheader("Project Overview")
    st.write(
        "An educational machine-learning application for "
        "Indian property price-per-sq-ft estimation."
    )

    st.write("Model: Tuned Random Forest")
    st.write("Dataset records: 414")
    st.write("Prediction unit: INR per sq ft")

    st.divider()
    st.subheader("How It Works")
    st.markdown(
        """
        1. Select the property location.
        2. Enter property and transaction details.
        3. Select **Generate Estimate**.
        """
    )

    st.caption(
        "Academic prototype. Not professional valuation advice."
    )


st.markdown(
    '<div class="hero-title">Property Valuation Workspace</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-description">
        Estimate historical property prices per square foot using
        property age, transport accessibility, nearby amenities,
        property area, and geographic location.
    </div>
    """,
    unsafe_allow_html=True
)




col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Held-Out Test R²", f"{metrics['test_r2']:.3f}")

with col2:
    st.metric("Test MAE", f"{metrics['test_mae']:.2f}")

with col3:
    st.metric("Test RMSE", f"{metrics['test_rmse']:.2f}")

with col4:
    st.metric("Test Properties", int(metrics["test_properties"]))

st.caption(
    "MAE and RMSE are measured in INR per square foot. "
    "R² is not an accuracy percentage."
)

st.divider()


st.subheader("Property Details")

with st.form("valuation_form"):
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("#### Location")

        city = st.selectbox(
            "City",
            options=artifact["cities"],
            index=artifact["cities"].index(example["City"])
        )

        state = st.selectbox(
            "State",
            options=artifact["states"],
            index=artifact["states"].index(example["State"])
        )

        locality = st.selectbox(
            "Locality",
            options=artifact["localities"],
            index=artifact["localities"].index(example["Locality"])
        )

        latitude = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=float(example["Latitude"]),
            step=0.0001,
            format="%.5f"
        )

        longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=float(example["Longitude"]),
            step=0.0001,
            format="%.5f"
        )

    with right:
        st.markdown("#### Property & Transaction")

        transaction_date = st.date_input(
            "Transaction Date",
            value=pd.Timestamp(
    year=int(example["Transaction_Year"]),
    month=int(example["Transaction_Month"]),
    day=1
    ).date()
        )

        house_age = st.number_input(
            "Property Age — Years",
            min_value=0.0,
            value=float(example["Property_Age_Years"]),
            step=1.0
        )

        metro_distance = st.number_input(
            "Distance to Metro — Kilometres",
            min_value=0.0,
            value=float(example["Distance_to_Metro_km"]),
            step=0.1,
            format="%.2f"
        )

        stores = st.number_input(
            "Nearby Convenience Stores",
            min_value=0,
            value=int(example["Nearby_Convenience_Stores"]),
            step=1
        )

        area_sqft = st.number_input(
            "Property Area — Square Feet",
            min_value=1.0,
            value=float(example["Property_Area_sqft"]),
            step=10.0,
            format="%.0f"
        )

    submitted = st.form_submit_button(
        "Generate Estimate",
        type="primary",
        use_container_width=True
    )


if submitted:
    transaction_timestamp = pd.Timestamp(transaction_date)

    input_values = {
        "Property_Age_Years": float(house_age),
        "Distance_to_Metro_km": float(metro_distance),
        "Nearby_Convenience_Stores": int(stores),
        "Latitude": float(latitude),
        "Longitude": float(longitude),
        "Property_Area_sqft": float(area_sqft),
        "City": city,
        "State": state,
        "Locality": locality,
        "Transaction_Year": transaction_timestamp.year,
        "Transaction_Month": transaction_timestamp.month,
        "Transaction_Seasonal_Value": np.sin(
            2 * np.pi * transaction_timestamp.month / 12
        ),
        "Transaction_Seasonal_Cosine": np.cos(
            2 * np.pi * transaction_timestamp.month / 12
        )
    }

    input_df = pd.DataFrame(
        [input_values],
        columns=artifact["features"]
    )

    try:
        prediction = float(
            artifact["pipeline"].predict(input_df)[0]
        )

        st.session_state["valuation_result"] = {
            "prediction": prediction,
            "inputs": input_values,
            "area_sqft": float(area_sqft)
        }

    except Exception as exc:
        st.session_state.pop("valuation_result", None)
        st.error("The estimate could not be generated.")

        with st.expander("Technical details"):
            st.code(str(exc))


if "valuation_result" in st.session_state:
    result = st.session_state["valuation_result"]
    prediction = result["prediction"]
    area = result["area_sqft"]

    st.divider()
    st.subheader("Valuation Result")

    if prediction <= 0:
        st.warning(
            "The model returned a non-positive estimate. "
            "Do not interpret it as a valid property valuation."
        )
    else:
        result_col, explanation_col = st.columns(
            [1.1, 1],
            gap="large"
        )

        with result_col:
            st.metric(
                "Estimated Price per Square Foot",
                f"₹{prediction:,.2f}"
            )

            st.caption("Unit: Indian Rupees per square foot")

            estimated_total = prediction * area

            st.metric(
                "Estimated Total Property Value",
                f"₹{estimated_total:,.0f}"
            )

            st.caption(
                f"Calculated as price per sq ft × {area:,.0f} sq ft."
            )

        with explanation_col:
            st.markdown("#### What This Means")
            st.write(
                "This is a historical price-per-square-foot estimate "
                "for the entered property details."
            )

            st.write(
                "It is not a guaranteed sale price, current market "
                "quotation, or professional valuation."
            )

        with st.expander("View Submitted Property Details"):
            inputs = result["inputs"]

            detail_table = pd.DataFrame({
                "Feature": [
                    "City",
                    "State",
                    "Locality",
                    "Transaction Date",
                    "Property Age",
                    "Metro Distance",
                    "Nearby Convenience Stores",
                    "Property Area",
                    "Latitude",
                    "Longitude"
                ],
                "Submitted Value": [
                    inputs["City"],
                    inputs["State"],
                    inputs["Locality"],
                    transaction_date.strftime("%d %B %Y"),
                    f"{inputs['Property_Age_Years']:.0f} years",
                    f"{inputs['Distance_to_Metro_km']:.2f} km",
                    str(inputs["Nearby_Convenience_Stores"]),
                    f"{inputs['Property_Area_sqft']:,.0f} sq ft",
                    f"{inputs['Latitude']:.5f}",
                    f"{inputs['Longitude']:.5f}"
                ]
            })

            st.dataframe(
                detail_table,
                hide_index=True,
                use_container_width=True
            )


st.divider()

with st.expander("Methodology & Model Evaluation"):
    st.markdown(
        """
        - Dataset: India real-estate valuation dataset
        - Total records: 414
        - Training / test split: 331 / 83 properties
        - Compared models: Linear Regression, Decision Tree, Random Forest
        - Model comparison: Shuffled 5-fold cross-validation
        - Hyperparameter tuning: GridSearchCV
        - Final evaluation: Held-out test subset
        - Target: Price per square foot in INR
        - Application inference: Saved pipeline, without retraining
        """
    )

    st.markdown("##### Model Comparison")
    st.dataframe(
        pd.DataFrame(metrics["comparison"]),
        hide_index=True,
        use_container_width=True
    )

    st.markdown("##### Selected Hyperparameters")
    st.json(metrics["best_parameters"])

    st.caption(
        "Results describe one held-out split. They do not establish "
        "performance across all Indian cities or future transactions."
    )

with st.expander("Scope & Limitations"):
    st.markdown(
        """
        - The model is trained on a small, historical dataset.
        - It is not validated for current market prices.
        - The primary prediction is price per square foot in INR.
        - Total value is calculated by multiplying the estimate by
          entered property area.
        - Locality, city, and state are categorical inputs.
        - Coordinates outside the training-data coverage may produce
          unreliable estimates.
        - This is an educational prototype, not professional valuation advice.
        """
    )

st.markdown(
    """
    <div class="hero-description">
        EstateWise India · Academic Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True
)