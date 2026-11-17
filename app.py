from pathlib import Path
import pickle

import pandas as pd
import streamlit as st


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="EstateWise | Property Valuation",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "model.pkl"


# --------------------------------------------------
# Load the trusted, locally generated model artifact
# --------------------------------------------------
@st.cache_resource
def load_artifact(model_path, modified_time):
    with open(model_path, "rb") as file:
        return pickle.load(file)


if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Place model.pkl in the same "
        "directory as app.py."
    )
    st.stop()

try:
    artifact = load_artifact(
        str(MODEL_PATH),
        MODEL_PATH.stat().st_mtime_ns
    )
except Exception as exc:
    st.error(
        "The saved model could not be loaded. "
        "Use the Python environment in which the model was trained."
    )
    with st.expander("Technical details"):
        st.code(str(exc))
    st.stop()

required_keys = {
    "pipeline",
    "features",
    "input_min",
    "input_max",
    "example_input",
    "metrics"
}

if not required_keys.issubset(artifact):
    st.error("The model artifact is missing required application metadata.")
    st.stop()

metrics = artifact["metrics"]
example = artifact["example_input"]


# --------------------------------------------------
# Styling
# --------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {
            max-width: 1200px;
            padding-top: 2.5rem;
            padding-bottom: 2rem;
        }

        .eyebrow {
            color: #64748b;
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 0.6rem;
        }

        .hero-title {
            font-size: 2.8rem;
            font-weight: 750;
            letter-spacing: -0.04em;
            line-height: 1.15;
            margin-bottom: 0.8rem;
        }

        .hero-description {
            color: #64748b;
            font-size: 1.05rem;
            line-height: 1.7;
            max-width: 850px;
            margin-bottom: 1.5rem;
        }

        .footer-note {
            color: #64748b;
            font-size: 0.8rem;
            line-height: 1.7;
        }

        div.stFormSubmitButton > button {
            border-radius: 8px;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Date options matching the dataset's month encoding
# --------------------------------------------------
MONTH_NAMES = [
    "January", "February", "March", "April",
    "May", "June", "July", "August",
    "September", "October", "November", "December"
]

first_month = int(round(
    float(artifact["input_min"]["transaction_date"]) * 12
))

last_month = int(round(
    float(artifact["input_max"]["transaction_date"]) * 12
))

month_options = list(range(first_month, last_month + 1))

default_month = int(round(
    float(example["transaction_date"]) * 12
))


def format_transaction_month(month_index):
    year, zero_based_month = divmod(int(month_index) - 1, 12)
    return f"{MONTH_NAMES[zero_based_month]} {year}"


def training_range(feature, decimals=2):
    minimum = float(artifact["input_min"][feature])
    maximum = float(artifact["input_max"][feature])
    return f"{minimum:.{decimals}f} to {maximum:.{decimals}f}"


def reset_form():
    st.session_state["transaction_month"] = default_month
    st.session_state["house_age"] = float(example["house_age"])
    st.session_state["mrt_distance"] = float(example["mrt_distance"])
    st.session_state["stores"] = int(example["stores"])
    st.session_state["latitude"] = float(example["latitude"])
    st.session_state["longitude"] = float(example["longitude"])
    st.session_state.pop("valuation_result", None)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.title("EstateWise")
    st.caption("PROPERTY VALUATION WORKSPACE")

    st.divider()

    st.subheader("Project Overview")
    st.write(
        "An educational machine-learning application for "
        "historical property price estimation."
    )

    st.write("Model: Tuned Random Forest")
    st.write("Dataset records: 414")
    st.write("Input features: 6")
    st.write("Location: Sindian District, Taiwan")

    st.divider()

    st.subheader("How It Works")
    st.markdown(
        """
        1. Review or modify the property details.
        2. Select **Generate Estimate**.
        3. Review the result and coverage warnings.
        """
    )

    st.button(
        "Reset to Example Property",
        on_click=reset_form,
        use_container_width=True
    )

    st.divider()

    st.caption(
        "The example is an existing training-data record. "
        "Its prediction is a demonstration, not an independent "
        "performance test."
    )

    st.caption(
        "Academic prototype. Not professional valuation advice."
    )


# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    '<div class="eyebrow">Real Estate Analytics · Machine Learning</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-title">Property Valuation Workspace</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-description">
        Estimate historical property prices per unit area using
        property age, transport accessibility, nearby amenities,
        and geographic location.
    </div>
    """,
    unsafe_allow_html=True
)

st.info(
    "Model scope: Historical transactions from Sindian District, "
    "New Taipei City, Taiwan. This application is not validated "
    "for current market prices or other property markets."
)


# --------------------------------------------------
# Evaluation overview
# --------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Held-Out Test R²", f"{metrics['test_r2']:.3f}")

with col2:
    st.metric("Test MAE", f"{metrics['test_mae']:.3f}")

with col3:
    st.metric("Test RMSE", f"{metrics['test_rmse']:.3f}")

with col4:
    st.metric("Test Properties", int(metrics["test_properties"]))

st.caption(
    "MAE and RMSE are measured in 10,000 TWD per ping. "
    "R² is not an accuracy percentage. These are aggregate test "
    "metrics, not uncertainty bounds for an individual prediction."
)

st.divider()


# --------------------------------------------------
# Input form
# --------------------------------------------------
st.subheader("Property Details")

st.write(
    "Start with the example property or enter your own values. "
    "Coordinates can extend beyond the training-data ranges; "
    "unsupported ranges receive a coverage warning."
)

with st.form("valuation_form"):
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("#### Property & Transaction")

        selected_month = st.selectbox(
            "Historical Transaction Month",
            options=month_options,
            index=month_options.index(default_month),
            format_func=format_transaction_month,
            key="transaction_month",
            help=(
                "Transaction month within the training period. "
                "The application converts this selection into "
                "the dataset's original numerical encoding."
            )
        )

        transaction_date = selected_month / 12.0

        house_age = st.number_input(
            "Property Age — Years",
            min_value=0.0,
            value=float(example["house_age"]),
            step=0.5,
            format="%.1f",
            key="house_age",
            help=(
                "Property age at transaction. "
                f"Observed training range: {training_range('house_age', 1)} years."
            )
        )

        mrt_distance = st.number_input(
            "Distance to Nearest MRT Station — Metres",
            min_value=0.0,
            value=float(example["mrt_distance"]),
            step=10.0,
            format="%.2f",
            key="mrt_distance",
            help=(
                "Distance to the nearest Mass Rapid Transit station. "
                f"Observed training range: {training_range('mrt_distance')} metres."
            )
        )

    with right:
        st.markdown("#### Amenities & Location")

        stores = st.number_input(
            "Nearby Convenience Stores",
            min_value=0,
            value=int(example["stores"]),
            step=1,
            key="stores",
            help=(
                "Number of nearby convenience stores. "
                f"Observed training range: {training_range('stores', 0)}."
            )
        )

        latitude = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=float(example["latitude"]),
            step=0.0001,
            format="%.5f",
            key="latitude",
            help=(
                "Geographic latitude. "
                f"Observed training range: {training_range('latitude', 5)}. "
                "Other valid coordinates are accepted with a warning."
            )
        )

        longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=float(example["longitude"]),
            step=0.0001,
            format="%.5f",
            key="longitude",
            help=(
                "Geographic longitude. "
                f"Observed training range: {training_range('longitude', 5)}. "
                "Other valid coordinates are accepted with a warning."
            )
        )

    st.caption(
        "Input validation checks basic validity. It does not establish "
        "that a property or feature combination exists or is represented "
        "by the training data."
    )

    submitted = st.form_submit_button(
        "Generate Estimate",
        type="primary",
        use_container_width=True
    )


# --------------------------------------------------
# Prediction and coverage checks
# --------------------------------------------------
if submitted:
    input_values = {
        "transaction_date": float(transaction_date),
        "house_age": float(house_age),
        "mrt_distance": float(mrt_distance),
        "stores": int(stores),
        "latitude": float(latitude),
        "longitude": float(longitude)
    }

    display_labels = {
        "transaction_date": "Transaction Month",
        "house_age": "Property Age",
        "mrt_distance": "MRT Distance",
        "stores": "Convenience Stores",
        "latitude": "Latitude",
        "longitude": "Longitude"
    }

    outside_features = []

    for feature in artifact["features"]:
        entered = input_values[feature]
        minimum = float(artifact["input_min"][feature])
        maximum = float(artifact["input_max"][feature])

        if entered < minimum - 1e-6 or entered > maximum + 1e-6:
            outside_features.append(display_labels[feature])

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
            "transaction_month": format_transaction_month(selected_month),
            "outside_features": outside_features
        }

    except Exception as exc:
        st.session_state.pop("valuation_result", None)
        st.error("The estimate could not be generated.")

        with st.expander("Technical details"):
            st.code(str(exc))


# --------------------------------------------------
# Result display
# --------------------------------------------------
if "valuation_result" in st.session_state:
    result = st.session_state["valuation_result"]
    prediction = result["prediction"]

    st.divider()
    st.subheader("Valuation Result")

    st.caption(
        "This result corresponds to the last submitted inputs. "
        "After changing the form, select Generate Estimate again."
    )

    if result["outside_features"]:
        st.warning(
            "Outside observed training ranges: "
            + ", ".join(result["outside_features"])
            + ". The output below is exploratory and is not "
            "validated for these inputs."
        )
    else:
        st.caption(
            "All submitted values are within their individual training "
            "ranges. This does not guarantee geographic coverage, "
            "a realistic combination, or prediction reliability."
        )

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
                "Historical Price-per-Unit-Area Estimate",
                f"{prediction:.2f}"
            )

            st.write("Unit: 10,000 TWD per ping")
            st.caption("TWD: New Taiwan Dollar")

        with explanation_col:
            st.markdown("#### What This Means")
            st.write(
                "This is an estimate of historical price per unit area, "
                "not the total sale price of the property."
            )
            st.write(
                "It is not a current market quotation, guaranteed "
                "transaction price, or prediction interval."
            )

        with st.expander("View Submitted Property Details"):
            submitted_inputs = result["inputs"]

            detail_table = pd.DataFrame({
                "Feature": [
                    "Transaction Month",
                    "Property Age (Years)",
                    "MRT Distance (Metres)",
                    "Nearby Convenience Stores",
                    "Latitude",
                    "Longitude"
                ],
                "Submitted Value": [
                    result["transaction_month"],
                    f"{submitted_inputs['house_age']:.1f}",
                    f"{submitted_inputs['mrt_distance']:.2f}",
                    str(submitted_inputs["stores"]),
                    f"{submitted_inputs['latitude']:.5f}",
                    f"{submitted_inputs['longitude']:.5f}"
                ]
            })

            st.dataframe(
                detail_table,
                hide_index=True,
                use_container_width=True
            )


# --------------------------------------------------
# Documentation
# --------------------------------------------------
st.divider()

with st.expander("Observed Training-Data Ranges"):
    range_table = pd.DataFrame({
        "Feature": [
            "Transaction Period",
            "Property Age (Years)",
            "MRT Distance (Metres)",
            "Nearby Convenience Stores",
            "Latitude",
            "Longitude"
        ],
        "Observed Range": [
            (
                f"{format_transaction_month(first_month)} – "
                f"{format_transaction_month(last_month)}"
            ),
            training_range("house_age", 1),
            training_range("mrt_distance", 2),
            training_range("stores", 0),
            training_range("latitude", 5),
            training_range("longitude", 5)
        ]
    })

    st.dataframe(
        range_table,
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "These are separate feature ranges, not a verified service-area "
        "boundary. Outside-range predictions are exploratory."
    )

with st.expander("Methodology & Model Evaluation"):
    st.markdown(
        """
        - Dataset: UCI Real Estate Valuation
        - Total records: 414
        - Training / test split: 331 / 83 properties
        - Compared models: Linear Regression, Decision Tree, Random Forest
        - Model comparison: Shuffled 5-fold cross-validation
        - Hyperparameter tuning: GridSearchCV
        - Final evaluation: Held-out test subset
        - Application inference: Saved pipeline, without retraining
        """
    )

    st.markdown("##### Selected Hyperparameters")
    st.json(metrics["best_parameters"])

    st.caption(
        "Results describe one held-out split. They do not establish "
        "performance across different markets or future transactions."
    )

with st.expander("Scope & Limitations"):
    st.markdown(
        """
        - The model represents a small, historical, geographically
          specific dataset.
        - It is not validated for current market prices or Indian properties.
        - The output is price per unit area, not total property value.
        - Features such as floor area, BHK, and property condition are not
          included in this model.
        - Basic input validity does not guarantee a realistic property.
        - Coordinates outside the observed training ranges are accepted
          for exploration, not as supported valuation locations.
        - The application does not provide a calibrated uncertainty interval.
        - This is an educational prototype, not professional valuation advice.
        """
    )

st.markdown(
    """
    <div class="footer-note">
        EstateWise · Academic Machine Learning Project<br>
        Data source: UCI Machine Learning Repository —
        Real Estate Valuation.
    </div>
    """,
    unsafe_allow_html=True
)