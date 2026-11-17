from pathlib import Path
import pickle

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="EstateWise | Property Valuation",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "model.pkl"


@st.cache_resource
def load_artifact():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Place model.pkl in the same "
        "directory as app.py, then restart the application."
    )
    st.stop()

try:
    artifact = load_artifact()
except Exception as error:  # noqa: BLE001 - surface any unpickling failure
    st.error(
        "The saved model could not be loaded. This usually means the "
        "installed scikit-learn version differs from the one used for "
        "training. Expected scikit-learn 1.9.1.\n\n"
        f"Details: {error}"
    )
    st.stop()
metrics = artifact["metrics"]
example = artifact["example_input"]


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
            margin-bottom: 0.5rem;
        }

        .hero-title {
            font-size: 3rem;
            font-weight: 750;
            letter-spacing: -0.045em;
            line-height: 1.1;
            margin-bottom: 0.75rem;
        }

        .hero-description {
            color: #64748b;
            font-size: 1.05rem;
            line-height: 1.7;
            max-width: 800px;
            margin-bottom: 1.5rem;
        }

        .section-description {
            color: #64748b;
            font-size: 0.95rem;
            margin-bottom: 1rem;
        }

        .footer-note {
            color: #64748b;
            font-size: 0.8rem;
            line-height: 1.7;
        }

        div.stButton > button,
        div.stFormSubmitButton > button {
            border-radius: 8px;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# Sidebar
with st.sidebar:
    st.title("EstateWise")
    st.caption("PROPERTY VALUATION WORKSPACE")

    st.divider()

    st.subheader("Project Overview")
    st.write(
        "Estimate historical property prices per unit area "
        "using a trained machine-learning model."
    )

    st.markdown(
        """
        - Model: Tuned Random Forest
        - Input features: 6
        - Dataset records: 414
        - Location: Sindian District, Taiwan
        """
    )

    st.divider()

    st.subheader("How to Use")
    st.markdown(
        """
        1. Review the pre-filled sample property.
        2. Adjust the property details.
        3. Select **Generate Estimate**.
        4. Review the estimate and its limitations.
        """
    )

    st.divider()

    st.caption(
        "Academic demonstration. Not a substitute for "
        "professional property valuation."
    )


# Header
st.markdown(
    '<div class="eyebrow">Machine Learning · Real Estate Analytics</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-title">Property Valuation Workspace</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-description">
        Explore a model-based estimate of historical property prices
        using property age, accessibility, nearby amenities,
        and geographic location.
    </div>
    """,
    unsafe_allow_html=True
)




# Performance overview
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.metric("Test R²", f"{metrics['test_r2']:.3f}")

with metric_col2:
    st.metric("Test MAE", f"{metrics['test_mae']:.3f}")

with metric_col3:
    st.metric("Test RMSE", f"{metrics['test_rmse']:.3f}")

with metric_col4:
    st.metric("Test Properties", metrics["test_properties"])

st.caption(
    "MAE and RMSE are expressed in 10,000 TWD per ping. "
    "R² is not an accuracy percentage."
)

st.divider()


def numeric_input(feature, label, help_text, step, display_format):
    return st.number_input(
        label=label,
        min_value=float(artifact["input_min"][feature]),
        max_value=float(artifact["input_max"][feature]),
        value=float(example[feature]),
        step=step,
        format=display_format,
        help=help_text
    )


# Property form
st.subheader("Property Details")

st.markdown(
    """
    <div class="section-description">
        Start with the pre-filled training-data example.
        Inputs are restricted to the training-data ranges.
    </div>
    """,
    unsafe_allow_html=True
)

with st.form("valuation_form"):
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("#### Property & Transaction")

                month_names = [
            "January", "February", "March", "April",
            "May", "June", "July", "August",
            "September", "October", "November", "December"
        ]

        # Build month options from the saved training-date range.
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
            year, zero_based_month = divmod(month_index - 1, 12)
            return f"{month_names[zero_based_month]} {year}"

        selected_month = st.selectbox(
            label="Historical Transaction Month",
            options=month_options,
            index=month_options.index(default_month),
            format_func=format_transaction_month,
            help=(
                "Select a transaction month within the training period. "
                "This is not a current-market valuation date."
            )
        )

        transaction_date = selected_month / 12.0

        house_age = numeric_input(
            feature="house_age",
            label="Property Age — Years",
            help_text="Age of the property at the time of transaction.",
            step=0.5,
            display_format="%.1f"
        )

        mrt_distance = numeric_input(
            feature="mrt_distance",
            label="Distance to Nearest MRT Station — Metres",
            help_text=(
                "Distance to the nearest Mass Rapid Transit station."
            ),
            step=10.0,
            display_format="%.2f"
        )

    with right:
        st.markdown("#### Amenities & Location")

        stores = st.number_input(
            label="Nearby Convenience Stores",
            min_value=int(artifact["input_min"]["stores"]),
            max_value=int(artifact["input_max"]["stores"]),
            value=int(example["stores"]),
            step=1,
            help="Number of nearby convenience stores recorded in the dataset."
        )

        latitude = numeric_input(
            feature="latitude",
            label="Latitude",
            help_text="Latitude within the dataset's geographic coverage.",
            step=0.0001,
            display_format="%.5f"
        )

        longitude = numeric_input(
            feature="longitude",
            label="Longitude",
            help_text="Longitude within the dataset's geographic coverage.",
            step=0.0001,
            display_format="%.5f"
        )

    st.caption(
        "Range checks do not guarantee that every combination "
        "of inputs represents a real property."
    )

    submitted = st.form_submit_button(
        "Generate Estimate",
        type="primary",
        width="stretch"
    )


# Prediction
if submitted:
    input_values = {
        "transaction_date": transaction_date,
        "house_age": house_age,
        "mrt_distance": mrt_distance,
        "stores": stores,
        "latitude": latitude,
        "longitude": longitude
    }

    input_df = pd.DataFrame(
        [input_values],
        columns=artifact["features"]
    )

    prediction = float(
        artifact["pipeline"].predict(input_df)[0]
    )

    st.divider()
    st.subheader("Valuation Result")

    if prediction <= 0:
        st.warning(
            "The model returned a non-positive estimate. "
            "This result should not be interpreted as a valid valuation."
        )
    else:
        result_col, context_col = st.columns(
            [1.2, 1],
            gap="large"
        )

        with result_col:
            st.metric(
                label="Estimated Price per Unit Area",
                value=f"{prediction:.2f}",
                help="Unit: 10,000 New Taiwan Dollars per ping."
            )

            st.write("Unit: 10,000 TWD per ping")

        with context_col:
            st.markdown("#### Interpretation")
            st.write(
                "This is a historical, model-based estimate "
                "of price per unit area—not the total property price."
            )
            st.write(
                "The estimate is not a guaranteed transaction price, "
                "a current market quotation, or a confidence interval."
            )

        with st.expander("View Submitted Property Details"):
            display_names = {
                "transaction_date": "Transaction Date",
                "house_age": "Property Age (Years)",
                "mrt_distance": "MRT Distance (Metres)",
                "stores": "Convenience Stores",
                "latitude": "Latitude",
                "longitude": "Longitude"
            }

            submitted_details = pd.DataFrame({
                "Feature": [
                    display_names[column]
                    for column in artifact["features"]
                ],
                "Value": [
                    input_values[column]
                    for column in artifact["features"]
                ]
            })

            st.dataframe(
                submitted_details,
                hide_index=True,
                width="stretch"
            )


# Methodology and limitations
st.divider()

with st.expander("Methodology & Model Evaluation"):
    st.markdown(
        """
        - Dataset: UCI Real Estate Valuation.
        - Training / test split: 331 / 83 properties.
        - Compared models: Linear Regression, Decision Tree,
          and Random Forest.
        - Model development: 5-fold cross-validation.
        - Hyperparameter tuning: GridSearchCV.
        - Final evaluation: Held-out test data.
        - Inference: Saved pipeline; no retraining on prediction.
        """
    )

    st.markdown("##### Selected Random Forest Settings")

    st.json(metrics["best_parameters"])

    st.caption(
        "Reported test metrics summarize performance on one "
        "held-out split. They do not establish performance in "
        "other locations or future market conditions."
    )

with st.expander("Scope & Limitations"):
    st.markdown(
        """
        - The model represents a small, geographically specific,
          historical dataset.
        - It is not validated for Indian properties or current prices.
        - The target is price per unit area, not total sale price.
        - Important property attributes outside the dataset
          are not represented.
        - Predictions are estimates, not professional advice.
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