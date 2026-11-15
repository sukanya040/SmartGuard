
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="SmartGuard | Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)





# --------------------------------------------------
# LOAD FINAL MODEL
# --------------------------------------------------

from pathlib import Path

MODEL_PATH = Path(__file__).parent / "smartguard_model_final.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

bundle = load_model()

model = bundle["model"]
threshold = bundle["threshold"]

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🛡️ SmartGuard")
st.sidebar.write("Intelligent Fraud Detection System")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Transaction Detection",
        "Risk Analysis",
        "Model Information"
    ]
)

if "results" not in st.session_state:
    st.session_state.results = None




# --
------------------------------------------------
# HOME PAGE
# --------------------------------------------------

if page == "Home":

    # Main hero banner
    st.markdown("""
    <div style="
        background: linear-gradient(120deg, #17243A, #1E3A5F);
        padding: 40px;
        border-radius: 18px;
        border: 1px solid #345477;
        margin-bottom: 28px;
    ">
        <h1 style="color: #FFFFFF; font-size: 42px;">
            🛡️ SmartGuard
        </h1>

        <h3 style="color: #60A5FA;">
            Smarter Fraud Detection. Safer E-commerce.
        </h3>

        <p style="color: #E2E8F0; font-size: 17px;">
            Cost-Sensitive and Explainable E-commerce Fraud Detection
            using Machine Learning.
        </p>

        <p style="color: #CBD5E1; font-size: 15px;">
            Identify suspicious transactions, estimate fraud probabilities,
            and support risk-based decisions to help reduce potential
            financial losses.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Model summary
    st.subheader("Model Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Selected Model", "XGBoost")

    col2.metric(
        "Features Used",
        len(bundle["feature_columns"])
    )

    col3.metric(
        "Fraud Threshold",
        f"{threshold:.2f}"
    )

    st.divider()

    # Platform capabilities
    st.subheader("Platform Capabilities")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div style="
            background: #17243A;
            padding: 22px;
            border: 1px solid #263B59;
            border-radius: 14px;
            margin-bottom: 18px;
        ">
            <h3 style="color: #60A5FA;">
                🔍 Transaction Detection
            </h3>
            <p style="color: #CBD5E1;">
                Upload transaction data and identify potentially
                fraudulent activities using the trained model.
            </p>
        </div>

        <div style="
            background: #17243A;
            padding: 22px;
            border: 1px solid #263B59;
            border-radius: 14px;
        ">
            <h3 style="color: #60A5FA;">
                📊 Risk Analysis
            </h3>
            <p style="color: #CBD5E1;">
                Explore low, medium, and high-risk transactions
                through visual summaries.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div style="
            background: #17243A;
            padding: 22px;
            border: 1px solid #263B59;
            border-radius: 14px;
            margin-bottom: 18px;
        ">
            <h3 style="color: #60A5FA;">
                ⚠️ Risk Recommendations
            </h3>
            <p style="color: #CBD5E1;">
                Receive recommended actions based on the
                predicted transaction risk level.
            </p>
        </div>

        <div style="
            background: #17243A;
            padding: 22px;
            border: 1px solid #263B59;
            border-radius: 14px;
        ">
            <h3 style="color: #60A5FA;">
                📥 Prediction Reports
            </h3>
            <p style="color: #CBD5E1;">
                Download transaction prediction results
                for further analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.caption(
        "SmartGuard | Machine Learning-Based E-commerce Fraud Detection"
    )

# --------------------------------------------------
# TRANSACTION DETECTION
# --------------------------------------------------

elif page == "Transaction Detection":

    st.title("🔍 Transaction Detection")

    st.write(
        "Upload transaction data to detect potentially fraudulent transactions."
    )

    transaction_file = st.file_uploader(
        "Upload Transaction CSV",
        type=["csv"],
        key="transaction_file"
    )

    identity_file = st.file_uploader(
        "Upload Identity CSV (Optional)",
        type=["csv"],
        key="identity_file"
    )

    if st.button("Detect Fraud", type="primary"):

        if transaction_file is None:

            st.warning("Please upload a transaction CSV file.")

        else:

            try:

                # Read transaction data
                df = pd.read_csv(transaction_file)

                # Merge identity data if uploaded
                if identity_file is not None:

                    identity_df = pd.read_csv(identity_file)

                    if "TransactionID" not in identity_df.columns:
                        raise ValueError(
                            "Identity CSV must contain TransactionID."
                        )

                    df = df.merge(
                        identity_df,
                        on="TransactionID",
                        how="left"
                    )

                # ------------------------------------------
                # BEHAVIOURAL FEATURE GENERATION
                # ------------------------------------------

                history = bundle["card1_behavior_history"]

                if "card1" in df.columns:

                    # Handle card1 values consistently
                    history_lookup = {
                        str(k): v for k, v in history.items()
                    }

                    card_values = df["card1"].astype(str)

                    card_history = card_values.map(history_lookup)

                    def get_history_value(value, field):
                        if isinstance(value, dict):
                            return value.get(field, 0)

                        if isinstance(value, (list, tuple)):
                            if field == "count" and len(value) > 0:
                                return value[0]
                            if field == "sum" and len(value) > 1:
                                return value[1]

                        return 0

                    counts = card_history.apply(
                        lambda x: get_history_value(x, "count")
                    )

                    totals = card_history.apply(
                        lambda x: get_history_value(x, "sum")
                    )

                    averages = np.where(
                        pd.to_numeric(counts, errors="coerce") > 0,
                        pd.to_numeric(totals, errors="coerce") /
                        np.maximum(
                            pd.to_numeric(counts, errors="coerce"), 1
                        ),
                        np.nan
                    )

                    df["card1_previous_count"] = counts
                    df["card1_previous_avg_amount"] = averages

                    if "TransactionAmt" in df.columns:

                        amount = pd.to_numeric(
                            df["TransactionAmt"],
                            errors="coerce"
                        )

                        df["card1_amount_deviation"] = amount - averages

                    else:

                        df["card1_amount_deviation"] = np.nan

                else:

                    df["card1_previous_count"] = np.nan
                    df["card1_previous_avg_amount"] = np.nan
                    df["card1_amount_deviation"] = np.nan

                # ------------------------------------------
                # FEATURE PREPARATION
                # ------------------------------------------

                feature_columns = bundle["feature_columns"]

                X = df.reindex(
                    columns=feature_columns
                ).copy()

                # Numerical preprocessing
                for col in bundle["numerical_features"]:

                    X[col] = pd.to_numeric(
                        X[col],
                        errors="coerce"
                    )

                    X[col] = X[col].replace(
                        [np.inf, -np.inf],
                        np.nan
                    )

                    X[col] = X[col].fillna(
                        bundle["numerical_medians"][col]
                    )

                # Categorical preprocessing
                for col in bundle["categorical_features"]:

                    X[col] = (
                        X[col]
                        .fillna("Missing")
                        .astype(str)
                        .map(
                            bundle["categorical_frequency_maps"][col]
                        )
                        .fillna(0)
                    )

                X = X.astype(np.float32)

                # ------------------------------------------
                # FRAUD PREDICTION
                # ------------------------------------------

                probabilities = model.predict_proba(X)[:, 1]

                predictions = probabilities >= threshold

                # Risk thresholds
                risk_thresholds = bundle["risk_thresholds"]

                low_limit = risk_thresholds.get("low", 0.30)

                risk_levels = np.select(
                    [
                        probabilities < low_limit,
                        probabilities < threshold,
                        probabilities >= threshold
                    ],
                    [
                        "Low",
                        "Medium",
                        "High"
                    ],
                    default="Low"
                )

                # Recommended actions
                actions = {
                    "Low": "Approve Transaction",
                    "Medium": "Send for Manual Review",
                    "High": "Hold Transaction and Investigate"
                }

                results = pd.DataFrame({

                    "Fraud Probability": probabilities,

                    "Prediction": np.where(
                        predictions,
                        "Fraud",
                        "Genuine"
                    ),

                    "Risk Level": risk_levels

                })

                results["Recommended Action"] = (
                    results["Risk Level"].map(actions)
                )

                if "TransactionID" in df.columns:

                    results.insert(
                        0,
                        "TransactionID",
                        df["TransactionID"].values
                    )

                st.session_state.results = results

                st.success(
                    "Transaction fraud detection completed!"
                )

            except Exception as e:

                st.error("Unable to process the uploaded file.")
                st.exception(e)

    # Display predictions
    if st.session_state.results is not None:

        results = st.session_state.results

        st.subheader("Prediction Results")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Transactions",
            len(results)
        )

        col2.metric(
            "Potential Fraud",
            int((results["Prediction"] == "Fraud").sum())
        )

        col3.metric(
            "Genuine Transactions",
            int((results["Prediction"] == "Genuine").sum())
        )

        st.dataframe(
            results,
            use_container_width=True
        )

        # Download predictions
        csv = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Prediction Results",
            data=csv,
            file_name="SmartGuard_Predictions.csv",
            mime="text/csv"
        )


# --------------------------------------------------
# RISK ANALYSIS
# --------------------------------------------------

elif page == "Risk Analysis":

    st.title("📊 Risk Analysis")

    results = st.session_state.results

    if results is None:

        st.info(
            "Please run Transaction Detection first to view risk analysis."
        )

    else:

        low = int(
            (results["Risk Level"] == "Low").sum()
        )

        medium = int(
            (results["Risk Level"] == "Medium").sum()
        )

        high = int(
            (results["Risk Level"] == "High").sum()
        )

        col1, col2, col3 = st.columns(3)

        col1.metric("Low Risk", low)
        col2.metric("Medium Risk", medium)
        col3.metric("High Risk", high)

        st.subheader("Risk Distribution")

        risk_data = pd.DataFrame({
            "Risk Level": ["Low", "Medium", "High"],
            "Transactions": [low, medium, high]
        })

        st.bar_chart(
            risk_data.set_index("Risk Level")
        )

        st.subheader("High-Risk Transactions")

        high_risk = results[
            results["Risk Level"] == "High"
        ]

        if len(high_risk) > 0:

            st.dataframe(
                high_risk,
                use_container_width=True
            )

        else:

            st.success("No high-risk transactions detected.")

        st.caption(
            "Risk categories are model predictions, not confirmed fraud labels."
        )


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

elif page == "Model Information":

    st.title("🤖 Model Information")

    st.subheader("Model Details")

    st.write("Machine Learning Model: XGBoost")

    st.write("Dataset: IEEE-CIS Fraud Detection")

    st.write(
        "Total Features:",
        len(bundle["feature_columns"])
    )

    st.write(
        "Numerical Features:",
        len(bundle["numerical_features"])
    )

    st.write(
        "Categorical Features:",
        len(bundle["categorical_features"])
    )

    st.write(
        "Fraud Decision Threshold:",
        threshold
    )

    st.divider()

    st.subheader("Important Features")

    importance = pd.DataFrame({

        "Feature": bundle["feature_columns"],

        "Importance": model.feature_importances_

    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    ).head(15)

    st.bar_chart(
        importance.set_index("Feature")
    )

    st.info("""
    SmartGuard uses XGBoost with cost-sensitive threshold selection
    and behavioural transaction features.

    Feature importance shows which features contribute most to
    the trained model overall. It is not an individual SHAP explanation.
    """)
