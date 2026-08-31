import os
import joblib
import pandas as pd
import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CreditScoring AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        opacity: 0.75;
        margin-bottom: 25px;
    }

    .risk-card {
        padding: 25px;
        border-radius: 18px;
        text-align: center;
        border: 1px solid rgba(128,128,128,0.25);
        margin-top: 10px;
    }

    .risk-number {
        font-size: 38px;
        font-weight: 800;
        margin: 8px 0;
    }

    .risk-label {
        font-size: 20px;
        font-weight: 700;
    }

    .metric-card {
        padding: 18px;
        border-radius: 15px;
        border: 1px solid rgba(128,128,128,0.25);
        text-align: center;
    }

    .metric-title {
        font-size: 14px;
        opacity: 0.7;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 700;
    }

    .section-title {
        font-size: 25px;
        font-weight: 700;
        margin-top: 20px;
    }

    .footer {
        text-align: center;
        opacity: 0.55;
        font-size: 13px;
        padding: 25px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "credit_scoring_optimized.pkl"
)

try:
    package = joblib.load(MODEL_PATH)

    model = package["model"]
    imputer = package["imputer"]
    threshold = package["threshold"]
    features = package["features"]

except Exception as e:

    st.error(
        "Unable to load the trained model."
    )

    st.exception(e)

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 💳 CreditScoring AI")

    st.caption(
        "Machine Learning Credit Risk Assessment"
    )

    st.divider()

    st.markdown("### 🤖 Model")

    st.write("**Algorithm:** Random Forest")

    st.write("**ROC-AUC:** 85.95%")

    st.write("**F1 Score:** 42.47%")

    st.write("**Decision Threshold:** 0.60")

    st.divider()

    st.markdown("### 📊 Risk Scale")

    st.write("🟢 **Low Risk** — < 30%")

    st.write("🟡 **Medium Risk** — 30–60%")

    st.write("🔴 **High Risk** — ≥ 60%")

    st.divider()

    st.caption(
        "Educational ML project developed for "
        "CodeAlpha Internship."
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">💳 CreditScoring AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered credit risk assessment using Random Forest'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">👤 Customer Financial Profile</div>',
    unsafe_allow_html=True
)

st.caption(
    "Enter the customer's financial and credit-history information."
)

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

    debt_ratio = st.number_input(
        "Debt Ratio",
        min_value=0.0,
        value=0.30,
        step=0.01,
        format="%.2f"
    )

    revolving_utilization = st.number_input(
        "Revolving Credit Utilization",
        min_value=0.0,
        value=0.50,
        step=0.01,
        format="%.2f"
    )

    dependents = st.number_input(
        "Number of Dependents",
        min_value=0,
        max_value=20,
        value=2
    )


with col2:

    open_credit = st.number_input(
        "Open Credit Lines & Loans",
        min_value=0,
        max_value=100,
        value=8
    )

    late_30_59 = st.number_input(
        "30–59 Days Past Due",
        min_value=0,
        max_value=50,
        value=0
    )

    late_60_89 = st.number_input(
        "60–89 Days Past Due",
        min_value=0,
        max_value=50,
        value=0
    )

    late_90 = st.number_input(
        "90+ Days Late",
        min_value=0,
        max_value=50,
        value=0
    )

    real_estate = st.number_input(
        "Real Estate Loans / Lines",
        min_value=0,
        max_value=50,
        value=1
    )


# ============================================================
# PREDICT BUTTON
# ============================================================

st.divider()

predict = st.button(
    "🔍  ANALYZE CREDIT RISK",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict:

    input_data = {

        "RevolvingUtilizationOfUnsecuredLines":
            revolving_utilization,

        "age":
            age,

        "NumberOfTime30-59DaysPastDueNotWorse":
            late_30_59,

        "DebtRatio":
            debt_ratio,

        "MonthlyIncome":
            monthly_income,

        "NumberOfOpenCreditLinesAndLoans":
            open_credit,

        "NumberOfTimes90DaysLate":
            late_90,

        "NumberRealEstateLoansOrLines":
            real_estate,

        "NumberOfTime60-89DaysPastDueNotWorse":
            late_60_89,

        "NumberOfDependents":
            dependents
    }

    input_df = pd.DataFrame([input_data])

    input_df = input_df[features]

    input_processed = pd.DataFrame(
        imputer.transform(input_df),
        columns=features
    )

    probability = model.predict_proba(
        input_processed
    )[0][1]

    probability_percent = probability * 100


    # ========================================================
    # RISK LEVEL
    # ========================================================

    if probability < 0.30:

        risk_level = "LOW RISK"
        risk_icon = "🟢"
        message = (
            "The model estimates a relatively lower "
            "credit-risk probability."
        )

    elif probability < 0.60:

        risk_level = "MEDIUM RISK"
        risk_icon = "🟡"
        message = (
            "The model estimates a moderate "
            "credit-risk probability."
        )

    else:

        risk_level = "HIGH RISK"
        risk_icon = "🔴"
        message = (
            "The model estimates a relatively higher "
            "credit-risk probability."
        )


    # ========================================================
    # RESULTS
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">📊 Credit Risk Assessment</div>',
        unsafe_allow_html=True
    )

    r1, r2, r3 = st.columns(3)

    with r1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">
                    RISK PROBABILITY
                </div>
                <div class="metric-value">
                    {probability_percent:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with r2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">
                    RISK LEVEL
                </div>
                <div class="metric-value">
                    {risk_icon} {risk_level}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with r3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">
                    DECISION THRESHOLD
                </div>
                <div class="metric-value">
                    {threshold:.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # PROBABILITY BAR
    # ========================================================

    st.markdown("### Risk Probability")

    st.progress(
        min(probability, 1.0)
    )

    st.info(message)


    # ========================================================
    # CUSTOMER SUMMARY
    # ========================================================

    with st.expander(
        "👤 View Customer Input Summary"
    ):

        summary = pd.DataFrame({
            "Parameter": [
                "Age",
                "Monthly Income",
                "Debt Ratio",
                "Credit Utilization",
                "Dependents",
                "Open Credit Lines",
                "30–59 Days Late",
                "60–89 Days Late",
                "90+ Days Late",
                "Real Estate Loans"
            ],

            "Value": [
                age,
                f"{monthly_income:,.2f}",
                debt_ratio,
                revolving_utilization,
                dependents,
                open_credit,
                late_30_59,
                late_60_89,
                late_90,
                real_estate
            ]
        })

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">🤖 Model Performance</div>',
    unsafe_allow_html=True
)

m1, m2, m3, m4 = st.columns(4)

with m1:

    st.metric(
        "Accuracy",
        "91.76%"
    )

with m2:

    st.metric(
        "Precision",
        "39.80%"
    )

with m3:

    st.metric(
        "Recall",
        "45.54%"
    )

with m4:

    st.metric(
        "ROC-AUC",
        "85.95%"
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.warning(
    "⚠️ This application is an educational machine-learning "
    "demonstration. The prediction should not be used as the "
    "sole basis for real-world lending or financial decisions."
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        CodeAlpha Machine Learning Internship •
        Credit Scoring Model • Random Forest
    </div>
    """,
    unsafe_allow_html=True
)