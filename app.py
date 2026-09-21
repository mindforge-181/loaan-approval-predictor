"""
app.py
Project: Loan Approval Predictor

Purpose:
Streamlit frontend that collects applicant details,
sends them to the FastAPI backend, and displays
the loan approval prediction and probability.

Run with:

    python -m streamlit run app.py
"""

import streamlit as st
import requests


# ---------------------------------------------------------
# 1. FastAPI URL
# ---------------------------------------------------------

API_URL = "http://localhost:8000/predict"


# ---------------------------------------------------------
# 2. Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout="centered"
)


# ---------------------------------------------------------
# 3. Title
# ---------------------------------------------------------

st.title("💰 Loan Approval Predictor")

st.write(
    "Enter the applicant's details below to "
    "check the loan approval prediction."
)


# ---------------------------------------------------------
# 4. Input form
# ---------------------------------------------------------

with st.form("loan_form"):

    col1, col2 = st.columns(2)


    with col1:

        income = st.number_input(
            "Annual Income",
            min_value=1.0,
            max_value=1_000_000.0,
            value=50_000.0,
            step=1_000.0
        )


        credit_score = st.number_input(
            "Credit Score",
            min_value=300.0,
            max_value=900.0,
            value=650.0,
            step=1.0
        )


    with col2:

        loan_amount = st.number_input(
            "Loan Amount Requested",
            min_value=1.0,
            max_value=1_000_000.0,
            value=20_000.0,
            step=1_000.0
        )


        employment_years = st.number_input(
            "Years of Employment",
            min_value=0.0,
            max_value=50.0,
            value=3.0,
            step=1.0
        )


    submitted = st.form_submit_button(
        "Check Approval"
    )


# ---------------------------------------------------------
# 5. Send request to FastAPI
# ---------------------------------------------------------

if submitted:

    payload = {
        "income": income,
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "employment_years": employment_years
    }


    try:

        response = requests.post(
            API_URL,
            json=payload,
            timeout=5
        )


        response.raise_for_status()


        result = response.json()


        prediction = result.get("prediction")

        loan_status = result.get("loan_status")

        probability = result.get(
            "approval_probability"
        )


        # -------------------------------------------------
        # 6. Display result
        # -------------------------------------------------

        st.divider()


        if loan_status == "Approved":

            st.success("✅ Loan Approved")


        elif loan_status == "Rejected":

            st.error("❌ Loan Rejected")


        else:

            st.warning(
                f"Unexpected API response: {result}"
            )


        # -------------------------------------------------
        # 7. Display probability
        # -------------------------------------------------

        if probability is not None:

            probability = float(probability)


            st.metric(
                "Approval Probability",
                f"{probability * 100:.1f}%"
            )


            st.progress(
                min(max(probability, 0.0), 1.0)
            )


        # -------------------------------------------------
        # 8. Display model prediction
        # -------------------------------------------------

        if prediction is not None:

            st.caption(
                f"Model prediction: {prediction}"
            )


    except requests.exceptions.ConnectionError:

        st.warning(
            "⚠️ Could not connect to the FastAPI backend.\n\n"
            "Make sure main.py is running with:\n\n"
            "`python -m uvicorn main:app --reload --port 8000`"
        )


    except requests.exceptions.Timeout:

        st.warning(
            "⚠️ The backend took too long to respond. "
            "Please try again."
        )


    except requests.exceptions.HTTPError as e:

        st.error(
            f"❌ API returned an error: {e}"
        )


    except Exception as e:

        st.error(
            f"❌ Something went wrong: {e}"
        )


# ---------------------------------------------------------
# 9. Footer
# ---------------------------------------------------------

st.divider()

st.caption(
    "Model: Random Forest Classifier | "
    "Features: income, credit score, loan amount, "
    "employment years"
)