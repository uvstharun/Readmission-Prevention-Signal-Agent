"""
Streamlit dashboard for the Readmission Prevention Signal Agent.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Readmission Prevention Agent",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

# Risk tier colors
TIER_COLORS = {"HIGH": "#dc3545", "MODERATE": "#fd7e14", "LOW": "#28a745"}
TIER_BADGES = {"HIGH": "🔴 HIGH", "MODERATE": "🟡 MODERATE", "LOW": "🟢 LOW"}

# Custom CSS
st.markdown("""
<style>
.risk-high { background:#fff0f0; border-left:4px solid #dc3545; padding:12px; border-radius:4px; }
.risk-moderate { background:#fff8f0; border-left:4px solid #fd7e14; padding:12px; border-radius:4px; }
.risk-low { background:#f0fff4; border-left:4px solid #28a745; padding:12px; border-radius:4px; }
.metric-card { background:#f8f9fa; padding:16px; border-radius:8px; text-align:center; }
.stAlert { border-radius:8px; }
</style>
""", unsafe_allow_html=True)


def api_call(method: str, endpoint: str, data: dict = None) -> dict:
    try:
        url = f"{API_BASE}{endpoint}"
        if method == "GET":
            response = requests.get(url, timeout=10)
        else:
            response = requests.post(url, json=data, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "API server not running. Start with: uvicorn src.api.main:app --reload"}
    except Exception as e:
        return {"error": str(e)}


def run_mock_pipeline(patient_data: dict) -> dict:
    """Run mock pipeline when API is unavailable."""
    import random
    score = 0.12
    score += min(patient_data.get("charlson_comorbidity_index", 0) * 0.05, 0.25)
    score += min(patient_data.get("prior_admissions_6mo", 0) * 0.08, 0.24)
    score += (1 - patient_data.get("followup_appointment_scheduled", 1)) * 0.10
    score += (1 - patient_data.get("housing_stability_flag", 1)) * 0.08
    score = min(0.95, max(0.05, score + random.uniform(-0.02, 0.02)))
    tier = "HIGH" if score >= 0.65 else ("MODERATE" if score >= 0.35 else "LOW")
    return {
        "risk_result": {"risk_score": round(score, 4), "risk_tier": tier,
                        "top_risk_drivers": [{"feature": "charlson_comorbidity_index", "shap_contribution": 0.15},
                                              {"feature": "prior_admissions_6mo", "shap_contribution": 0.12}]},
        "clinical_narrative": {
            "risk_summary": f"Patient has {tier} readmission risk with score of {score:.1%}",
            "top_risk_drivers_explained": [{"driver": "Comorbidity burden", "explanation": "High CCI increases complexity"}],
            "case_manager_talking_points": ["Review medications", "Schedule follow-up", "Assess social needs"],
            "priority_actions": ["Schedule follow-up within 7 days", "Complete medication reconciliation"],
        },
        "care_gap_result": {"total_gaps": 2, "care_gaps": [
            {"gap": "No Follow-Up Scheduled", "priority": "HIGH"},
            {"gap": "Polypharmacy", "priority": "MEDIUM"},
        ]},
        "workflow_result": {"interventions_triggered": 3},
    }


def home_page():
    st.title("🏥 Readmission Prevention Signal Agent")
    st.caption("AI-powered 30-day readmission risk scoring and care transition management")

    result = api_call("GET", "/dashboard/summary")

    if "error" in result:
        st.warning(f"⚠️ {result['error']}")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Patients Scored Today", "—")
        with col2:
            st.metric("High Risk Patients", "—")
        with col3:
            st.metric("Interventions Triggered", "—")
        with col4:
            st.metric("Active Watchlist", "—")
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Patients Scored Today", result.get("total_patients_scored_today", 0))
        with col2:
            st.metric("High Risk Patients", result.get("high_risk_count_today", 0),
                      delta="↑ requires attention" if result.get("high_risk_count_today", 0) > 0 else None)
        with col3:
            st.metric("Interventions Triggered", result.get("interventions_triggered_today", 0))
        with col4:
            st.metric("Active Watchlist", result.get("active_watchlist_count", 0))

    st.divider()

    # Sample risk distribution
    st.subheader("Risk Distribution Overview")
    col1, col2 = st.columns([1, 2])
    with col1:
        risk_data = pd.DataFrame({"Risk Tier": ["HIGH", "MODERATE", "LOW"], "Count": [23, 45, 82]})
        fig = px.pie(risk_data, values="Count", names="Risk Tier",
                     color="Risk Tier",
                     color_discrete_map={"HIGH": "#dc3545", "MODERATE": "#fd7e14", "LOW": "#28a745"},
                     title="Today's Risk Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        trend_data = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=30, freq="D"),
            "Readmission Rate": np.random.uniform(0.10, 0.18, 30),
        })
        fig2 = px.line(trend_data, x="Date", y="Readmission Rate", title="30-Day Readmission Rate Trend")
        fig2.update_layout(yaxis_tickformat=".1%")
        st.plotly_chart(fig2, use_container_width=True)


def patient_scorer_page():
    st.title("Patient Risk Scorer")
    st.caption("Enter a new discharge patient to trigger the AI agent pipeline")

    with st.form("patient_form"):
        st.subheader("Demographics")
        col1, col2, col3 = st.columns(3)
        with col1:
            patient_id = st.text_input("Patient ID", value="PT999001")
            age = st.number_input("Age", min_value=18, max_value=100, value=72)
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female", "Non-binary"])
            race = st.selectbox("Race", ["White", "Black or African American", "Asian", "Hispanic or Latino", "Other"])
        with col3:
            insurance = st.selectbox("Insurance", ["Medicare", "Medicaid", "Commercial", "Medicare Advantage", "Uninsured"])
            language = st.selectbox("Language", ["English", "Spanish", "Mandarin", "Vietnamese", "Other"])

        st.subheader("Encounter Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            admission_date = st.date_input("Admission Date")
            discharge_date = st.date_input("Discharge Date")
        with col2:
            los = st.number_input("Length of Stay (days)", min_value=1, max_value=60, value=5)
            admit_type = st.selectbox("Admission Type", ["Emergency", "Urgent", "Elective"])
        with col3:
            discharge_disp = st.selectbox("Discharge Disposition",
                                          ["Home", "Home with Home Health", "SNF", "Rehab Facility", "AMA"])
            department = st.selectbox("Department",
                                      ["Medicine", "Cardiology", "Pulmonology", "Nephrology", "Oncology"])

        st.subheader("Clinical Data")
        col1, col2, col3 = st.columns(3)
        with col1:
            primary_dx = st.text_input("Primary ICD-10 Code", value="I50.20")
            cci = st.slider("Charlson Comorbidity Index", 0, 12, 3)
        with col2:
            prior_admissions = st.number_input("Prior Admissions (6 mo)", 0, 20, 1)
            prior_ed = st.number_input("Prior ED Visits (6 mo)", 0, 20, 2)
        with col3:
            n_meds = st.number_input("Active Medications", 0, 30, 8)
            high_risk_med = st.checkbox("High-Risk Medications (anticoagulants/insulin/opioids)")

        st.subheader("Social Determinants & Care Transition")
        col1, col2, col3 = st.columns(3)
        with col1:
            housing = st.checkbox("Housing Stable", value=True)
            transport = st.checkbox("Transportation Available", value=True)
        with col2:
            social_support = st.slider("Social Support Score", 1, 7, 4)
            followup = st.checkbox("Follow-Up Appointment Scheduled")
        with col3:
            pcp = st.checkbox("PCP Assigned", value=True)
            dc_instructions = st.checkbox("Discharge Instructions Given", value=True)

        use_llm = st.toggle("Generate LLM Clinical Narrative", value=True)
        submitted = st.form_submit_button("Run Agent Pipeline", type="primary", use_container_width=True)

    if submitted:
        patient_data = {
            "patient_id": patient_id,
            "age": age,
            "gender": gender,
            "race": race,
            "ethnicity": "Non-Hispanic",
            "preferred_language": language,
            "zip_code": "10001",
            "admission_date": str(admission_date),
            "discharge_date": str(discharge_date),
            "length_of_stay_days": los,
            "admission_type": admit_type,
            "discharge_disposition": discharge_disp,
            "attending_department": department,
            "primary_diagnosis_code": primary_dx,
            "secondary_diagnosis_codes": "",
            "charlson_comorbidity_index": cci,
            "prior_admissions_6mo": prior_admissions,
            "prior_ed_visits_6mo": prior_ed,
            "prior_readmissions_1yr": 0,
            "num_active_medications": n_meds,
            "high_risk_medication_flag": int(high_risk_med),
            "insurance_type": insurance,
            "housing_stability_flag": int(housing),
            "transportation_access_flag": int(transport),
            "social_support_score": social_support,
            "followup_appointment_scheduled": int(followup),
            "pcp_assigned_flag": int(pcp),
            "discharge_instructions_given": int(dc_instructions),
            "use_llm": use_llm,
        }

        with st.spinner("Running agent pipeline..."):
            result = api_call("POST", "/discharge", patient_data)
            if "error" in result:
                st.info("Running in offline mode (API not connected)...")
                result = run_mock_pipeline(patient_data)

        # Display results
        risk_result = result.get("risk_result", {})
        tier = risk_result.get("risk_tier", "UNKNOWN")
        score = risk_result.get("risk_score", 0)

        st.divider()
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            tier_color = TIER_COLORS.get(tier, "#6c757d")
            st.markdown(f"""
            <div style='background:{tier_color}22; border:2px solid {tier_color}; border-radius:8px; padding:20px; text-align:center'>
            <h1 style='color:{tier_color}; margin:0'>{score:.1%}</h1>
            <h3 style='color:{tier_color}; margin:4px 0'>Risk Score</h3>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style='background:{tier_color}22; border:2px solid {tier_color}; border-radius:8px; padding:20px; text-align:center'>
            <h1 style='color:{tier_color}; margin:0'>{TIER_BADGES.get(tier, tier)}</h1>
            <h3 style='color:{tier_color}; margin:4px 0'>Risk Tier</h3>
            </div>""", unsafe_allow_html=True)
        with col3:
            narrative = result.get("clinical_narrative", {})
            st.info(f"**Clinical Summary:** {narrative.get('risk_summary', 'N/A')}")

        # Clinical narrative
        with st.expander("Clinical Narrative (LLM-Generated)", expanded=True):
            narrative = result.get("clinical_narrative", {})
            if "top_risk_drivers_explained" in narrative:
                st.markdown("**Top Risk Drivers:**")
                for driver in narrative.get("top_risk_drivers_explained", []):
                    st.markdown(f"- **{driver.get('driver', '')}**: {driver.get('explanation', '')}")
            if "case_manager_talking_points" in narrative:
                st.markdown("**Case Manager Talking Points:**")
                for point in narrative.get("case_manager_talking_points", []):
                    st.markdown(f"- {point}")
            if "priority_actions" in narrative:
                st.markdown("**Priority Actions:**")
                for action in narrative.get("priority_actions", []):
                    st.markdown(f"✅ {action}")

        # Care gaps
        care_gap_result = result.get("care_gap_result", {})
        gaps = care_gap_result.get("care_gaps", [])
        if gaps:
            with st.expander(f"Care Gaps Identified ({len(gaps)})", expanded=True):
                for gap in gaps:
                    priority = gap.get("priority", "LOW")
                    color = {"HIGH": "#dc3545", "MEDIUM": "#fd7e14", "LOW": "#28a745"}.get(priority, "#6c757d")
                    st.markdown(f"""
                    <div style='border-left:3px solid {color}; padding:8px; margin:4px 0; background:{color}11'>
                    <b style='color:{color}'>[{priority}]</b> {gap.get('gap', '')}<br>
                    <small>{gap.get('rationale', '')}</small>
                    </div>""", unsafe_allow_html=True)


def watchlist_page():
    st.title("30-Day Monitoring Watchlist")

    result = api_call("GET", "/watchlist")

    if "error" in result:
        st.warning(result["error"])
        return

    patients = result.get("patients", [])
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Active", result.get("total_active", 0))
    with col2:
        st.metric("High Risk", result.get("high_risk", 0))
    with col3:
        st.metric("Moderate Risk", result.get("moderate_risk", 0))
    with col4:
        st.metric("Escalated", result.get("escalated", 0))

    if patients:
        df = pd.DataFrame(patients)
        display_cols = [c for c in ["patient_id", "risk_tier", "risk_score", "discharge_date",
                                     "days_remaining", "escalation_status"] if c in df.columns]
        st.dataframe(df[display_cols], use_container_width=True)
    else:
        st.info("No patients currently on the watchlist.")

    if st.button("Run Monitoring Cycle"):
        with st.spinner("Running monitoring cycle..."):
            cycle_result = api_call("POST", "/monitoring/run-cycle")
            if "error" not in cycle_result:
                st.success(f"Monitoring cycle complete: {cycle_result.get('patients_monitored', 0)} patients checked, "
                           f"{cycle_result.get('patients_escalated', 0)} escalated")


def model_insights_page():
    st.title("Model Insights & Performance")

    # Try to load evaluation report
    report_path = "models/evaluation_report.json"
    if os.path.exists(report_path):
        with open(report_path) as f:
            report = json.load(f)

        col1, col2, col3 = st.columns(3)
        best_metrics = report.get("best_model_metrics", {})
        with col1:
            st.metric("AUC-ROC", f"{best_metrics.get('test_auc_roc', 0):.4f}")
        with col2:
            st.metric("Average Precision", f"{best_metrics.get('test_avg_precision', 0):.4f}")
        with col3:
            st.metric("Brier Score", f"{best_metrics.get('test_brier_score', 0):.4f}")

        st.info(f"Best Model: **{report.get('best_model', 'N/A')}** | "
                f"Features: {report.get('n_features', 'N/A')} | "
                f"Training samples: {report.get('n_train', 'N/A')}")

        # Model comparison
        all_models = report.get("all_models", {})
        if all_models:
            model_df = pd.DataFrame([
                {"Model": k, "AUC-ROC": v.get("test_auc_roc", 0), "Avg Precision": v.get("test_avg_precision", 0)}
                for k, v in all_models.items()
            ])
            fig = px.bar(model_df, x="Model", y="AUC-ROC", title="Model Comparison - AUC-ROC",
                         color="AUC-ROC", color_continuous_scale="Blues")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Model evaluation report not found. Train the model first with: `python -m src.models.model_trainer`")

    # Display saved plots if available
    for plot_name, plot_path in [("ROC Curve", "models/roc_curve.png"), ("Calibration Curve", "models/calibration_curve.png")]:
        if os.path.exists(plot_path):
            col1, col2 = st.columns(2)
            with col1:
                st.image(plot_path, caption=plot_name)


def main():
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/hospital.png", width=60)
        st.title("Navigation")
        page = st.radio("", ["Home", "Patient Scorer", "Watchlist", "Model Insights"], label_visibility="collapsed")
        st.divider()
        st.caption("Readmission Prevention Signal Agent v1.0")
        st.caption("Built with Claude AI + XGBoost")

    if page == "Home":
        home_page()
    elif page == "Patient Scorer":
        patient_scorer_page()
    elif page == "Watchlist":
        watchlist_page()
    elif page == "Model Insights":
        model_insights_page()


if __name__ == "__main__":
    main()
