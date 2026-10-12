"""
Streamlit demo for the Hospital Readmission Risk Predictor
==========================================================
An interactive front end for readmission_risk.py. Enter a (fictional) patient in
the sidebar and see the predicted 30-day readmission risk, the action tier, and
which factors push this patient's risk up or down.

Run:
    pip install -r requirements.txt
    streamlit run streamlit_app.py

SYNTHETIC DATA ONLY. Educational demo, not for clinical use.
"""

import pandas as pd
import streamlit as st
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

# Reuse the data generator, models, and helpers from the core module.
from readmission_risk import (
    ALL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_models,
    explain,
    generate_synthetic_data,
    risk_tier,
)

st.set_page_config(page_title="Readmission Risk Predictor", page_icon="🏥", layout="wide")


# ---------------------------------------------------------------------------
# Training (cached so the app only trains once per session, not on every click)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training models on synthetic patients...")
def load_everything(n_patients: int = 5000, seed: int = 42):
    """Generate data, train both models, and compute test-set metrics."""
    df = generate_synthetic_data(n_patients, seed)
    X_train, X_test, y_train, y_test = train_test_split(
        df[ALL_FEATURES], df[TARGET], test_size=0.25,
        random_state=seed, stratify=df[TARGET],
    )
    models = build_models()
    for model in models.values():
        model.fit(X_train, y_train)

    metrics = []
    for name, model in models.items():
        probs = model.predict_proba(X_test)[:, 1]
        metrics.append(
            {
                "model": name,
                "AUROC": round(roc_auc_score(y_test, probs), 3),
                "Brier (lower is better)": round(brier_score_loss(y_test, probs), 3),
                "% flagged at 0.25": f"{(probs >= 0.25).mean():.1%}",
            }
        )

    # Global importance: computed once, silently (explain() prints to console).
    importance = explain(models["gradient_boosting"], X_test, y_test)
    return df, models, pd.DataFrame(metrics), importance


def patient_drivers(model, patient: dict, reference: pd.DataFrame) -> pd.Series:
    """Per-patient explanation using simple counterfactuals.

    For each feature, replace this patient's value with the typical cohort value
    (the median) and measure how the predicted risk changes. A positive number
    means that feature is pushing THIS patient's risk up compared with a typical
    patient. This is easy to explain to a clinician, unlike a black-box score.
    """
    base = float(model.predict_proba(pd.DataFrame([patient])[ALL_FEATURES])[0, 1])
    effects = {}
    for feature in ALL_FEATURES:
        altered = dict(patient)
        altered[feature] = reference[feature].median()
        new = float(model.predict_proba(pd.DataFrame([altered])[ALL_FEATURES])[0, 1])
        effects[feature] = base - new  # how much the feature adds to risk
    return pd.Series(effects).sort_values()


df, models, metrics, importance = load_everything()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🏥 Hospital Readmission Risk Predictor")
st.caption(
    "AI-driven predictive analytics demo using **synthetic data only**. "
    "Not for clinical use."
)

# ---------------------------------------------------------------------------
# Sidebar: patient input form
# ---------------------------------------------------------------------------
st.sidebar.header("Patient at discharge")
patient = {
    "age": st.sidebar.slider("Age", 18, 98, 72),
    "length_of_stay_days": st.sidebar.slider("Length of stay (days)", 1, 30, 6),
    "num_prior_admissions_12m": st.sidebar.slider("Admissions in past 12 months", 0, 8, 2),
    "num_medications": st.sidebar.slider("Number of medications", 0, 30, 10),
    "num_chronic_conditions": st.sidebar.slider("Chronic conditions", 0, 10, 3),
    "creatinine_mg_dl": st.sidebar.slider("Creatinine (mg/dL)", 0.4, 5.0, 1.3, 0.1),
    "has_heart_failure": int(st.sidebar.checkbox("Heart failure", value=True)),
    "lives_alone": int(st.sidebar.checkbox("Lives alone", value=False)),
    "poor_medication_adherence": int(st.sidebar.checkbox("Poor medication adherence", value=True)),
    "follow_up_scheduled": int(st.sidebar.checkbox("Follow-up visit scheduled", value=False)),
}

model_choice = st.sidebar.radio(
    "Model",
    ["gradient_boosting", "logistic"],
    help="Gradient boosting is calibrated; the logistic baseline over-predicts "
    "risk because it uses class balancing.",
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_predict, tab_models, tab_about = st.tabs(["Predict", "Model performance", "About"])

with tab_predict:
    model = models[model_choice]
    risk = float(model.predict_proba(pd.DataFrame([patient])[ALL_FEATURES])[0, 1])
    cohort_rate = float(df[TARGET].mean())

    col1, col2, col3 = st.columns(3)
    col1.metric("30-day readmission risk", f"{risk:.1%}")
    col2.metric("Cohort average", f"{cohort_rate:.1%}", f"{(risk - cohort_rate):+.1%} vs average", delta_color="inverse")
    col3.metric("Action tier", risk_tier(risk).split("->")[0].strip())

    st.progress(min(risk, 1.0))
    st.info(risk_tier(risk).split("->")[1].strip().capitalize())

    st.subheader("What is driving this patient's risk?")
    st.caption(
        "Change in predicted risk when each factor is swapped for the typical "
        "(median) patient value. Positive = pushes this patient's risk up."
    )
    effects = patient_drivers(model, patient, df[ALL_FEATURES])
    effects = effects[effects.abs() > 0.001]  # hide features with no effect
    st.bar_chart(effects * 100, horizontal=True)
    st.caption("Values are percentage points of risk.")

    st.subheader("What if? Try a quick intervention")
    what_if = dict(patient)
    what_if["follow_up_scheduled"] = 1
    what_if["poor_medication_adherence"] = 0
    risk_after = float(model.predict_proba(pd.DataFrame([what_if])[ALL_FEATURES])[0, 1])
    st.write(
        f"Scheduling a follow-up visit and improving medication adherence would change "
        f"risk from **{risk:.1%}** to **{risk_after:.1%}**. "
        "(Illustrative only: the effect comes from the synthetic data's built-in assumptions, "
        "not from real causal evidence.)"
    )

with tab_models:
    st.subheader("Held-out test performance")
    st.dataframe(metrics, hide_index=True, width="stretch")
    st.markdown(
        "- **AUROC** measures how well the model ranks high-risk above low-risk patients.\n"
        "- **Brier score** measures whether the probabilities are trustworthy. "
        "A model can rank well yet still be badly calibrated.\n"
        "- **% flagged** is the alert burden. Too many alerts cause alert fatigue, "
        "a documented problem with the Epic Sepsis Model."
    )
    st.subheader("Overall risk drivers (permutation importance)")
    st.bar_chart(importance.head(8), horizontal=True)

with tab_about:
    st.markdown(
        """
**Innovation studied:** AI predictive analytics for hospital readmission and patient deterioration.

**Why these design choices?**
- Patient-reported behavior (adherence, living situation) improves readmission prediction in recent cardiac research.
- Independent validation of a widely deployed sepsis model found weak discrimination and heavy alert burden,
  so this app reports calibration and alert burden, and explains each prediction.

**Limitations:** synthetic data, no external validation, no fairness audit.
See the README for sources.
"""
    )
