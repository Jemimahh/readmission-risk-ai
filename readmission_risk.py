"""
Hospital Readmission Risk Predictor
===================================
A simple AI-driven healthcare application that predicts the probability that a
patient will be readmitted to the hospital within 30 days of discharge.

Innovation studied: AI-driven predictive analytics for hospital readmission
and patient deterioration (see README.md for the research summary).

IMPORTANT: This project uses SYNTHETIC data only. It is an educational demo and
must NOT be used to make real clinical decisions.

Usage
-----
    python readmission_risk.py                 # train, evaluate, and run demo patients
    python readmission_risk.py --patients 8000 # use a larger synthetic dataset
    python readmission_risk.py --seed 7        # change the random seed

Pipeline
--------
    1. generate_synthetic_data()  -> create realistic-looking patient records
    2. build_models()             -> define two models (baseline + ML)
    3. evaluate()                 -> AUROC, precision/recall, calibration
    4. explain()                  -> which features drive risk (transparency)
    5. predict_patient()          -> score a new patient and flag risk tier
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------
# Numeric features come from typical EHR discharge data.
NUMERIC_FEATURES = [
    "age",
    "length_of_stay_days",
    "num_prior_admissions_12m",
    "num_medications",
    "num_chronic_conditions",
    "creatinine_mg_dl",
]
# Binary features (0/1) capture behavior and social factors. The research shows
# patient-reported data like medication adherence improves prediction.
BINARY_FEATURES = [
    "has_heart_failure",
    "lives_alone",
    "poor_medication_adherence",
    "follow_up_scheduled",
]
ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES
TARGET = "readmitted_30d"


# ---------------------------------------------------------------------------
# 1. Synthetic data
# ---------------------------------------------------------------------------
def generate_synthetic_data(n_patients: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic patient table with a plausible risk relationship.

    The true readmission probability is built from a logistic formula so that
    older age, longer stays, prior admissions, heart failure, poor adherence,
    etc. raise risk, while a scheduled follow-up lowers it. A model should be
    able to rediscover these patterns from the data.
    """
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(66, 15, n_patients), 18, 98)
    chronic = rng.poisson(2.2, n_patients)
    df = pd.DataFrame(
        {
            "age": age,
            "length_of_stay_days": np.clip(rng.gamma(2.0, 2.5, n_patients), 1, 30),
            "num_prior_admissions_12m": rng.poisson(0.9, n_patients),
            "num_medications": np.clip(rng.poisson(6 + chronic, n_patients), 0, 30),
            "num_chronic_conditions": chronic,
            "creatinine_mg_dl": np.clip(rng.normal(1.1, 0.4, n_patients), 0.4, 5.0),
            "has_heart_failure": rng.binomial(1, 0.22, n_patients),
            "lives_alone": rng.binomial(1, 0.28, n_patients),
            "poor_medication_adherence": rng.binomial(1, 0.30, n_patients),
            "follow_up_scheduled": rng.binomial(1, 0.65, n_patients),
        }
    )

    # Hidden "true" risk formula (log-odds). Coefficients are illustrative.
    log_odds = (
        -4.2
        + 0.025 * (df["age"] - 65)
        + 0.06 * df["length_of_stay_days"]
        + 0.45 * df["num_prior_admissions_12m"]
        + 0.04 * df["num_medications"]
        + 0.15 * df["num_chronic_conditions"]
        + 0.35 * (df["creatinine_mg_dl"] - 1.0)
        + 0.80 * df["has_heart_failure"]
        + 0.35 * df["lives_alone"]
        + 0.70 * df["poor_medication_adherence"]
        - 0.60 * df["follow_up_scheduled"]
        # A non-linear effect: very long stays AND many prior admissions compound.
        + 0.05 * df["length_of_stay_days"] * df["num_prior_admissions_12m"]
    )
    prob = 1 / (1 + np.exp(-log_odds))
    df[TARGET] = rng.binomial(1, prob)
    return df


# ---------------------------------------------------------------------------
# 2. Models
# ---------------------------------------------------------------------------
def build_models() -> dict:
    """Return the models to compare.

    - logistic: a transparent baseline (similar in spirit to scoring tools
      already used in hospitals).
    - gradient_boosting: a more flexible ML model that can capture non-linear
      patterns. It is wrapped in CalibratedClassifierCV so the predicted
      probabilities match real-world frequencies (calibration matters a lot in
      clinical settings; see the Epic Sepsis Model validation study).
    """
    preprocess = ColumnTransformer(
        [("num", StandardScaler(), NUMERIC_FEATURES)],
        remainder="passthrough",  # binary columns pass through unchanged
    )
    logistic = Pipeline(
        [
            ("prep", preprocess),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    gradient_boosting = CalibratedClassifierCV(
        GradientBoostingClassifier(random_state=0), method="isotonic", cv=3
    )
    return {"logistic": logistic, "gradient_boosting": gradient_boosting}


# ---------------------------------------------------------------------------
# 3. Evaluation
# ---------------------------------------------------------------------------
def evaluate(models: dict, X_test, y_test) -> None:
    """Print discrimination (AUROC), calibration (Brier), and a class report."""
    print("\n=== Model evaluation on held-out test set ===")
    for name, model in models.items():
        probs = model.predict_proba(X_test)[:, 1]
        auroc = roc_auc_score(y_test, probs)
        # Brier score: mean squared error of probabilities (lower is better).
        brier = brier_score_loss(y_test, probs)
        print(f"\n[{name}]  AUROC = {auroc:.3f}   Brier = {brier:.3f}")
        # Use a 0.25 alert threshold: readmission is relatively rare, so a
        # lower threshold than 0.5 catches more high-risk patients.
        preds = (probs >= 0.25).astype(int)
        print(classification_report(y_test, preds, digits=3, zero_division=0))
    # Alert burden: too many alerts causes "alert fatigue" in clinicians.
    best = models["gradient_boosting"].predict_proba(X_test)[:, 1]
    print(
        "Alert burden (gradient_boosting, threshold 0.25): "
        f"{(best >= 0.25).mean():.1%} of patients flagged"
    )


# ---------------------------------------------------------------------------
# 4. Explainability
# ---------------------------------------------------------------------------
def explain(model, X_test, y_test, top_n: int = 6) -> pd.Series:
    """Rank features by permutation importance (drop in AUROC when shuffled).

    Transparency helps clinicians trust and challenge a model, which is a
    common criticism of proprietary black-box tools.
    """
    result = permutation_importance(
        model, X_test, y_test, scoring="roc_auc", n_repeats=5, random_state=0
    )
    importance = pd.Series(result.importances_mean, index=X_test.columns)
    importance = importance.sort_values(ascending=False)
    print(f"\n=== Top {top_n} risk drivers (permutation importance) ===")
    for feature, value in importance.head(top_n).items():
        print(f"  {feature:<28} {value:.4f}")
    return importance


# ---------------------------------------------------------------------------
# 5. Scoring a single patient
# ---------------------------------------------------------------------------
def risk_tier(probability: float) -> str:
    """Convert a probability into a simple action tier."""
    if probability >= 0.35:
        return "HIGH   -> care-team call within 48h, med reconciliation"
    if probability >= 0.20:
        return "MEDIUM -> schedule follow-up visit, send reminders"
    return "LOW    -> standard discharge instructions"


def predict_patient(model, patient: dict) -> float:
    """Score one patient (a dict of feature values) and print the result."""
    row = pd.DataFrame([patient])[ALL_FEATURES]
    probability = float(model.predict_proba(row)[0, 1])
    print(f"  30-day readmission risk: {probability:.1%}  |  {risk_tier(probability)}")
    return probability


def main() -> None:
    parser = argparse.ArgumentParser(description="Readmission risk demo (synthetic data)")
    parser.add_argument("--patients", type=int, default=5000, help="synthetic patients")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    # Step 1: data
    df = generate_synthetic_data(args.patients, args.seed)
    print(f"Generated {len(df)} synthetic patients; "
          f"readmission rate = {df[TARGET].mean():.1%}")

    # Split into train/test; stratify keeps the readmission rate equal in both.
    X_train, X_test, y_train, y_test = train_test_split(
        df[ALL_FEATURES], df[TARGET], test_size=0.25,
        random_state=args.seed, stratify=df[TARGET],
    )

    # Step 2: train every model
    models = build_models()
    for model in models.values():
        model.fit(X_train, y_train)

    # Steps 3 and 4: evaluate and explain
    evaluate(models, X_test, y_test)
    explain(models["gradient_boosting"], X_test, y_test)

    # Step 5: demo patients
    print("\n=== Demo patients ===")
    demo = {
        "Patient A (older, heart failure, poor adherence, lives alone)": {
            "age": 79, "length_of_stay_days": 9, "num_prior_admissions_12m": 3,
            "num_medications": 14, "num_chronic_conditions": 5,
            "creatinine_mg_dl": 1.8, "has_heart_failure": 1, "lives_alone": 1,
            "poor_medication_adherence": 1, "follow_up_scheduled": 0,
        },
        "Patient B (middle-aged, healthy, follow-up booked)": {
            "age": 45, "length_of_stay_days": 2, "num_prior_admissions_12m": 0,
            "num_medications": 3, "num_chronic_conditions": 1,
            "creatinine_mg_dl": 0.9, "has_heart_failure": 0, "lives_alone": 0,
            "poor_medication_adherence": 0, "follow_up_scheduled": 1,
        },
    }
    for label, patient in demo.items():
        print(label)
        predict_patient(models["gradient_boosting"], patient)

    print("\nReminder: synthetic data, educational use only. Not for clinical use.")


if __name__ == "__main__":
    main()
