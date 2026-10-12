# Hospital Readmission Risk Predictor

A simple AI-driven healthcare application that estimates a patient's probability of 30-day hospital readmission, flags a risk tier, and explains which factors drive risk.

> **Educational demo.** It uses synthetic data only and must not be used for clinical decisions.

## Chosen innovation: AI predictive analytics for readmission and patient deterioration

Hospitals increasingly use machine learning on EHR data to flag patients likely to deteriorate or return after discharge, so care teams can intervene early (medication reconciliation, follow-up calls, scheduled visits).

### Research summary

- **Patient-reported data improves prediction (2026).** A University at Buffalo study combined patient-reported behavioral survey data with electronic health records to predict hospitalizations and 90-day readmissions in cardiac patients. It highlights medication adherence and socioeconomic factors as important signals. My app mirrors this idea with features such as `poor_medication_adherence` and `lives_alone`.
- **Deep learning beats traditional scores in the ICU (2026).** A retrospective study using MIMIC-IV and an external hospital dataset found that a GRU-D++ model outperformed the SWIFT score for predicting ICU readmission or death within seven days (AUROC 0.802 internal, 0.756 external). Performance dropped on external data, which is a reminder that models must be validated outside the site where they were trained.
- **Cautionary case: the Epic Sepsis Model.** A JAMA Internal Medicine external validation (Wong et al.) of 27,697 patients found an AUROC of 0.63, far below the developer's reported performance. The model missed 67% of sepsis cases while alerting on 18% of all hospitalized patients, creating alert fatigue. This motivated the design choices below: calibrated probabilities, an alert-burden metric, transparent feature importance, and a simple baseline for comparison.

### How the research shaped the app

| Lesson from research | Design choice in code |
|---|---|
| Behavioral and social data add predictive value | Adherence, living alone, follow-up scheduled as features |
| Proprietary and opaque models draw criticism | Permutation-importance explanation of risk drivers |
| Poor calibration undermines trust | Isotonic-calibrated gradient boosting, Brier score reported |
| Alert fatigue is a real harm | Alert burden (% flagged) printed at the chosen threshold |
| Models need external validation | Documented limitation: synthetic data, no external validation |

## How it works

1. Generate synthetic patients with a hidden, plausible risk formula
2. Train a logistic regression baseline and a calibrated gradient boosting model
3. Evaluate with AUROC, Brier score, precision/recall, and alert burden
4. Explain top risk drivers
5. Score example patients and assign a LOW / MEDIUM / HIGH action tier

## Run it

```bash
pip install -r requirements.txt
python readmission_risk.py
# optional: python readmission_risk.py --patients 8000 --seed 7
```

### Interactive demo (Streamlit)

```bash
streamlit run streamlit_app.py
```
[Steamlit demo video](./media/readmission-risk-steamlitDemo.mp4)

Enter a fictional patient in the sidebar to see the predicted risk, the action tier, a per-patient breakdown of what raises or lowers risk, and a what-if scenario (follow-up scheduled plus better adherence). A second tab compares the two models on AUROC, Brier score, and alert burden. Switching to the logistic baseline shows how a model can rank patients reasonably yet give inflated probabilities.

Sample results (seed 42): gradient boosting AUROC about 0.80 with a Brier score of 0.082. The top driver is prior admissions, followed by length of stay.

## Limitations

- Synthetic data, so results do not reflect real patients.
- The logistic baseline uses `class_weight="balanced"`, which inflates its probabilities. That is why its Brier score (0.183) is worse than the calibrated model's even though its AUROC is similar. This is a useful illustration of why discrimination alone is not enough.
- No bias or fairness audit, and no external validation.

## Sources

- University at Buffalo, "UB pharmacy professor develops AI model to predict hospitalization of at-risk cardiac patients" (Jan 2026): https://www.buffalo.edu/news/releases/2026/01/Okere-hospitalization-study.html
- "AI-Driven Predictions of Readmission and Mortality for Improved Discharge Decisions in Critical Care: A Retrospective Study" (Mar 2026): https://pure.seoultech.ac.kr/en/publications/ai-driven-predictions-of-readmission-and-mortality-for-improved-d/
- Wong et al., "External Validation of a Widely Implemented Proprietary Sepsis Prediction Model in Hospitalized Patients," JAMA Internal Medicine: https://pmc.ncbi.nlm.nih.gov/articles/PMC8218233
- External validation of the Epic sepsis model in two county emergency departments: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11560849/
- "Predictive models for ICU patient readmission based on machine learning: A systematic review" (2026): https://journals.sagepub.com/doi/full/10.1177/17511437261431540

## AI assistance disclosure

Per the assignment note, this project was created with help from an LLM (Claude) and reviewed by the author.
