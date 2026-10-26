# Sri Lanka Dengue Early-Warning: Random Forest vs XGBoost (district level)

Predicts, for any of the **25 districts**, whether the dengue incidence **two weeks from now** will be a *high week for that district* (above its own training-period 75th percentile). The web app shows the **Random Forest and XGBoost predictions side by side** with an agreement banner, per-model explanations and the full held-out comparison.

## Data
`data/dengue_srilanka_full_2006_2025.csv` - weekly district cases, population, geography and climate, 25 districts x 991 weeks (Dec 2006 - Dec 2025). **Add the original source/citation of this file here** (the file does not state it).
Excluded on purpose (leakage / redundant): `is_epidemic_year` (set from a whole year's outcome), `cases_zscore` (global statistics), `incidence_rate`, `population_source`.
65 implausible zero-case weeks in busy districts (mostly 2012, 2016, 2022) were treated as reporting gaps and interpolated.

## Results (chronological test, 2021-2025; CV = 5-fold expanding window)
| | Random Forest | XGBoost |
|---|---|---|
| CV AUC | 0.863 | 0.866 |
| Test AUC (95% block-bootstrap CI) | 0.876 (0.852-0.899) | 0.875 (0.850-0.898) |
| Test PR-AUC | 0.827 | 0.826 |
| Recall / precision at cut-off 0.5 | 0.85 / 0.68 | 0.68 / 0.78 |
| Brier score | 0.151 | 0.140 |

Paired AUC difference RF - XGBoost = +0.0007 (95% CI -0.0023 to +0.0043); precision at 80% recall 0.716 vs 0.714. **They rank risk equally well; they differ in probability scale** (RF more alarmist, XGBoost more conservative and better calibrated). Baselines: logistic regression 0.872 test AUC, persistence rule 0.850, soft voting 0.877: the ensembles add little over simple approaches. Climate features add nothing measurable. Details and figures: `notebooks/dengue_rf_vs_xgboost.ipynb`.

## Layout
```
data/                  CSV
dengue_sl.py           cleaning, features, shared by notebook + pickles + app
notebooks/             executed notebook (+ .html copy)
models/                rf_model.pkl, xgb_model.pkl, model_meta.json
streamlit_app.py       Streamlit UI: both predictions + comparison
figures/               all plots and result CSVs
prompts/               GenAI prompt used for the UI
requirements.txt       app requirements (Streamlit Cloud reads this)
requirements-notebook.txt
```

## Run
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py                       # http://localhost:8501
pip install -r requirements-notebook.txt && jupyter notebook notebooks/dengue_rf_vs_xgboost.ipynb   # ~10 min on 1 CPU
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a public GitHub repo (the `.pkl` files total ~9 MB, which is fine).
2. share.streamlit.io -> Create app -> select the repo, branch `main`, main file `streamlit_app.py`.
3. Advanced settings -> **Python 3.12** (the pickles were built with scikit-learn 1.8 / xgboost 3.4, pinned in `requirements.txt`).
4. Deploy, then paste the link here: *(live demo URL)*

## Limitations
One chronological split; districts are correlated in space and time; notified cases are not true incidence; 2007-2008 look under-reported; part of the population series is modelled; coursework prototype, not validated for public-health use.
