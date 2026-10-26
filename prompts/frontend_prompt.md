# GenAI prompt used to generate the front end

Tool: Claude (the same prompt works in Gemini / ChatGPT).

```
I have developed two machine-learning models and I need a Streamlit front end that shows BOTH predictions
side by side so the user can compare them.

MODELS
- models/rf_model.pkl (Random Forest) and models/xgb_model.pkl (XGBoost): scikit-learn Pipelines saved with pickle.
  Their first step is FunctionTransformer(dengue_sl.add_derived), so import dengue_sl before unpickling.
- Binary classifiers; predict_proba(df)[:,1] = probability that a Sri Lankan district has a HIGH-dengue week
  2 weeks from now (incidence above that district's own 75th percentile of the training period).
- models/model_meta.json holds: per-district table (population, latitude, longitude, coastal, elevation_code,
  baseline_p75, province), training range of each field, four real example weeks, test metrics for both models
  and their paired comparison (label agreement, AUC difference with CI, McNemar p).

MODEL INPUT COLUMNS (one-row DataFrame, this order)
epi_week, cases_t, cases_t1, cases_t2, cases_t3, rain_t, rain_4wk, rain_8wk, tmax_4wk, tmin_4wk, humidity_4wk,
population, latitude, longitude, coastal, elevation_code, baseline_p75
The user types the first 11 (+ population); the last 5 are looked up from the chosen district.

REQUIREMENTS
1. District selectbox that auto-fills population; a number_input for every dynamic field (with units).
2. Sidebar buttons that load the four real example weeks (use session_state + callbacks).
3. "Predict with both models" button -> show Random Forest probability, XGBoost probability and their difference
   as metrics, a banner (both HIGH / both normal / models DISAGREE), and a bar chart comparing the two probabilities.
4. "Why?" section: top-8 feature contributions for each model side by side (SHAP for RF, native pred_contribs for XGBoost).
5. Second tab "Test-set comparison": metrics table for both models, agreement / AUC-difference / McNemar metrics,
   and the PNG figures in figures/.
6. Validate epi_week (1-53), warn when a value is outside the training range, no ML jargon in the main tab.
7. Footer: coursework decision-support prototype.
Return the full code.
```

## Follow-up prompts used
- "Use width='stretch' instead of use_container_width."
- "Add a disagreement example week so the comparison is visible in a demo."
