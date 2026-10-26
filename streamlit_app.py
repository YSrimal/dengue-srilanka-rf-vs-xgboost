"""
Sri Lanka Dengue Early Warning - Random Forest vs XGBoost (district level).
Run locally:   streamlit run streamlit_app.py
Deploy:        Streamlit Community Cloud -> main file = streamlit_app.py  (Python 3.12)
"""
import json, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
import dengue_sl as dsl                      # also makes the pickles importable
from dengue_sl import RAW_FIELDS, DYNAMIC

RF_C, XG_C = "#2a6db3", "#b3402a"
st.set_page_config(page_title="Dengue Early Warning: RF vs XGBoost", page_icon="🦟", layout="wide")


@st.cache_resource
def load():
    with open(ROOT / "models" / "rf_model.pkl", "rb") as f:
        rf = pickle.load(f)
    with open(ROOT / "models" / "xgb_model.pkl", "rb") as f:
        xgb = pickle.load(f)
    return rf, xgb, json.load(open(ROOT / "models" / "model_meta.json"))


RF, XGB, META = load()
DIST, RANGES, H = META["districts"], META["field_ranges"], META["horizon_weeks"]
LABELS = {"epi_week": ("Epidemiological week", "1-53"), "cases_t": ("Cases this week", "cases"), "cases_t1": ("Cases 1 week ago", "cases"),
          "cases_t2": ("Cases 2 weeks ago", "cases"), "cases_t3": ("Cases 3 weeks ago", "cases"), "rain_t": ("Rainfall this week", "mm"),
          "rain_4wk": ("Rainfall, 4-week mean", "mm/week"), "rain_8wk": ("Rainfall, 8-week mean", "mm/week"),
          "tmax_4wk": ("Max temperature, 4-week mean", "°C"), "tmin_4wk": ("Min temperature, 4-week mean", "°C"),
          "humidity_4wk": ("Humidity, 4-week mean", "%")}


# ---------- state helpers ----------
def on_district():
    st.session_state["population"] = float(DIST[st.session_state["district"]]["population"])


def load_example(i):
    ex = META["examples"][i]
    st.session_state["district"] = ex["district"]
    for k, v in ex["inputs"].items():
        st.session_state[k] = float(v)


st.session_state.setdefault("district", "Colombo")
st.session_state.setdefault("population", float(DIST["Colombo"]["population"]))
for k in DYNAMIC:
    st.session_state.setdefault(k, float(RANGES[k]["median"]))


def predict(values, district):
    d = DIST[district]
    row = {**values, "latitude": d["latitude"], "longitude": d["longitude"], "coastal": d["coastal"],
           "elevation_code": d["elevation_code"], "baseline_p75": d["baseline_p75"]}
    X = pd.DataFrame([row])[RAW_FIELDS].astype(float)
    return X, float(RF.predict_proba(X)[0, 1]), float(XGB.predict_proba(X)[0, 1])


def contributions(X):
    """Top feature contributions for each model (XGBoost native; Random Forest via SHAP if available)."""
    out = {}
    try:
        import xgboost as xgb
        Xf = XGB.named_steps["features"].transform(X)
        c = XGB.named_steps["model"].get_booster().predict(xgb.DMatrix(Xf, feature_names=list(Xf.columns)), pred_contribs=True)[0][:-1]
        out["XGBoost"] = pd.Series(c, index=Xf.columns)
    except Exception:
        pass
    try:
        import shap
        Xf = RF.named_steps["features"].transform(X)
        sv = shap.TreeExplainer(RF.named_steps["model"])(Xf, check_additivity=False)
        v = sv.values[0, :, 1] if sv.values.ndim == 3 else sv.values[0]
        out["Random Forest"] = pd.Series(v, index=Xf.columns)
    except Exception:
        pass
    return out


# ---------- sidebar ----------
with st.sidebar:
    st.header("Load a real week")
    for i, ex in enumerate(META["examples"]):
        st.button(ex["label"], key=f"ex{i}", on_click=load_example, args=(i,), help=f"{ex['district']} · week of {ex['week_start']} · {ex['outcome']}", width="stretch")
    st.divider()
    m = META["metrics"]
    st.caption(f"**Test period:** {META['test_range'][0]} to {META['test_range'][1]}  \n"
               f"**Random Forest** AUC {m['Random Forest']['test_auc']:.3f}  \n**XGBoost** AUC {m['XGBoost']['test_auc']:.3f}  \n"
               "Data: dengue_srilanka_full_2006_2025 (25 districts, weekly)")

st.title("🦟 Sri Lanka Dengue Early Warning: Random Forest vs XGBoost")
st.markdown(f"Probability that a district has a **high-dengue week {H} weeks from now**, i.e. incidence above that district's own "
            "usual-high level (75th percentile of its training history). Both models predict from the same inputs.")

tab1, tab2, tab3 = st.tabs(["🔮 Predict & compare", "📊 Test-set comparison", "ℹ️ About"])

with tab1:
    c0, c1 = st.columns([1, 1])
    c0.selectbox("District", sorted(DIST), key="district", on_change=on_district)
    c1.number_input("Population (auto-filled, editable)", min_value=10000.0, step=1000.0, key="population")
    d = DIST[st.session_state["district"]]
    st.caption(f"{st.session_state['district']} · {d['province']} province · district 'high' level: **{d['baseline_p75']:.1f} cases per 100,000** per week")
    cols = st.columns(3)
    values = {}
    for i, k in enumerate(DYNAMIC):
        lab, unit = LABELS[k]
        values[k] = cols[i % 3].number_input(f"{lab} ({unit})", min_value=0.0, step=1.0 if k.startswith("cases") or k == "epi_week" else 0.5, key=k)
    values["population"] = st.session_state["population"]

    if st.button("Predict with both models", type="primary"):
        if not 1 <= values["epi_week"] <= 53:
            st.error("Epidemiological week must be between 1 and 53."); st.stop()
        X, p_rf, p_xg = predict(values, st.session_state["district"])
        l_rf, l_xg = p_rf >= 0.5, p_xg >= 0.5
        a, b, c = st.columns([1, 1, 1.3])
        a.metric("🌲 Random Forest", f"{p_rf:.0%}", "HIGH" if l_rf else "normal", delta_color="inverse" if l_rf else "normal")
        b.metric("⚡ XGBoost", f"{p_xg:.0%}", "HIGH" if l_xg else "normal", delta_color="inverse" if l_xg else "normal")
        c.metric("Difference (RF − XGBoost)", f"{(p_rf - p_xg) * 100:+.0f} pts", f"avg {(p_rf + p_xg) / 2:.0%}", delta_color="off")
        if l_rf and l_xg:
            st.error("**Both models agree: HIGH week expected.** Consider stepping up vector control and hospital readiness.")
        elif not l_rf and not l_xg:
            st.success("**Both models agree: a high week is unlikely** on current indicators.")
        else:
            st.warning(f"**The models disagree:** {'Random Forest' if l_rf else 'XGBoost'} says HIGH, the other says normal. "
                       "Treat this as an uncertain week and monitor closely.")
        left, right = st.columns([1, 1.4])
        left.bar_chart(pd.DataFrame({"Random Forest": [p_rf], "XGBoost": [p_xg]}, index=["P(high week)"]), color=[RF_C, XG_C], height=320)
        left.caption("Decision cut-off = 50%.")
        inc_now = values["cases_t"] / values["population"] * 1e5
        right.write(f"**Context.** This week: **{inc_now:.1f}** cases per 100,000, which is **{inc_now / (d['baseline_p75'] + 0.1):.1f}×** this district's usual-high level.")
        for k in DYNAMIC:
            if k != "epi_week" and not RANGES[k]["min"] <= values[k] <= RANGES[k]["max"]:
                right.warning(f"{LABELS[k][0]} = {values[k]:g} is outside the training range ({RANGES[k]['min']:.1f}–{RANGES[k]['max']:.1f}).")
        st.subheader("Why did each model say that?")
        contrib = contributions(X)
        cc = st.columns(2)
        for col, (nm, colr, unit) in zip(cc, [("Random Forest", RF_C, "probability"), ("XGBoost", XG_C, "log-odds")]):
            if nm in contrib:
                s = contrib[nm].sort_values(key=abs, ascending=False).head(8)[::-1]
                col.markdown(f"**{nm}** (top drivers, {unit} units)")
                col.bar_chart(s, horizontal=True, color=colr, height=320)
            else:
                col.info(f"Explanation for {nm} unavailable in this environment.")
        st.caption("Positive bars push the risk up, negative bars push it down (SHAP for Random Forest, native XGBoost contributions for XGBoost).")

with tab2:
    m, cm, cv = META["metrics"], META["comparison"], META["cv_auc"]
    st.subheader("Held-out test period")
    t = pd.DataFrame({"Random Forest": m["Random Forest"], "XGBoost": m["XGBoost"]}).T
    t.insert(0, "cv_auc", [cv["Random Forest"], cv["XGBoost"]])
    t = t[["cv_auc", "test_auc", "auc_lo", "auc_hi", "test_pr_auc", "recall", "precision", "f1", "accuracy", "brier"]]
    st.dataframe(t.style.format("{:.3f}").highlight_max(axis=0, color="#d6e9d6", subset=[c for c in t.columns if c != "brier"]).highlight_min(axis=0, color="#d6e9d6", subset=["brier"]), width="stretch")
    k1, k2, k3 = st.columns(3)
    k1.metric("Label agreement RF vs XGBoost", f"{cm['label_agreement']:.1%}")
    k2.metric("AUC difference (RF − XGB)", f"{cm['d_auc']:+.4f}", f"95% CI {cm['d_auc_lo']:+.4f} to {cm['d_auc_hi']:+.4f}", delta_color="off")
    k3.metric("McNemar p-value (errors differ?)", f"{cm['mcnemar_p']:.3g}", f"RF-only correct {int(cm['rf_only_correct'])} vs XGB-only {int(cm['xgb_only_correct'])}", delta_color="off")
    for fn, cap in [("04_model_comparison.png", "All models: test and cross-validated AUC"), ("05_roc_pr_calibration.png", "ROC, precision-recall, calibration, probability distributions"),
                    ("07_rf_vs_xgb_agreement.png", "Where the two models agree and disagree"), ("08_auc_by_district_year.png", "AUC by district and by year"),
                    ("11_shap_importance_compare.png", "SHAP importance: Random Forest vs XGBoost")]:
        p = ROOT / "figures" / fn
        if p.exists():
            st.image(str(p), caption=cap, width="stretch")
    inf = ROOT / "figures" / "inference_all_districts.csv"
    if inf.exists():
        st.subheader("Left-out week: both models, all 25 districts")
        st.dataframe(pd.read_csv(inf).drop(columns=["date"]).round(3), width="stretch", hide_index=True)

with tab3:
    st.markdown(f"""
**Task.** For one district and one week, predict whether the incidence **{H} weeks later** will exceed that district's own 75th-percentile level (training period).
**Data.** Weekly district-level dengue cases, population and climate for 25 districts, Dec 2006 – Dec 2025 (`dengue_srilanka_full_2006_2025.csv`).
**Models.** Random Forest and XGBoost, tuned with expanding-window time-series cross-validation (no default parameters); evaluated on the final ~25% of weeks.
**Leakage control.** `is_epidemic_year`, `cases_zscore`, `incidence_rate` and `population_source` are excluded; reporting-gap zeros were imputed.
**Caution.** Coursework decision-support prototype, not validated for operational public-health use. Notified cases are not true incidence, and the models are weaker at explosive, sudden surges.
""")
