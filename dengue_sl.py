"""
Shared logic for the Sri Lanka district-level dengue early-warning models (RF vs XGBoost).
Imported by the notebook, the pickled pipelines and the Streamlit app so training and
inference use exactly the same transformations.
"""
import numpy as np
import pandas as pd

HORIZON = 2          # predict 2 weeks ahead
ELEV = {"Low": 0, "Mid": 1, "Highland": 2}

# fields the user enters (change every week)
DYNAMIC = ["epi_week", "cases_t", "cases_t1", "cases_t2", "cases_t3",
           "rain_t", "rain_4wk", "rain_8wk", "tmax_4wk", "tmin_4wk", "humidity_4wk"]
# fields looked up automatically from the chosen district
STATIC = ["population", "latitude", "longitude", "coastal", "elevation_code", "baseline_p75"]
RAW_FIELDS = DYNAMIC + STATIC
CLIMATE_FIELDS = ["rain_t", "rain_4wk", "rain_8wk", "tmax_4wk", "tmin_4wk", "humidity_4wk", "temp_range"]

# columns of the CSV that must NOT be used as features (documented in the notebook)
LEAKY = {"is_epidemic_year": "defined from a whole year's outcome (looks into the future)",
         "cases_zscore": "z-score computed with global mean/std of the whole period",
         "population_source": "bookkeeping flag, not a risk factor",
         "incidence_rate": "just cases / population (re-derived from inputs)"}


def add_derived(X):
    """Derived features from the raw fields (works on a DataFrame with RAW_FIELDS)."""
    X = pd.DataFrame(X).copy()
    pop = X["population"].astype(float)
    for k in ["cases_t", "cases_t1", "cases_t2", "cases_t3"]:
        X["inc_" + k[6:]] = X[k] / pop * 1e5                       # cases per 100,000 people
    X["rel_t"] = X["inc_t"] / (X["baseline_p75"] + 0.1)            # current level vs the district's usual 'high'
    X["inc_mean4"] = X[["inc_t", "inc_t1", "inc_t2", "inc_t3"]].mean(axis=1)
    X["rel_mean4"] = X["inc_mean4"] / (X["baseline_p75"] + 0.1)
    X["growth"] = (X["inc_t"] + 0.1) / (X["inc_t1"] + 0.1)
    X["trend3"] = X["inc_t"] - X["inc_t3"]
    X["week_sin"] = np.sin(2 * np.pi * X["epi_week"] / 52.0)
    X["week_cos"] = np.cos(2 * np.pi * X["epi_week"] / 52.0)
    X["temp_range"] = X["tmax_4wk"] - X["tmin_4wk"]
    return X.drop(columns=["epi_week", "cases_t", "cases_t1", "cases_t2", "cases_t3"])


def load_clean(path):
    """Read the CSV, repair implausible zeros, add helper columns; sorted by week then district."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["start.date"])
    df = df.sort_values(["district", "date"]).reset_index(drop=True)
    # A zero in a normally busy district (+-13 week median >= 15 cases) is a reporting gap, not 'no dengue'
    med = df.groupby("district")["cases"].transform(lambda s: s.rolling(27, center=True, min_periods=9).median())
    df["unreported"] = (df["cases"] == 0) & (med >= 15)
    df.loc[df["unreported"], "cases"] = np.nan
    df["cases"] = df.groupby("district")["cases"].transform(lambda s: s.interpolate(limit_direction="both"))
    df["incidence_rate"] = df["cases"] / df["population"] * 1e5
    df["elevation_code"] = df["elevation_category"].map(ELEV)
    return df.sort_values(["date", "district"]).reset_index(drop=True)


def build_supervised(df, horizon=HORIZON):
    """One row per (district, week t): inputs known at week t -> incidence at week t+horizon."""
    d = df.sort_values(["district", "date"]).reset_index(drop=True)
    g = d.groupby("district", sort=False)
    X = pd.DataFrame({"district": d["district"], "date": d["date"], "epi_week": d["week"], "cases_t": d["cases"]})
    for k in (1, 2, 3):
        X[f"cases_t{k}"] = g["cases"].shift(k)
    X["rain_t"] = d["rainfall_mm"]
    X["rain_4wk"] = g["rainfall_mm"].transform(lambda s: s.rolling(4).mean())
    X["rain_8wk"] = g["rainfall_mm"].transform(lambda s: s.rolling(8).mean())
    X["tmax_4wk"] = g["temp_max_c"].transform(lambda s: s.rolling(4).mean())
    X["tmin_4wk"] = g["temp_min_c"].transform(lambda s: s.rolling(4).mean())
    X["humidity_4wk"] = g["humidity_pct"].transform(lambda s: s.rolling(4).mean())
    X["population"], X["latitude"], X["longitude"] = d["population"], d["latitude"], d["longitude"]
    X["coastal"], X["elevation_code"] = d["coastal_district"], d["elevation_code"]
    X["y_inc_next"] = g["incidence_rate"].shift(-horizon)
    X["y_cases_next"] = g["cases"].shift(-horizon)
    X = X.dropna().sort_values(["date", "district"]).reset_index(drop=True)
    return X
