from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

FULL_DATA_FILE = DATA_DIR / "full_data_flightdelay.csv"
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"

AIRLINES_FILE = FULL_DATA_FILE
LEGACY_AIRLINES_FILE = None
AIRPORTS_FILE = None
RUNWAYS_FILE = None

TARGET = "DEP_DEL15"

# ── Final cleaned feature list ────────────────────────────────────────────
BASE_FEATURES = [
    "MONTH",
    "DAY_OF_WEEK",
    "DEP_TIME_BLK",
    "DISTANCE_GROUP",
    "SEGMENT_NUMBER",
    "CONCURRENT_FLIGHTS",
    "NUMBER_OF_SEATS",
    "CARRIER_NAME",
    "AIRPORT_FLIGHTS_MONTH",
    "AIRLINE_FLIGHTS_MONTH",
    "AIRLINE_AIRPORT_FLIGHTS_MONTH",
    "AVG_MONTHLY_PASS_AIRPORT",
    "AVG_MONTHLY_PASS_AIRLINE",
    "FLT_ATTENDANTS_PER_PASS",
    "GROUND_SERV_PER_PASS",
    "PLANE_AGE",
    "DEPARTING_AIRPORT",
    "LATITUDE",
    "LONGITUDE",
    "PREVIOUS_AIRPORT",
    "PRCP",
    "SNOW",
    "SNWD",
    "TMAX",
    "AWND",
]

HISTORICAL_FEATURES = [
    "CARRIER_HISTORICAL",
    "DEP_AIRPORT_HIST",
    "DAY_HISTORICAL",
    "DEP_BLOCK_HIST",
]

FEATURES = BASE_FEATURES + HISTORICAL_FEATURES

# ── Reduced target-encoding features ──────────────────────────────────────
RATE_FEATURES = []

# ── Keep only safe encodings ──────────────────────────────────────────────
ENCODED_FEATURES = []

# ── Data loading ───────────────────────────────────────────────────────────────

def load_raw_data(
    airlines_path=AIRLINES_FILE,
    airports_path=AIRPORTS_FILE,
    runways_path=RUNWAYS_FILE,
):
    """Load project data and standardize the flight table schema."""
    airlines = _load_airlines_data(airlines_path)
    airports = pd.read_excel(airports_path) if airports_path else pd.DataFrame()
    runways = pd.read_excel(runways_path) if runways_path else pd.DataFrame()
    return airlines, airports, runways


# ── Internal helpers ───────────────────────────────────────────────────────────

def _load_airlines_data(airlines_path):
    airlines_path = Path(airlines_path)
    if airlines_path.suffix.lower() == ".csv":
        return _load_flights_csv(airlines_path)
    return pd.read_excel(airlines_path)


def _hhmm_to_minutes(series):
    values = pd.to_numeric(series, errors="coerce")
    hours = values // 100
    minutes = values % 100
    return hours * 60 + minutes


def _load_flights_csv(csv_path):
    flights = pd.read_csv(csv_path)
    if TARGET in flights.columns:
        return flights

    usecols = [
        "FL_DATE",
        "AIRLINE_CODE",
        "FL_NUMBER",
        "ORIGIN",
        "DEST",
        "CRS_DEP_TIME",
        "DEP_DELAY",
        "CANCELLED",
        "DIVERTED",
        "CRS_ELAPSED_TIME",
    ]
    flights = pd.read_csv(csv_path, usecols=usecols)
    flights["FL_DATE"] = pd.to_datetime(flights["FL_DATE"], errors="coerce")

    airlines = pd.DataFrame(
        {
            "Airline": flights["AIRLINE_CODE"].astype(str),
            "Flight": pd.to_numeric(flights["FL_NUMBER"], errors="coerce"),
            "AirportFrom": flights["ORIGIN"].astype(str),
            "AirportTo": flights["DEST"].astype(str),
            "DayOfWeek": flights["FL_DATE"].dt.dayofweek + 1,
            "Time": _hhmm_to_minutes(flights["CRS_DEP_TIME"]),
            "Length": pd.to_numeric(flights["CRS_ELAPSED_TIME"], errors="coerce"),
            "Delay": (
                (pd.to_numeric(flights["DEP_DELAY"], errors="coerce") >= 15)
                | (pd.to_numeric(flights["CANCELLED"], errors="coerce").fillna(0) == 1)
                | (pd.to_numeric(flights["DIVERTED"], errors="coerce").fillna(0) == 1)
            ).astype(int),
        }
    )
    airlines = airlines.dropna(
        subset=[
            "Airline",
            "Flight",
            "AirportFrom",
            "AirportTo",
            "DayOfWeek",
            "Time",
            "Length",
            "Delay",
        ]
    ).copy()
    airlines["Flight"] = airlines["Flight"].astype(int)
    airlines["DayOfWeek"] = airlines["DayOfWeek"].astype(int)
    airlines["Time"] = airlines["Time"].astype(int)
    return airlines


def _validate_columns(airlines):
    required = set(BASE_FEATURES + [TARGET])
    missing = required.difference(airlines.columns)
    if missing:
        raise ValueError(f"Airlines data is missing required columns: {sorted(missing)}")


def _available_features(df):
    return [feature for feature in FEATURES if feature in df.columns]


def _add_time_and_route_features(df):
    if "Time" not in df.columns:
        return

    df["DepartureHour"]   = df["Time"] // 60

    df["TimeSin"] = np.sin(2 * np.pi * df["Time"] / 1440)
    df["TimeCos"] = np.cos(2 * np.pi * df["Time"] / 1440)

    if {"AirportFrom", "AirportTo"}.issubset(df.columns):
        df["Route"] = df["AirportFrom"].astype(str) + "_" + df["AirportTo"].astype(str)

    df["IsRushHour"] = df["DepartureHour"].isin([7,8,9,17,18,19]).astype(int)


def _fit_metadata(train_df):
    """
    Leak-free target encoding using K-Fold (out-of-fold encoding).

    Each row's delay rate is computed WITHOUT using its own target,
    preventing self-leakage.
    """
    global_delay_rate = float(train_df[TARGET].mean())
    smoothing_strength = 20

    metadata = {
        "features": _available_features(train_df),
        "global_delay_rate": global_delay_rate,
        "smoothing_strength": smoothing_strength,
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # ── K-Fold Target Encoding (Leak-Free) ───────────────────────────────
    for feature_name, source_column, metadata_key in RATE_FEATURES:

        oof_values = np.zeros(len(train_df))

        for train_idx, val_idx in kf.split(train_df):
            train_fold = train_df.iloc[train_idx]
            val_fold = train_df.iloc[val_idx]

            grouped = train_fold.groupby(source_column)[TARGET].agg(["sum", "count"])

            smoothed_rate = (
                (grouped["sum"] + smoothing_strength * global_delay_rate)
                / (grouped["count"] + smoothing_strength)
            )

            oof_values[val_idx] = val_fold[source_column].map(smoothed_rate)

        # Fill missing with global mean
        oof_values = np.where(np.isnan(oof_values), global_delay_rate, oof_values)

        # ⚠️ IMPORTANT: store FULL mapping for inference
        full_grouped = train_df.groupby(source_column)[TARGET].agg(["sum", "count"])
        full_smoothed = (
            (full_grouped["sum"] + smoothing_strength * global_delay_rate)
            / (full_grouped["count"] + smoothing_strength)
        )

        metadata[metadata_key] = full_smoothed.to_dict()

    for _, source_column, metadata_key in ENCODED_FEATURES:
        values = sorted(train_df[source_column].dropna().unique())
        metadata[metadata_key] = {value: index for index, value in enumerate(values)}

    return metadata


def _apply_delay_rate_features(df, metadata):
    """Map pre-fitted delay rates; unseen categories fall back to global mean."""
    default_rate = metadata["global_delay_rate"]
    for feature_name, source_column, metadata_key in RATE_FEATURES:
        df[feature_name] = df[source_column].map(metadata[metadata_key]).fillna(default_rate)


def _apply_encoded_features(df, metadata):
    """Map pre-fitted label encodings; unseen categories get -1."""
    for feature_name, source_column, metadata_key in ENCODED_FEATURES:
        df[feature_name] = df[source_column].map(metadata[metadata_key]).fillna(-1).astype(int)


# ── Public API ─────────────────────────────────────────────────────────────────

def build_feature_frame(airlines, airports, runways, metadata=None):
    """
    Build a feature matrix from an airlines slice.

    Parameters
    ----------
    airlines : pd.DataFrame
        A slice of the raw airlines data (train, validation, or test).
        Must NOT contain rows from outside the intended split when
        metadata=None — otherwise delay rates leak from those rows.
    airports : pd.DataFrame
        Full airports lookup table (no target column, safe to pass in full).
    runways : pd.DataFrame
        Full runways lookup table (no target column, safe to pass in full).
    metadata : dict or None
        If None, statistics are fitted from `airlines` (use only for train).
        If provided, pre-fitted statistics are applied — use for val/test.

    Returns
    -------
    X        : pd.DataFrame  — feature matrix
    y        : pd.Series     — target (Delay)
    metadata : dict          — fitted statistics (pass to val/test calls)
    df       : pd.DataFrame  — full frame with all intermediate columns
    """
    _validate_columns(airlines)

    df = airlines.copy()
    _add_time_and_route_features(df)

    # ── Fit or apply target-encoded features ──────────────────────────────────
    # LEAKAGE NOTE: _fit_metadata reads TARGET (Delay) — it must only ever
    # see train rows. When metadata is None this is the training call; when
    # metadata is provided this is a val/test call and no fitting occurs.
    if metadata is None:
        metadata = _fit_metadata(df)   # fits on train rows only

    _apply_delay_rate_features(df, metadata)
    _apply_encoded_features(df, metadata)

    active_features = metadata.get("features", _available_features(df))
    X = df[active_features]
    y = df[TARGET]
    return X, y, metadata, df


def split_and_preprocess(
    airlines,
    airports,
    runways,
    test_size=0.20,
    valid_size=0.20,
    random_state=42,
):
    """
    Convenience function: split raw data then build leakage-free features.

    This is the SAFE replacement for load_and_preprocess(). It performs
    the train/validation/test split before any target-based fitting, so
    delay rates are always computed on train rows only.

    Returns
    -------
    X_train, X_valid, X_test : pd.DataFrame
    y_train, y_valid, y_test : pd.Series
    metadata                 : dict  (pass to train.py bundle for predict.py)
    """
    _validate_columns(airlines)

    # Step 1 — split raw rows (no features computed yet)
    airlines_train_valid, airlines_test = train_test_split(
        airlines,
        test_size=test_size,
        random_state=random_state,
        stratify=airlines[TARGET],
    )
    airlines_train, airlines_valid = train_test_split(
        airlines_train_valid,
        test_size=valid_size,
        random_state=random_state,
        stratify=airlines_train_valid[TARGET],
    )

    print(f"  Train      : {len(airlines_train):,} rows")
    print(f"  Validation : {len(airlines_valid):,} rows")
    print(f"  Test       : {len(airlines_test):,} rows")

    # Step 2 — fit metadata on train only, apply to all splits
    X_train, y_train, metadata, _ = build_feature_frame(
        airlines_train, airports, runways, metadata=None   # fits here
    )
    X_valid, y_valid, _, _ = build_feature_frame(
        airlines_valid, airports, runways, metadata=metadata  # applies only
    )
    X_test, y_test, _, _ = build_feature_frame(
        airlines_test, airports, runways, metadata=metadata   # applies only
    )

    return X_train, X_valid, X_test, y_train, y_valid, y_test, metadata


# ── Legacy helper (DEPRECATED — use split_and_preprocess instead) ─────────────

def load_and_preprocess():
    """
    DEPRECATED: This function fits metadata on the full dataset before
    splitting, which causes target leakage into the test set.

    Use split_and_preprocess() instead:

        airlines, airports, runways = load_raw_data()
        X_train, X_valid, X_test, y_train, y_valid, y_test, metadata = (
            split_and_preprocess(airlines, airports, runways)
        )
    """
    raise RuntimeError(
        "load_and_preprocess() is deprecated due to data leakage.\n"
        "Use split_and_preprocess() instead — see docstring for usage."
    )
