import joblib
import pandas as pd

from preprocess import MODEL_DIR


MODEL_PATH = MODEL_DIR / "airline_delay_model.pkl"


CATEGORICAL_FEATURES = {
    "DEP_TIME_BLK",
    "CARRIER_NAME",
    "DEPARTING_AIRPORT",
    "PREVIOUS_AIRPORT",
}


def _to_dataframe(rows):
    """Accept DataFrame, dict, or list-of-dicts and return a DataFrame."""
    if isinstance(rows, pd.DataFrame):
        return rows.copy()
    if isinstance(rows, dict):
        return pd.DataFrame([rows])
    if isinstance(rows, list):
        return pd.DataFrame(rows)
    raise TypeError("Input must be a pandas DataFrame, dict, or list of dicts.")


def _normalize_preprocess_rows(rows, required_features):
    rows = _to_dataframe(rows)

    canonical_map = {feature.lower(): feature for feature in required_features}
    rename_map = {}
    for column in rows.columns:
        normalized = str(column).strip().lower()
        if normalized in canonical_map:
            rename_map[column] = canonical_map[normalized]
    rows = rows.rename(columns=rename_map)

    missing = set(required_features).difference(rows.columns)
    if missing:
        raise ValueError(
            "Missing required preprocessing features: "
            f"{sorted(missing)}. Provide the same feature columns used by preprocess.py."
        )

    prepared = rows[required_features].copy()
    for column in required_features:
        if column in CATEGORICAL_FEATURES:
            prepared[column] = prepared[column].astype(str).str.strip()
        else:
            prepared[column] = pd.to_numeric(prepared[column], errors="raise")
    return prepared


def predict_model_ready(X_new):
    """Return predictions for a model-ready feature DataFrame."""
    predictions, _ = predict_model_ready_with_probabilities(X_new)
    return predictions


def predict_model_ready_with_probabilities(X_new):
    """Return predictions and probabilities for already engineered features."""
    X_new = _to_dataframe(X_new)
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    threshold = bundle.get("threshold", 0.5)
    features = bundle["features"]

    missing_features = set(features).difference(X_new.columns)
    if missing_features:
        raise ValueError(
            "Missing required features: "
            f"{sorted(missing_features)}. Use predict_user_flight() for preprocess-aligned inputs."
        )

    X_new = X_new[features]
    probabilities = model.predict_proba(X_new)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return predictions, probabilities


def predict_user_flight(rows):
    """
    Predict delay from preprocess-aligned feature rows.

    Inputs must include the same feature names used by preprocess.py
    and saved in the trained model bundle.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python src/backend/train.py` first."
        )

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    threshold = bundle.get("threshold", 0.5)
    features = bundle["features"]

    user_rows = _normalize_preprocess_rows(rows, features)
    X_new = user_rows[features]

    probabilities = model.predict_proba(X_new)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    results = user_rows.copy()
    results["DelayProbability"] = probabilities
    results["Prediction"] = predictions
    results["PredictionLabel"] = results["Prediction"].map({0: "On-time", 1: "Delayed"})
    results["Threshold"] = threshold
    return results


def predict(X_new):
    """Backward-compatible alias for model-ready feature prediction."""
    return predict_model_ready(X_new)


def predict_with_probabilities(X_new):
    """Backward-compatible alias for model-ready feature prediction."""
    return predict_model_ready_with_probabilities(X_new)


def _prompt_required(prompt):
    value = input(prompt).strip()
    while not value:
        value = input(prompt).strip()
    return value


def prompt_user_flight():
    print("Enter preprocess-aligned feature values for delay prediction.")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python src/backend/train.py` first."
        )
    features = joblib.load(MODEL_PATH).get("features", [])
    if not features:
        raise ValueError("No feature list found in model bundle.")
    row = {}
    for feature in features:
        row[feature] = _prompt_required(f"{feature}: ")
    return row


def main():
    user_flight = prompt_user_flight()
    result = predict_user_flight(user_flight).iloc[0]
    print("\nPrediction result")
    if "DEPARTING_AIRPORT" in result and "PREVIOUS_AIRPORT" in result:
        print(f"  Route             : {result['PREVIOUS_AIRPORT']} -> {result['DEPARTING_AIRPORT']}")
    if "CARRIER_NAME" in result:
        print(f"  Airline           : {result['CARRIER_NAME']}")
    print(f"  Prediction        : {result['PredictionLabel']}")
    print(f"  Delay probability : {result['DelayProbability']:.4f}")
    print(f"  Threshold         : {result['Threshold']:.4f}")


if __name__ == "__main__":
    main()
