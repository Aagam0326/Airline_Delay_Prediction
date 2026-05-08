from pathlib import Path
import json
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
BACKEND_DIR = SRC_DIR / "backend"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from backend.predict import MODEL_PATH, predict_user_flight  # noqa: E402
from backend.preprocess import BASE_FEATURES, OUTPUT_DIR  # noqa: E402
from backend.train import get_feature_importances, load_train_test_data, split_features_target  # noqa: E402


CATEGORICAL_FEATURES = {
    "DEP_TIME_BLK",
    "CARRIER_NAME",
    "DEPARTING_AIRPORT",
    "PREVIOUS_AIRPORT",
}

METRICS_PATH = OUTPUT_DIR / "metrics.json"
EVALUATION_IMAGE_PATH = OUTPUT_DIR / "model_evaluation.png"

EVALUATION_TABLE_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "precision": "Delayed precision",
    "recall": "Delayed recall",
    "f1_score": "Delayed F1",
    "roc_auc": "ROC-AUC",
    "average_precision": "PR-AUC",
    "false_positives": "False positives",
    "true_positives": "True positives",
}

HIGHLIGHT_METRICS = [
    ("accuracy", "Accuracy"),
    ("balanced_accuracy", "Balanced accuracy"),
    ("precision", "Delayed precision"),
    ("recall", "Delayed recall"),
]


DEFAULT_VALUES = {
    "MONTH": 1,
    "DAY_OF_WEEK": 1,
    "DEP_TIME_BLK": "0600-0659",
    "DISTANCE_GROUP": 1,
    "SEGMENT_NUMBER": 1,
    "CONCURRENT_FLIGHTS": 1,
    "NUMBER_OF_SEATS": 150,
    "CARRIER_NAME": "AA",
    "AIRPORT_FLIGHTS_MONTH": 1000,
    "AIRLINE_FLIGHTS_MONTH": 10000,
    "AIRLINE_AIRPORT_FLIGHTS_MONTH": 200,
    "AVG_MONTHLY_PASS_AIRPORT": 1000000,
    "AVG_MONTHLY_PASS_AIRLINE": 20000000,
    "FLT_ATTENDANTS_PER_PASS": 0.0002,
    "GROUND_SERV_PER_PASS": 0.0002,
    "PLANE_AGE": 8,
    "DEPARTING_AIRPORT": "JFK",
    "LATITUDE": 40.6413,
    "LONGITUDE": -73.7781,
    "PREVIOUS_AIRPORT": "LAX",
    "PRCP": 0.0,
    "SNOW": 0.0,
    "SNWD": 0.0,
    "TMAX": 25.0,
    "AWND": 5.0,
}


@st.cache_resource(show_spinner=False)
def load_model_bundle():
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_training_metrics():
    if not METRICS_PATH.exists():
        return {}
    with open(METRICS_PATH, "r", encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


@st.cache_data(show_spinner=False)
def load_evaluation_data():
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    features = bundle["features"]
    threshold = bundle.get("threshold", 0.5)

    _, test_df = load_train_test_data()
    X, y = split_features_target(test_df)
    missing_features = set(features).difference(X.columns)
    if missing_features:
        raise RuntimeError(
            "The saved model expects features that test.csv does not contain: "
            f"{sorted(missing_features)}. Run `python src/backend/train.py` to retrain the model."
        )

    X = X[features]
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    importances = pd.Series(get_feature_importances(model)).sort_values(ascending=False)

    return {
        "y_true": y.to_numpy(),
        "y_pred": y_pred,
        "y_prob": y_prob,
        "threshold": threshold,
        "feature_importances": importances.head(12),
    }


def probability_band(probability):
    if probability >= 0.70:
        return "High delay risk"
    if probability >= 0.40:
        return "Moderate delay risk"
    return "Lower delay risk"


def format_metric_value(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, int):
        return f"{value:,}"
    return value


def apply_page_theme():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            *,*::before,*::after{box-sizing:border-box}
            html,.stApp{background-color:#1C1C1F!important;font-family:'Inter',sans-serif;color:#E4E4E7}
            #MainMenu,header,footer,[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important}
            [data-testid="stSidebar"]{background-color:#141416!important;border-right:1px solid #2A2A2E!important}
            [data-testid="stSidebar"] .stRadio>label{display:none}
            [data-testid="stSidebar"] .stRadio div[role="radiogroup"]{display:flex;flex-direction:column;gap:2px}
            [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label{display:flex!important;align-items:center;gap:10px;padding:10px 16px;border-radius:8px;cursor:pointer;font-size:14px;font-weight:500;color:#8A8A92;transition:all .15s ease;border:none!important}
            [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover{background:#1F1F23;color:#E4E4E7}
            [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-checked="true"]{background:rgba(232,93,38,.12);color:#E85D26;font-weight:600}
            [data-testid="stSidebar"] .stRadio input[type="radio"]{display:none!important}
            [data-testid="stSidebar"] .stCaption{color:#4A4A52!important;font-size:11px;padding:0 16px}
            .block-container{padding:24px 32px!important;max-width:1400px!important}
            h1,h2,h3,h4{font-family:'Inter',sans-serif!important;color:#FFFFFF!important;font-weight:600!important;letter-spacing:-.3px}
            h1{font-size:22px!important;font-weight:700!important}
            h2{font-size:17px!important}
            h3{font-size:14px!important}
            p,label,span{color:#8A8A92!important;font-family:'Inter',sans-serif}
            .page-header{display:flex;align-items:flex-end;justify-content:space-between;padding:0 0 20px;border-bottom:1px solid #2A2A2E;margin-bottom:24px}
            .page-header h1{margin:0;font-size:22px!important;color:#FFFFFF!important;font-weight:700!important}
            .page-header p{margin:4px 0 0;font-size:13px;color:#6B6B72!important}
            .page-header .badge{background:rgba(232,93,38,.15);color:#E85D26;border:1px solid rgba(232,93,38,.3);padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600}
            div[data-testid="stMetric"]{background:#232327!important;border:1px solid #2A2A2E!important;border-radius:12px!important;padding:18px 20px!important;box-shadow:none!important}
            div[data-testid="stMetricLabel"]>div{color:#6B6B72!important;font-size:11px!important;text-transform:uppercase;letter-spacing:.08em}
            div[data-testid="stMetricValue"]>div{color:#FFFFFF!important;font-size:26px!important;font-weight:700!important}
            div[data-testid="stForm"]{background:#232327!important;border:1px solid #2A2A2E!important;border-radius:12px!important;padding:20px!important;box-shadow:none!important}
            [data-testid="stExpander"]{background:#232327!important;border:1px solid #2A2A2E!important;border-radius:10px!important;box-shadow:none!important;margin-bottom:10px}
            [data-testid="stExpander"]:hover{border-color:#3A3A3F!important}
            .stTextInput input,.stNumberInput input{background-color:#1C1C1F!important;color:#E4E4E7!important;border:1px solid #2A2A2E!important;border-radius:8px!important}
            .stTextInput input:focus,.stNumberInput input:focus{border-color:#E85D26!important;box-shadow:0 0 0 2px rgba(232,93,38,.15)!important}
            .stNumberInput button{background:#2A2A2E!important;border:none!important;color:#8A8A92!important}
            .stButton>button[kind="primary"],.stFormSubmitButton>button{background:#E85D26!important;color:#FFFFFF!important;border:none!important;border-radius:8px!important;font-weight:600!important;font-size:14px!important;padding:10px 20px!important;transition:all .15s ease!important;box-shadow:0 2px 8px rgba(232,93,38,.3)!important;letter-spacing:0!important;text-transform:none!important}
            .stButton>button[kind="primary"]:hover,.stFormSubmitButton>button:hover{background:#D4511E!important;box-shadow:0 4px 16px rgba(232,93,38,.45)!important;transform:translateY(-1px)!important}
            .stTabs [data-baseweb="tab-list"]{background:#1C1C1F;border-bottom:1px solid #2A2A2E;gap:0;padding:0;border-radius:0}
            .stTabs [data-baseweb="tab"]{background:transparent;color:#6B6B72;font-weight:500;font-size:13px;padding:12px 20px;border-radius:0;border-bottom:2px solid transparent}
            .stTabs [aria-selected="true"]{background:transparent!important;color:#E85D26!important;border-bottom:2px solid #E85D26!important;font-weight:600!important}
            .metrics-table-wrap{border-radius:12px;overflow:hidden;border:1px solid #2A2A2E;background:#232327;margin-top:12px}
            .metrics-table-wrap table{width:100%;border-collapse:collapse;font-family:'Inter',sans-serif}
            .metrics-table-wrap thead tr{background:#1C1C1F;border-bottom:1px solid #2A2A2E}
            .metrics-table-wrap thead th{padding:12px 18px;text-align:left;font-size:11px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:#6B6B72}
            .metrics-table-wrap tbody tr{border-bottom:1px solid #2A2A2E;transition:background .15s ease}
            .metrics-table-wrap tbody tr:last-child{border-bottom:none}
            .metrics-table-wrap tbody tr:hover{background:#27272B}
            .metrics-table-wrap tbody td{padding:12px 18px;font-size:14px;color:#A0A0A8}
            .metrics-table-wrap tbody td:first-child{font-weight:500;color:#E4E4E7}
            .metrics-table-wrap tbody td:last-child{color:#E85D26;font-weight:600}
            .stProgress>div>div>div>div{background:#E85D26!important}
            [data-testid="stDataFrame"]{background:transparent!important;border-radius:10px}
            [data-testid="stDataFrame"]>div{background:#232327!important;border:1px solid #2A2A2E!important;border-radius:10px!important}
            [data-testid="stInfo"]{background:rgba(232,93,38,.08)!important;border:1px solid rgba(232,93,38,.2)!important;border-radius:8px!important}
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_hero(title, caption):
    st.markdown(
        f"""
        <div class="page-header">
            <div>
                <h1>✈️ {title}</h1>
                <p>{caption}</p>
            </div>
            <div class="badge">ML Platform</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_confusion_matrix_figure(evaluation_data):
    cm = confusion_matrix(evaluation_data["y_true"], evaluation_data["y_pred"], labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(cm, cmap="viridis", alpha=0.9)
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        title="Confusion Matrix",
        xlabel="Predicted label",
        ylabel="Actual label",
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["On-time", "Delayed"],
        yticklabels=["On-time", "Delayed"],
    )

    cell_labels = [["TN", "FP"], ["FN", "TP"]]
    cutoff = cm.max() / 2 if cm.size else 0
    for row in range(2):
        for col in range(2):
            color = "white" if cm[row, col] < cutoff else "black"
            ax.text(
                col,
                row,
                f"{cell_labels[row][col]}\n{cm[row, col]:,}",
                ha="center",
                va="center",
                color=color,
                fontweight="bold",
            )
    fig.tight_layout()
    return fig


def make_roc_curve_figure(evaluation_data, training_metrics):
    fpr, tpr, _ = roc_curve(evaluation_data["y_true"], evaluation_data["y_prob"])
    roc_auc = training_metrics.get("roc_auc")
    label = f"AUC: {roc_auc:.4f}" if roc_auc is not None else "ROC curve"

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(fpr, tpr, linewidth=2, color="#E85D26", label=label)
    ax.plot([0, 1], [0, 1], "--", color="#4A4A52", linewidth=1)
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    return fig


def make_precision_recall_figure(evaluation_data, training_metrics):
    precision, recall, _ = precision_recall_curve(
        evaluation_data["y_true"],
        evaluation_data["y_prob"],
    )
    avg_precision = training_metrics.get("average_precision")
    label = f"PR-AUC: {avg_precision:.4f}" if avg_precision is not None else "PR curve"

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(recall, precision, linewidth=2, color="#E85D26", label=label)
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    return fig


def make_probability_distribution_figure(evaluation_data):
    y_true = evaluation_data["y_true"]
    y_prob = evaluation_data["y_prob"]
    threshold = evaluation_data["threshold"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(y_prob[y_true == 0], bins=30, alpha=0.7, label="On-time", color="#3498DB")
    ax.hist(y_prob[y_true == 1], bins=30, alpha=0.7, label="Delayed", color="#E74C3C")
    ax.axvline(threshold, color="#8A8A92", linestyle="--", linewidth=1.5, label="Threshold")
    ax.set_title("Predicted Delay Probability")
    ax.set_xlabel("Probability")
    ax.set_ylabel("Flights")
    ax.legend()
    fig.tight_layout()
    return fig


def make_feature_importance_figure(evaluation_data):
    importances = evaluation_data["feature_importances"].sort_values()

    fig, ax = plt.subplots(figsize=(7, 5))
    importances.plot(kind="barh", ax=ax, color="#E85D26")
    ax.set_title("Top 12 Feature Importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    return fig


def render_separate_evaluation_graphs(training_metrics):
    try:
        with st.spinner("Preparing separate evaluation graphs..."):
            evaluation_data = load_evaluation_data()
    except Exception as exc:
        st.info(f"Could not prepare evaluation graphs: {exc}")
        return

    tabs = st.tabs(
        [
            "Confusion Matrix",
            "ROC Curve",
            "Precision-Recall",
            "Probability Distribution",
            "Feature Importance",
        ]
    )
    chart_builders = [
        make_confusion_matrix_figure,
        lambda data: make_roc_curve_figure(data, training_metrics),
        lambda data: make_precision_recall_figure(data, training_metrics),
        make_probability_distribution_figure,
        make_feature_importance_figure,
    ]

    for tab, build_chart in zip(tabs, chart_builders):
        with tab:
            fig = build_chart(evaluation_data)
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)


def render_evaluation_section(training_metrics):
    render_hero(
        "Evaluation Dashboard",
        "Model metrics and graphs generated with the same test-set logic as src/backend/evaluate.py.",
    )

    threshold_details = training_metrics.get("threshold_details", {})
    summary_rows = []
    if threshold_details.get("threshold") is not None:
        summary_rows.append(
            {
                "Metric": "Decision threshold",
                "Value": f"{float(threshold_details['threshold']):.4f}",
            }
        )

    for key, label in EVALUATION_TABLE_LABELS.items():
        if key in training_metrics:
            summary_rows.append(
                {
                    "Metric": label,
                    "Value": format_metric_value(training_metrics[key]),
                }
            )

    if (
        "false_positives" in training_metrics
        and "true_positives" in training_metrics
        and training_metrics["true_positives"]
    ):
        fp_tp_ratio = training_metrics["false_positives"] / training_metrics["true_positives"]
        summary_rows.append({"Metric": "FP/TP ratio", "Value": f"{fp_tp_ratio:.4f}"})

    if training_metrics.get("evaluation_scope"):
        summary_rows.append(
            {
                "Metric": "Evaluation scope",
                "Value": training_metrics["evaluation_scope"],
            }
        )

    if summary_rows:
        metric_cols = st.columns(4)
        for col, (metric_key, label) in zip(metric_cols, HIGHLIGHT_METRICS):
            if metric_key in training_metrics:
                col.metric(label, format_metric_value(training_metrics[metric_key]))

    if summary_rows:
        st.subheader("Detailed Metrics")
        rows_html = "".join(
            f"<tr><td>{row['Metric']}</td><td>{row['Value']}</td></tr>"
            for row in summary_rows
        )
        st.markdown(
            f"""
            <div class="metrics-table-wrap">
                <table>
                    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
                    <tbody>{rows_html}</tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Evaluation metric values not found. Run `python src/backend/evaluate.py`.")

    if EVALUATION_IMAGE_PATH.exists():
        with st.expander("Evaluation overview image from evaluate.py"):
            st.image(str(EVALUATION_IMAGE_PATH), use_container_width=True)

    render_separate_evaluation_graphs(training_metrics)


def render_prediction_page(model_features):
    render_hero(
        "Airline Delay Predictor",
        "Enter custom flight details and estimate the probability of a delayed arrival.",
    )

    with st.form("flight_form"):
        st.subheader("Simulate Flight Scenarios")
        st.caption("Adjust the operational and environmental conditions below to view model predictions.")
        
        # Group features logically for a cleaner user experience
        feature_groups = {}
        for f in model_features:
            if f in ["MONTH", "DAY_OF_WEEK", "DEP_TIME_BLK", "SEGMENT_NUMBER", "CONCURRENT_FLIGHTS"]:
                feature_groups.setdefault("🗓️ Time & Schedule", []).append(f)
            elif f in ["DEPARTING_AIRPORT", "PREVIOUS_AIRPORT", "LATITUDE", "LONGITUDE", "DISTANCE_GROUP"]:
                feature_groups.setdefault("📍 Route & Location", []).append(f)
            elif f in ["CARRIER_NAME", "NUMBER_OF_SEATS", "PLANE_AGE"]:
                feature_groups.setdefault("✈️ Carrier & Aircraft", []).append(f)
            elif f in ["PRCP", "SNOW", "SNWD", "TMAX", "AWND"]:
                feature_groups.setdefault("🌦️ Weather Conditions", []).append(f)
            else:
                feature_groups.setdefault("📊 Traffic & Operational Metrics", []).append(f)

        user_flight = {}
        for group_name, group_features in feature_groups.items():
            with st.expander(group_name, expanded=True):
                # Use a fixed 3 columns so input boxes maintain a proper ratio and don't stretch
                cols = st.columns(3)
                for idx, feature in enumerate(group_features):
                    target_col = cols[idx % len(cols)]
                    default_value = DEFAULT_VALUES.get(feature, 0.0)
                    
                    # Convert ALL_CAPS_SNAKE_CASE to more readable Title Case
                    friendly_label = feature.replace("_", " ").title()
                    
                    with target_col:
                        if feature in CATEGORICAL_FEATURES:
                            user_flight[feature] = st.text_input(
                                friendly_label,
                                value=str(default_value),
                            ).strip()
                        else:
                            user_flight[feature] = st.number_input(
                                friendly_label,
                                value=float(default_value),
                                step=1.0 if isinstance(default_value, int) else 0.1
                            )

        submitted = st.form_submit_button("Predict Delay Probability", type="primary", use_container_width=True)

    if not submitted:
        st.info("Set the flight inputs above, then submit to see the model prediction.")
        return

    try:
        with st.spinner("Building features and predicting..."):
            result = predict_user_flight(user_flight).iloc[0]
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    probability = float(result["DelayProbability"])
    label = result["PredictionLabel"]
    threshold = float(result["Threshold"])
    decision_margin = probability - threshold

    st.subheader("Prediction")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Result", label)
    metric_col2.metric("Delay probability", f"{probability:.1%}")
    metric_col3.metric("Decision threshold", f"{threshold:.1%}", f"{decision_margin:+.1%}")

    st.progress(min(max(probability, 0.0), 1.0), text=probability_band(probability))

    details_payload = {feature: result.get(feature) for feature in model_features}
    details_payload["Threshold"] = f"{threshold:.3f}"
    details = pd.DataFrame([details_payload])
    st.dataframe(details, use_container_width=True, hide_index=True)


def main():
    st.set_page_config(
        page_title="Airline Delay Predictor",
        page_icon="✈️",
        layout="wide",
    )
    apply_page_theme()
    
    # Configure Matplotlib for Dark Theme
    plt.style.use('dark_background')
    plt.rcParams.update({
        "figure.facecolor": "none",
        "axes.facecolor": "#232327",
        "axes.edgecolor": "#2A2A2E",
        "grid.color": "#2A2A2E",
        "text.color": "#E4E4E7",
        "axes.labelcolor": "#8A8A92",
        "xtick.color": "#6B6B72",
        "ytick.color": "#6B6B72",
        "font.family": "sans-serif",
    })

    with st.sidebar:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;padding:20px 16px 16px;
             border-bottom:1px solid #2A2A2E;margin-bottom:8px">
            <div style="width:34px;height:34px;background:#E85D26;border-radius:8px;
                 display:flex;align-items:center;justify-content:center;font-size:18px">✈️</div>
            <div>
                <div style="font-size:15px;font-weight:700;color:#FFFFFF">AirDelay</div>
                <div style="font-size:11px;color:#6B6B72">ML Prediction Platform</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        page = st.radio(
            "Navigation",
            ["✈️  Prediction", "📊  Analytics"],
        )
        st.caption("Use Prediction for custom inputs and Analytics for model evaluation.")

    if not MODEL_PATH.exists():
        st.error("Model file not found. Run `python src/backend/train.py` first.")
        st.stop()

    try:
        bundle = load_model_bundle()
    except Exception as exc:
        st.error(f"Could not load model bundle: {exc}")
        st.stop()
        return

    model_features = bundle.get("features", BASE_FEATURES)
    training_metrics = load_training_metrics()

    if page == "✈️  Prediction":
        render_prediction_page(model_features)
    else:
        render_evaluation_section(training_metrics)


if __name__ == "__main__":
    main()