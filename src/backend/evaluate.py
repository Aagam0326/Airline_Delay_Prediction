import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from preprocess import MODEL_DIR, OUTPUT_DIR
from train import get_feature_importances, load_train_test_data, split_features_target


MODEL_PATH = MODEL_DIR / "airline_delay_model.pkl"


def evaluate_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python src/backend/train.py` first."
        )

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
    fpr, tpr, _ = roc_curve(y, y_prob)
    precision, recall, _ = precision_recall_curve(y, y_prob)
    accuracy = accuracy_score(y, y_pred)
    balanced_accuracy = balanced_accuracy_score(y, y_pred)
    delayed_precision = precision_score(y, y_pred, zero_division=0)
    delayed_recall = recall_score(y, y_pred, zero_division=0)
    delayed_f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_prob)
    avg_precision = average_precision_score(y, y_prob)

    OUTPUT_DIR.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Airline Delay Model Evaluation - Provided Test CSV",
        fontsize=14,
        fontweight="bold",
    )

    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["On-time", "Delayed"])
    disp.plot(ax=axes[0, 0], colorbar=False, cmap="Blues", values_format="d")
    axes[0, 0].set_title("Confusion Matrix")
    axes[0, 0].set_xlabel("Predicted label")
    axes[0, 0].set_ylabel("Actual label")
    cell_names = [["TN", "FP"], ["FN", "TP"]]
    max_cell = cm.max()
    for row in range(2):
        for col in range(2):
            text_color = "white" if cm[row, col] > max_cell / 2 else "black"
            axes[0, 0].text(
                col,
                row + 0.28,
                cell_names[row][col],
                ha="center",
                va="center",
                color=text_color,
                fontsize=11,
                fontweight="bold",
            )

    axes[0, 1].plot(fpr, tpr, linewidth=2, label=f"AUC: {roc_auc:.4f}")
    axes[0, 1].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0, 1].set_title("ROC Curve")
    axes[0, 1].set_xlabel("False Positive Rate")
    axes[0, 1].set_ylabel("True Positive Rate")
    axes[0, 1].legend(loc="lower right")

    axes[0, 2].plot(recall, precision, linewidth=2, color="#2E7D32")
    axes[0, 2].set_title("Precision-Recall Curve")
    axes[0, 2].set_xlabel("Recall")
    axes[0, 2].set_ylabel("Precision")
    axes[0, 2].text(
        0.05,
        0.05,
        f"AP: {avg_precision:.4f}",
        transform=axes[0, 2].transAxes,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    axes[1, 0].hist(
        y_prob[y == 0],
        bins=30,
        alpha=0.7,
        label="On-time",
        color="#1976D2",
    )
    axes[1, 0].hist(
        y_prob[y == 1],
        bins=30,
        alpha=0.7,
        label="Delayed",
        color="#D32F2F",
    )
    axes[1, 0].axvline(threshold, color="black", linestyle="--", linewidth=2)
    axes[1, 0].set_title("Predicted Delay Probability")
    axes[1, 0].set_xlabel("Probability")
    axes[1, 0].set_ylabel("Flights")
    axes[1, 0].legend()

    importances = pd.Series(get_feature_importances(model)).sort_values()
    importances.tail(12).plot(kind="barh", ax=axes[1, 1], color="#FF5722")
    axes[1, 1].set_title("Top 12 Feature Importances")
    axes[1, 1].set_xlabel("Importance")

    tn, fp, fn, tp = cm.ravel()
    false_positive_rate = fp / (fp + tn) if fp + tn else 0
    false_negative_rate = fn / (fn + tp) if fn + tp else 0
    fp_tp_ratio = fp / tp if tp else float("inf")
    on_time_count = int((y == 0).sum())
    delayed_count = int((y == 1).sum())
    delayed_rate = delayed_count / len(y) if len(y) else 0
    metrics_text = "\n".join(
        [
            f"Threshold: {threshold:.3f}",
            f"On-time flights: {on_time_count:,}",
            f"Delayed flights: {delayed_count:,}",
            f"Delayed rate: {delayed_rate:.4f}",
            f"Accuracy: {accuracy:.4f}",
            f"Balanced Accuracy: {balanced_accuracy:.4f}",
            f"Delayed Precision: {delayed_precision:.4f}",
            f"Delayed Recall: {delayed_recall:.4f}",
            f"Delayed F1: {delayed_f1:.4f}",
            f"ROC-AUC: {roc_auc:.4f}",
            f"PR-AUC: {avg_precision:.4f}",
            f"True Negatives: {tn:,}",
            f"False Positives: {fp:,}",
            f"FP/TP Ratio: {fp_tp_ratio:.3f}",
            f"False Positive Rate: {false_positive_rate:.4f}",
            f"False Negatives: {fn:,}",
            f"False Negative Rate: {false_negative_rate:.4f}",
            f"True Positives: {tp:,}",
        ]
    )
    axes[1, 2].axis("off")
    axes[1, 2].set_title("Metrics Summary")
    axes[1, 2].text(
        0.05,
        0.95,
        metrics_text,
        transform=axes[1, 2].transAxes,
        va="top",
        fontsize=12,
        linespacing=1.6,
    )

    output_path = OUTPUT_DIR / "model_evaluation.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved evaluation plot to {output_path}")
    print("\nClass balance:")
    print(f"  On-time flights: {on_time_count:,}")
    print(f"  Delayed flights: {delayed_count:,}")
    print(f"  Delayed rate   : {delayed_rate:.4f}")
    print("\nClassification report:")
    print(classification_report(y, y_pred, target_names=["On-time", "Delayed"]))
    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Balanced accuracy : {balanced_accuracy:.4f}")
    print(f"Delayed precision: {delayed_precision:.4f}")
    print(f"Delayed recall   : {delayed_recall:.4f}")
    print(f"Delayed F1       : {delayed_f1:.4f}")
    print(f"ROC-AUC          : {roc_auc:.4f}")
    print(f"PR-AUC           : {avg_precision:.4f}")
    print(f"Threshold        : {threshold:.3f}")
    print(f"FP/TP ratio      : {fp_tp_ratio:.3f}")
    print("Confusion matrix counts:")
    print(f"  True Negatives : {tn:,}")
    print(f"  False Positives: {fp:,}")
    print(f"  False Negatives: {fn:,}")
    print(f"  True Positives : {tp:,}")


def main():
    evaluate_model()


if __name__ == "__main__":
    main()
