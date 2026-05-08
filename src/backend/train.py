import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from preprocess import (
    DATA_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
)


MODEL_PATH = MODEL_DIR / "airline_delay_model.pkl"
BEST_PARAMS_PATH = OUTPUT_DIR / "best_params.json"
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
TARGET = "DEP_DEL15"
MIN_DELAY_PRECISION = 0.00


def compute_scale_pos_weight(y_true):
    """Return neg/pos ratio for XGBoost class-imbalance handling."""
    positives = int((y_true == 1).sum())
    negatives = int((y_true == 0).sum())
    if positives == 0:
        return 1.0
    return float(max(1.0, negatives / positives))


def print_class_balance(label, y_true):
    """Print class counts so imbalance is visible in training logs."""
    counts = y_true.value_counts().reindex([0, 1], fill_value=0)
    total = int(counts.sum())
    delayed_rate = counts[1] / total if total else 0.0
    print(
        f"  {label:<16}: on-time={counts[0]:,}, delayed={counts[1]:,}, "
        f"delayed_rate={delayed_rate:.3f}"
    )


def load_train_test_data(train_path=TRAIN_FILE, test_path=TEST_FILE):
    """Load the provided model-ready train/test CSV files."""
    train_path = Path(train_path)
    test_path = Path(test_path)
    missing = [
        str(path)
        for path in (train_path, test_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing dataset file(s): {missing}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    for label, df in (("train", train_df), ("test", test_df)):
        if TARGET not in df.columns:
            raise ValueError(f"{label}.csv is missing target column {TARGET!r}")
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
        if df[TARGET].isna().any():
            raise ValueError(f"{label}.csv has missing/non-numeric {TARGET} values")
        df[TARGET] = df[TARGET].astype(int)

    return train_df, test_df


def split_features_target(df):
    """Return X/y from a model-ready dataframe."""
    X = df.drop(columns=[TARGET]).copy()
    y = df[TARGET].astype(int).copy()
    return X, y


def make_one_hot_encoder():
    """Create an encoder compatible with both old and new sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_estimator(X_train, y_train, params=None):
    """Create a pipeline that encodes categorical columns before XGBoost."""
    if params is None:
        params = get_model_params(y_train).copy()
    else:
        params = params.copy()
    params.pop("early_stopping_rounds", None)

    categorical_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = [column for column in X_train.columns if column not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", make_one_hot_encoder(), categorical_features),
            ("numeric", "passthrough", numeric_features),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", XGBClassifier(**params)),
        ]
    )


def get_transformed_feature_names(model):
    """Return feature names after one-hot encoding when available."""
    preprocessor = model.named_steps["preprocessor"]
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        return []


def get_feature_importances(model):
    """Return model importances keyed by transformed feature name."""
    xgb_model = model.named_steps["model"]
    feature_names = get_transformed_feature_names(model)
    if not feature_names or len(feature_names) != len(xgb_model.feature_importances_):
        feature_names = [f"feature_{idx}" for idx in range(len(xgb_model.feature_importances_))]
    return (
        pd.Series(xgb_model.feature_importances_, index=feature_names)
        .sort_values(ascending=False)
        .to_dict()
    )


def find_precision_constrained_threshold(y_true, y_prob, min_precision=MIN_DELAY_PRECISION):
    """
    Pick a threshold that balances delayed precision and recall.

    With min_precision=0.00, this is pure validation F1 optimization. Increase
    MIN_DELAY_PRECISION only when you want to trade recall/F1 for fewer false
    positives.
    """
    thresholds = np.linspace(0.05, 0.95, 901)
    candidates = []

    for threshold in thresholds:
        y_pred = y_prob >= threshold
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        if tp == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        candidates.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "fp": int(fp),
                "tp": int(tp),
                "meets_precision": bool(precision >= min_precision),
            }
        )

    if not candidates:
        return 0.5, {"note": "No threshold produced positive delayed predictions."}

    valid_candidates = [item for item in candidates if item["meets_precision"]]
    if valid_candidates:
        best = max(
            valid_candidates,
            key=lambda item: (item["f1"], item["recall"], -item["threshold"]),
        )
        best["note"] = f"precision >= {min_precision:.2f}"
        return best["threshold"], best

    best = max(candidates, key=lambda item: (item["precision"], item["f1"], item["recall"]))
    best["note"] = f"fallback: no threshold reached precision >= {min_precision:.2f}"
    return best["threshold"], best


def classification_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def get_model_params(y_train=None):
    """Return XGBoost hyperparameters (single source of truth for train + CV)."""
    dynamic_scale_pos_weight = (
        compute_scale_pos_weight(y_train) if y_train is not None else None
    )

    if BEST_PARAMS_PATH.exists():
        with open(BEST_PARAMS_PATH) as f:
            tuned_params = json.load(f)
        tuned_params.pop("_meta", None)
        tuned_params.pop("early_stopping_rounds", None)
        if dynamic_scale_pos_weight is not None:
            tuned_params["scale_pos_weight"] = dynamic_scale_pos_weight
        return tuned_params

    params = {
        "n_estimators": 700,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "min_child_weight": 10,
        "gamma": 0.3,
        "reg_lambda": 5,
        "reg_alpha": 1,
        "max_delta_step": 1,
        "eval_metric": "logloss",
        "early_stopping_rounds": 50,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }
    if dynamic_scale_pos_weight is not None:
        params["scale_pos_weight"] = dynamic_scale_pos_weight
    return params


def run_cross_validation(train_valid_df, n_splits=5):
    """Run StratifiedKFold CV on the provided train.csv rows only."""
    print(f"\nRunning {n_splits}-fold cross-validation on train.csv...")
    print(
        f"  {'Fold':<6} {'Val AUC':>9} {'Val AP':>9} {'Bal Acc':>9} "
        f"{'F1':>9} {'Prec':>9} {'Recall':>9} {'FP/TP':>9} {'Threshold':>10}"
    )
    print(f"  {'-'*6} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*10}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_all = train_valid_df[TARGET]

    fold_aucs        = []
    fold_avg_precisions = []
    fold_accuracies  = []
    fold_balanced_accuracies = []
    fold_f1_scores   = []
    fold_precisions   = []
    fold_recalls      = []
    fold_thresholds = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_valid_df, y_all), start=1):
        fold_train = train_valid_df.iloc[train_idx]
        fold_val = train_valid_df.iloc[val_idx]
        X_fold_train, y_fold_train = split_features_target(fold_train)
        X_fold_val, y_fold_val = split_features_target(fold_val)

        fold_model = build_estimator(X_fold_train, y_fold_train)
        fold_model.fit(X_fold_train, y_fold_train)

        val_prob  = fold_model.predict_proba(X_fold_val)[:, 1]
        fold_threshold, _ = find_precision_constrained_threshold(y_fold_val, val_prob)
        fold_metrics = classification_metrics(y_fold_val, val_prob, fold_threshold)
        fp_tp_ratio = (
            fold_metrics["false_positives"] / fold_metrics["true_positives"]
            if fold_metrics["true_positives"]
            else float("inf")
        )

        fold_aucs.append(fold_metrics["roc_auc"])
        fold_avg_precisions.append(fold_metrics["average_precision"])
        fold_accuracies.append(fold_metrics["accuracy"])
        fold_balanced_accuracies.append(fold_metrics["balanced_accuracy"])
        fold_f1_scores.append(fold_metrics["f1_score"])
        fold_precisions.append(fold_metrics["precision"])
        fold_recalls.append(fold_metrics["recall"])
        fold_thresholds.append(fold_threshold)
        print(
            f"  {fold_idx:<6} {fold_metrics['roc_auc']:>9.4f} "
            f"{fold_metrics['average_precision']:>9.4f} "
            f"{fold_metrics['balanced_accuracy']:>9.4f} "
            f"{fold_metrics['f1_score']:>9.4f} "
            f"{fold_metrics['precision']:>9.4f} "
            f"{fold_metrics['recall']:>9.4f} "
            f"{fp_tp_ratio:>9.3f} "
            f"{fold_threshold:>10.3f}"
        )

    mean_auc = float(np.mean(fold_aucs))
    std_auc  = float(np.std(fold_aucs))
    mean_avg_precision = float(np.mean(fold_avg_precisions))
    std_avg_precision  = float(np.std(fold_avg_precisions))
    mean_accuracy = float(np.mean(fold_accuracies))
    std_accuracy  = float(np.std(fold_accuracies))
    mean_balanced_accuracy = float(np.mean(fold_balanced_accuracies))
    std_balanced_accuracy  = float(np.std(fold_balanced_accuracies))
    mean_f1 = float(np.mean(fold_f1_scores))
    std_f1  = float(np.std(fold_f1_scores))
    mean_precision = float(np.mean(fold_precisions))
    std_precision  = float(np.std(fold_precisions))
    mean_recall = float(np.mean(fold_recalls))
    std_recall  = float(np.std(fold_recalls))
    median_threshold = float(np.median(fold_thresholds))

    print(f"  {'-'*6} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*10}")
    print(
        f"  {'Mean':<6} {mean_auc:>9.4f} {mean_avg_precision:>9.4f} "
        f"{mean_balanced_accuracy:>9.4f} {mean_f1:>9.4f} {mean_precision:>9.4f} {mean_recall:>9.4f}"
    )
    print(
        f"  {'Std':<6} {std_auc:>9.4f} {std_avg_precision:>9.4f} "
        f"{std_balanced_accuracy:>9.4f} {std_f1:>9.4f} {std_precision:>9.4f} {std_recall:>9.4f}"
    )
    print(f"\n  CV ROC-AUC : {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  CV PR-AUC  : {mean_avg_precision:.4f} +/- {std_avg_precision:.4f}")
    print(f"  CV F1      : {mean_f1:.4f} +/- {std_f1:.4f}")
    print(f"  CV Precision: {mean_precision:.4f} +/- {std_precision:.4f}")
    print(f"  CV Recall  : {mean_recall:.4f} +/- {std_recall:.4f}")
    print(f"  Median F1 threshold: {median_threshold:.3f}")

    return {
        "median_threshold": median_threshold,
        "fold_thresholds":  fold_thresholds,
        "fold_aucs":       fold_aucs,
        "fold_avg_precisions": fold_avg_precisions,
        "fold_accuracies": fold_accuracies,
        "fold_balanced_accuracies": fold_balanced_accuracies,
        "fold_f1_scores":  fold_f1_scores,
        "fold_precisions": fold_precisions,
        "fold_recalls":    fold_recalls,
        "mean_auc":        mean_auc,
        "std_auc":         std_auc,
        "mean_avg_precision": mean_avg_precision,
        "std_avg_precision": std_avg_precision,
        "mean_accuracy":   mean_accuracy,
        "std_accuracy":    std_accuracy,
        "mean_balanced_accuracy": mean_balanced_accuracy,
        "std_balanced_accuracy": std_balanced_accuracy,
        "mean_f1":         mean_f1,
        "std_f1":          std_f1,
        "mean_precision":  mean_precision,
        "std_precision":   std_precision,
        "mean_recall":     mean_recall,
        "std_recall":      std_recall,
        "n_splits":        n_splits,
    }


def train_model(run_cv=False, cv_splits=5):
    """
    Full training pipeline for the provided train.csv/test.csv dataset.

    Parameters
    ----------
    run_cv    : bool — whether to run CV before final training (default False)
    cv_splits : int  — number of CV folds (default 5)
    """
    print("Loading datasets...")
    train_valid_df, test_df = load_train_test_data()
    print(f"  Train CSV: {train_valid_df.shape[0]:,} rows x {train_valid_df.shape[1]} columns")
    print(f"  Test CSV : {test_df.shape[0]:,} rows x {test_df.shape[1]} columns")

    missing_test_columns = set(train_valid_df.columns).difference(test_df.columns)
    if missing_test_columns:
        raise ValueError(
            "test.csv is missing columns found in train.csv: "
            f"{sorted(missing_test_columns)}"
        )

    train_df, valid_df = train_test_split(
        train_valid_df,
        test_size=0.20,
        random_state=42,
        stratify=train_valid_df[TARGET],
    )

    cv_results = None
    if run_cv:
        cv_results = run_cross_validation(train_valid_df, n_splits=cv_splits)

    print("\nPreparing train/validation/test matrices...")
    X_train, y_train = split_features_target(train_df)
    X_valid, y_valid = split_features_target(valid_df)
    X_test, y_test = split_features_target(test_df)
    print(f"  Features          : {X_train.shape[1]}")
    print(f"  Train samples     : {X_train.shape[0]:,}")
    print(f"  Validation samples: {X_valid.shape[0]:,}")
    print(f"  Test samples      : {X_test.shape[0]:,}")
    print("\nClass balance:")
    print_class_balance("Train", y_train)
    print_class_balance("Validation", y_valid)
    print_class_balance("Test", y_test)

    # ── Final model training ──────────────────────────────────────────────────
    print("\nTraining final XGBoost model...")
    params = get_model_params(y_train)
    print(f"  scale_pos_weight        : {params.get('scale_pos_weight', 1.0):.3f}")

    model = build_estimator(X_train, y_train, params=params)
    model.fit(X_train, y_train)

    valid_prob = model.predict_proba(X_valid)[:, 1]
    threshold, threshold_details = find_precision_constrained_threshold(y_valid, valid_prob)
    validation_metrics = classification_metrics(y_valid, valid_prob, threshold)
    validation_fp_tp_ratio = (
        validation_metrics["false_positives"] / validation_metrics["true_positives"]
        if validation_metrics["true_positives"]
        else float("inf")
    )

    print(f"  F1 threshold       : {threshold:.3f}")
    print(f"  Validation F1      : {validation_metrics['f1_score']:.4f}")
    print(f"  Validation precision: {validation_metrics['precision']:.4f}")
    print(f"  Validation recall  : {validation_metrics['recall']:.4f}")
    print(f"  Validation PR-AUC  : {validation_metrics['average_precision']:.4f}")
    print(f"  Validation FP/TP   : {validation_fp_tp_ratio:.3f}")

    X_full = pd.concat([X_train, X_valid])
    y_full = pd.concat([y_train, y_valid])

    final_params = get_model_params(y_full)
    print(f"  final scale_pos_weight  : {final_params.get('scale_pos_weight', 1.0):.3f}")

    model = build_estimator(X_full, y_full, params=final_params)
    model.fit(X_full, y_full)

    if cv_results:
        gap = abs(validation_metrics["f1_score"] - cv_results["mean_f1"])
        flag = " (gap > 1% - possible variance)" if gap > 0.01 else " (ok)"
        print(f"  CV mean F1                : {cv_results['mean_f1']:.4f}{flag}")

    feature_importances = get_feature_importances(model)

    print("\nEvaluating final model on provided test.csv...")
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    test_metrics = classification_metrics(y_test, y_prob, threshold)
    test_fp_tp_ratio = (
        test_metrics["false_positives"] / test_metrics["true_positives"]
        if test_metrics["true_positives"]
        else float("inf")
    )

    print(classification_report(y_test, y_pred, target_names=["On-time", "Delayed"]))
    print(f"Accuracy          : {test_metrics['accuracy']:.4f}")
    print(f"Balanced accuracy : {test_metrics['balanced_accuracy']:.4f}")
    print(f"Delayed precision : {test_metrics['precision']:.4f}")
    print(f"Delayed recall    : {test_metrics['recall']:.4f}")
    print(f"F1-score          : {test_metrics['f1_score']:.4f}")
    print(f"ROC-AUC           : {test_metrics['roc_auc']:.4f}")
    print(f"PR-AUC            : {test_metrics['average_precision']:.4f}")
    print(f"FP/TP ratio       : {test_fp_tp_ratio:.3f}")
    if cv_results:
        print(f"CV AUC   : {cv_results['mean_auc']:.4f} +/- {cv_results['std_auc']:.4f}  "
              f"test AUC within {abs(test_metrics['roc_auc'] - cv_results['mean_auc']):.4f} of CV mean")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ── Save model bundle ─────────────────────────────────────────────────────
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    bundle = {
        "model":              model,
        "threshold":          threshold,
        "threshold_metric":   "validation_f1",
        "feature_importances": feature_importances,
        "metrics": {
            "accuracy":            test_metrics["accuracy"],
            "balanced_accuracy":   test_metrics["balanced_accuracy"],
            "f1_score":            test_metrics["f1_score"],
            "precision":           test_metrics["precision"],
            "recall":              test_metrics["recall"],
            "roc_auc":             test_metrics["roc_auc"],
            "average_precision":   test_metrics["average_precision"],
            "false_positives":     test_metrics["false_positives"],
            "true_positives":      test_metrics["true_positives"],
            "scale_pos_weight":    float(final_params.get("scale_pos_weight", 1.0)),
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
            "validation_f1":       validation_metrics["f1_score"],
            "validation_precision": validation_metrics["precision"],
            "validation_recall":   validation_metrics["recall"],
            "validation_average_precision": validation_metrics["average_precision"],
            "validation_false_positives": validation_metrics["false_positives"],
            "validation_true_positives": validation_metrics["true_positives"],
            "threshold_details":   threshold_details,
            "threshold_min_precision": MIN_DELAY_PRECISION,
            "evaluation_scope":    "provided_test_csv",
            "features":            X_train.columns.tolist(),
            "target":              TARGET,
            "train_file":          str(TRAIN_FILE),
            "test_file":           str(TEST_FILE),
            **({"cv": cv_results} if cv_results else {}),
        },
        "features": X_train.columns.tolist(),
        "target": TARGET,
        "dataset_format": "provided_train_test_csv",
    }
    joblib.dump(bundle, MODEL_PATH)

    metrics_path = OUTPUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(bundle["metrics"], f, indent=4)

    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Metrics saved to {metrics_path}")
    return bundle


def main():
    # Set run_cv=True manually if you want the slower 5-fold report.
    train_model(run_cv=False, cv_splits=5)


if __name__ == "__main__":
    main()
