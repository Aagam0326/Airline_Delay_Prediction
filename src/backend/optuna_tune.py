"""
optuna_tune.py — XGBoost hyperparameter tuning for the airline delay project
=============================================================================

Usage
-----
    python src/backend/optuna_tune.py                  # 50 trials, 3-fold CV (default)
    python src/backend/optuna_tune.py --trials 100     # more trials
    python src/backend/optuna_tune.py --trials 30 --folds 5 --timeout 3600

What it does
------------
1. Loads raw data and builds a leakage-free train/val split.
2. Runs Optuna TPE search over XGBoost hyperparameters.
3. Each trial uses 3-fold StratifiedKFold CV on the train set,
   refitting target-encoded features inside every fold to prevent leakage.
4. Saves the best parameters to outputs/best_params.json.
5. Saves a full study summary to outputs/tuning_summary.json.
6. Prints a ranked table of all completed trials.

Leakage note
------------
Target-encoded features (delay rates, counts) are refitted inside each
CV fold using only that fold's train rows. sklearn's cross_val_score
cannot enforce this — hence the manual fold loop in run_trial_cv().
"""

import argparse
import json
import time
import warnings

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from preprocess import OUTPUT_DIR, build_feature_frame, load_raw_data

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

BEST_PARAMS_PATH  = OUTPUT_DIR / "best_params.json"
TUNING_SUMMARY_PATH = OUTPUT_DIR / "tuning_summary.json"

# ── Search space ──────────────────────────────────────────────────────────────
# These ranges are informed by the current hand-tuned params in train.py and
# expand modestly around them so Optuna can explore without going too wild.

SEARCH_SPACE = {
    "n_estimators":      ("int",   200,  600),
    "max_depth":         ("int",   4,    10),
    "learning_rate":     ("float", 0.01, 0.15,  True),   # log scale
    "subsample":         ("float", 0.6,  1.0),
    "colsample_bytree":  ("float", 0.6,  1.0),
    "min_child_weight":  ("int",   1,    10),
    "gamma":             ("float", 0.0,  1.0),
    "reg_alpha":         ("float", 0.0,  2.0),            # L1
    "reg_lambda":        ("float", 0.5,  5.0),            # L2
}


def compute_scale_pos_weight(y_true):
    """Return neg/pos ratio for XGBoost class-imbalance handling."""
    positives = int((y_true == 1).sum())
    negatives = int((y_true == 0).sum())
    if positives == 0:
        return 1.0
    return float(max(1.0, negatives / positives))


def suggest_params(trial):
    """Sample one set of hyperparameters from the search space."""
    params = {}
    for name, spec in SEARCH_SPACE.items():
        kind = spec[0]
        if kind == "int":
            params[name] = trial.suggest_int(name, spec[1], spec[2])
        elif kind == "float":
            log = len(spec) > 3 and spec[3]
            params[name] = trial.suggest_float(name, spec[1], spec[2], log=log)
    return params


def run_trial_cv(params, airlines_train, airports, runways, n_folds, random_state):
    """
    Evaluate one set of hyperparameters with leakage-free StratifiedKFold CV.

    Each fold:
      - Fits build_feature_frame metadata only on that fold's train rows.
      - Applies pre-fitted metadata to the fold's val rows.
      - Trains XGBoost with early stopping (val set = fold val, not test set).

    Returns mean ROC-AUC across folds.
    """
    skf    = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    y_all  = airlines_train["Delay"]
    aucs   = []

    # Early stopping uses fold validation as the monitor set.
    base_xgb_params = {
        **params,
        "eval_metric":           "logloss",
        "early_stopping_rounds": 20,
        "random_state":          random_state,
        "n_jobs":                -1,
        "verbosity":             0,
    }

    for train_idx, val_idx in skf.split(airlines_train, y_all):
        fold_train = airlines_train.iloc[train_idx]
        fold_val   = airlines_train.iloc[val_idx]

        # Leakage-free feature build: fit on train rows, apply to val rows
        X_tr, y_tr, fold_meta, _ = build_feature_frame(
            fold_train, airports, runways, metadata=None
        )
        X_val, y_val, _, _ = build_feature_frame(
            fold_val, airports, runways, metadata=fold_meta
        )

        xgb_params = {
            **base_xgb_params,
            "scale_pos_weight": compute_scale_pos_weight(y_tr),
        }
        model = XGBClassifier(**xgb_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        val_prob = model.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, val_prob))

    return float(np.mean(aucs))


def build_objective(airlines_train, airports, runways, n_folds, random_state):
    """Return an Optuna objective closure that captures the dataset."""

    def objective(trial):
        params   = suggest_params(trial)
        mean_auc = run_trial_cv(params, airlines_train, airports, runways, n_folds, random_state)
        return mean_auc

    return objective


def print_trial_table(study):
    """Print a ranked summary table of all completed trials."""
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    trials_sorted = sorted(trials, key=lambda t: t.value, reverse=True)

    print(f"\n{'Rank':<5} {'Trial':>6} {'ROC-AUC':>9}  Key parameters")
    print(f"{'-'*5} {'-'*6} {'-'*9}  {'-'*50}")

    for rank, trial in enumerate(trials_sorted[:20], start=1):
        p = trial.params
        summary = (
            f"depth={p.get('max_depth','?')}  "
            f"lr={p.get('learning_rate', 0):.4f}  "
            f"n={p.get('n_estimators','?')}  "
            f"sub={p.get('subsample', 0):.2f}  "
            f"col={p.get('colsample_bytree', 0):.2f}"
        )
        marker = " ← best" if rank == 1 else ""
        print(f"{rank:<5} {trial.number:>6} {trial.value:>9.4f}  {summary}{marker}")

    if len(trials_sorted) > 20:
        print(f"  ... ({len(trials_sorted) - 20} more trials not shown)")


def run_tuning(n_trials=50, n_folds=3, timeout=None, random_state=42):
    """
    Main tuning pipeline.

    Parameters
    ----------
    n_trials      : int  — number of Optuna trials
    n_folds       : int  — CV folds per trial (3 is fast; 5 is more reliable)
    timeout       : int  — optional wall-clock limit in seconds
    random_state  : int  — seed for splits and model

    Returns
    -------
    best_params : dict
    """
    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading datasets...")
    airlines, airports, runways = load_raw_data()
    print(f"  Airlines : {airlines.shape[0]:,} rows")
    print(f"  Airports : {airports.shape[0]:,} rows")
    print(f"  Runways  : {runways.shape[0]:,} rows")

    # Hold out a clean test set — tuning never sees it
    airlines_train, airlines_test = train_test_split(
        airlines,
        test_size=0.20,
        random_state=random_state,
        stratify=airlines["Delay"],
    )
    print(f"\n  Tuning on {len(airlines_train):,} train rows")
    print(f"  Held-out test set: {len(airlines_test):,} rows (never touched during tuning)")

    # ── Run Optuna study ──────────────────────────────────────────────────────
    print(f"\nStarting Optuna TPE search")
    print(f"  Trials  : {n_trials}")
    print(f"  CV folds: {n_folds} (leakage-free — features refit inside each fold)")
    if timeout:
        print(f"  Timeout : {timeout}s")
    print(f"  Optimising: ROC-AUC (mean across folds)\n")

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study   = optuna.create_study(direction="maximize", sampler=sampler)

    objective = build_objective(airlines_train, airports, runways, n_folds, random_state)

    start = time.time()
    completed = 0

    def progress_callback(study, trial):
        nonlocal completed
        completed += 1
        elapsed = time.time() - start
        best    = study.best_value
        current = trial.value if trial.value is not None else float("nan")
        eta_str = ""
        if completed > 1:
            avg_sec = elapsed / completed
            remaining = (n_trials - completed) * avg_sec
            eta_str = f"  ETA ~{remaining/60:.1f}min"
        print(
            f"  Trial {trial.number:>3}/{n_trials}  "
            f"AUC={current:.4f}  best={best:.4f}  "
            f"[{elapsed/60:.1f}min elapsed]{eta_str}"
        )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        callbacks=[progress_callback],
        show_progress_bar=False,
    )

    elapsed_total = time.time() - start

    # ── Results ───────────────────────────────────────────────────────────────
    best_params = study.best_params
    best_auc    = study.best_value
    n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])

    print(f"\n{'='*60}")
    print(f"  Tuning complete — {n_completed} trials in {elapsed_total/60:.1f}min")
    print(f"{'='*60}")
    train_scale_pos_weight = compute_scale_pos_weight(airlines_train["Delay"])
    print(f"\n  Best CV ROC-AUC : {best_auc:.4f}")
    print(f"  Train scale_pos_weight (reference): {train_scale_pos_weight:.3f}")
    print(f"\n  Best parameters:")
    for k, v in best_params.items():
        current = {
            "n_estimators": 450, "max_depth": 9, "learning_rate": 0.035,
            "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 3,
            "gamma": 0.05, "reg_alpha": 0.0, "reg_lambda": 2.0,
        }
        old_val = current.get(k, "—")
        changed = " ←" if isinstance(v, float) and abs(v - old_val) > 0.001 else (
                  " ←" if isinstance(v, int)   and v != old_val else ""
        )
        print(f"    {k:<22} {str(v):<12}  (was {old_val}){changed}")

    print_trial_table(study)

    # ── Save outputs ──────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(exist_ok=True)

    # best_params.json — drop straight into train.py model_params
    output_params = {
        **best_params,
        "eval_metric":           "logloss",
        "early_stopping_rounds": 25,
        "random_state":          random_state,
        "n_jobs":                -1,
        "verbosity":             0,
        "_meta": {
            "best_cv_roc_auc": round(best_auc, 6),
            "n_trials":        n_completed,
            "n_folds":         n_folds,
            "tuning_time_min": round(elapsed_total / 60, 2),
            "class_imbalance_strategy": "dynamic_scale_pos_weight_per_fold",
            "train_scale_pos_weight_reference": round(train_scale_pos_weight, 6),
        },
    }
    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(output_params, f, indent=4)
    print(f"\n  Saved best params → {BEST_PARAMS_PATH}")

    # tuning_summary.json — all trials for analysis
    all_trials = [
        {
            "trial":   t.number,
            "auc":     round(t.value, 6) if t.value else None,
            "params":  t.params,
            "state":   str(t.state),
        }
        for t in sorted(study.trials, key=lambda t: t.value or 0, reverse=True)
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    with open(TUNING_SUMMARY_PATH, "w") as f:
        json.dump({"best": output_params, "all_trials": all_trials}, f, indent=2)
    print(f"  Saved full study  → {TUNING_SUMMARY_PATH}")

    # ── How to use the results ────────────────────────────────────────────────
    print(f"""
How to use best_params.json in train.py
----------------------------------------
Replace your model_params dict with:

    import json
    with open("outputs/best_params.json") as f:
        model_params = json.load(f)
    model_params.pop("_meta", None)   # remove metadata key

    model = XGBClassifier(**model_params)
""")

    return best_params


def main():
    parser = argparse.ArgumentParser(
        description="Tune XGBoost hyperparameters for airline delay prediction using Optuna."
    )
    parser.add_argument(
        "--trials", type=int, default=50,
        help="Number of Optuna trials (default: 50)"
    )
    parser.add_argument(
        "--folds", type=int, default=3,
        help="CV folds per trial — 3 is fast, 5 is more reliable (default: 3)"
    )
    parser.add_argument(
        "--timeout", type=int, default=None,
        help="Optional wall-clock time limit in seconds"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    run_tuning(
        n_trials=args.trials,
        n_folds=args.folds,
        timeout=args.timeout,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()