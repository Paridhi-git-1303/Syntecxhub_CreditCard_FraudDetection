"""
Credit Card / Online Payments Fraud Detection
================================================

End-to-end pipeline that:
  1. Loads and explores a highly imbalanced fraud dataset (EDA + class-imbalance plots).
  2. Engineers a small set of balance-based features.
  3. Builds a train/test split that is stratified on the fraud label.
  4. Applies a sampling strategy on the training data only
     (none / random undersampling / random oversampling / SMOTE).
  5. Trains a Random Forest and (optionally) an XGBoost classifier.
  6. Evaluates both models with precision, recall, F1, ROC-AUC, PR-AUC,
     confusion matrices and a precision-recall curve.
  7. Sweeps the decision threshold and prints a business-style cost/benefit
     table so a non-technical stakeholder can pick an operating point.
  8. Persists trained models, plots and a text report under ``outputs/``.

Usage
-----
    python src/fraud_detection.py --data data/AIML_Dataset.csv \
        --sampling smote --model both --fraud-cost 500 --review-cost 5

Run ``python src/fraud_detection.py --help`` for every option.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # safe for headless / CI environments
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.utils import resample
import joblib

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    XGBOOST_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE

    IMBLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    IMBLEARN_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Paths / constants
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"
TARGET = "isFraud"
RANDOM_STATE = 42

for d in (FIGURES_DIR, MODELS_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def savefig(name: str) -> None:
    path = FIGURES_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [saved figure] {path.relative_to(ROOT_DIR)}")


# --------------------------------------------------------------------------- #
# 1. Data loading
# --------------------------------------------------------------------------- #
def load_data(path: str) -> pd.DataFrame:
    section("1. LOADING DATA")
    df = pd.read_csv(path)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    return df


# --------------------------------------------------------------------------- #
# 2. Exploratory Data Analysis
# --------------------------------------------------------------------------- #
def run_eda(df: pd.DataFrame) -> None:
    section("2. EXPLORATORY DATA ANALYSIS")

    fraud_counts = df[TARGET].value_counts()
    fraud_pct = round(fraud_counts.get(1, 0) / len(df) * 100, 4)
    print(f"Class counts:\n{fraud_counts}")
    print(f"Fraud rate: {fraud_pct}% of all transactions")

    if "isFlaggedFraud" in df.columns:
        print(f"\nisFlaggedFraud counts:\n{df['isFlaggedFraud'].value_counts()}")

    # --- Class imbalance bar chart -----------------------------------------
    plt.figure(figsize=(5, 4))
    ax = sns.barplot(x=fraud_counts.index.astype(str), y=fraud_counts.values,
                      palette=["#2c7fb8", "#d7191c"])
    ax.set_xticklabels(["Legit (0)", "Fraud (1)"])
    for i, v in enumerate(fraud_counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    plt.title(f"Class Imbalance — Fraud is only {fraud_pct}% of transactions")
    plt.ylabel("Count")
    savefig("01_class_imbalance.png")

    # --- Transaction type distribution -------------------------------------
    if "type" in df.columns:
        plt.figure(figsize=(7, 4))
        df["type"].value_counts().plot(kind="bar", color="darkgreen")
        plt.title("Transaction Types")
        plt.xlabel("Transaction Type")
        plt.ylabel("Count")
        savefig("02_transaction_types.png")

        plt.figure(figsize=(7, 4))
        fraud_by_type = df.groupby("type")[TARGET].mean().sort_values(ascending=False)
        fraud_by_type.plot(kind="bar", color="salmon")
        plt.title("Fraud Rate by Transaction Type")
        plt.ylabel("Fraud Rate")
        savefig("03_fraud_rate_by_type.png")
        print(f"\nFraud rate by type:\n{fraud_by_type}")

    # --- Amount distribution -------------------------------------------------
    if "amount" in df.columns:
        print(f"\nAmount summary:\n{df['amount'].describe()}")

        plt.figure(figsize=(7, 4))
        sns.histplot(np.log1p(df["amount"]), bins=100, kde=True, color="green")
        plt.title("Transaction Amount Distribution (log scale)")
        plt.xlabel("Log(Amount + 1)")
        savefig("04_amount_distribution.png")

        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df[df["amount"] < 50_000], x=TARGET, y="amount")
        plt.title("Amount vs isFraud (transactions under 50k)")
        savefig("05_amount_vs_fraud.png")

    # --- Frauds over time ----------------------------------------------------
    if "step" in df.columns:
        frauds_per_step = df[df[TARGET] == 1]["step"].value_counts().sort_index()
        plt.figure(figsize=(8, 4))
        plt.plot(frauds_per_step.index, frauds_per_step.values, color="crimson")
        plt.xlabel("Step (time)")
        plt.ylabel("Number of Frauds")
        plt.title("Frauds Over Time")
        plt.grid(True)
        savefig("06_frauds_over_time.png")

    # --- Correlation heatmap (numeric features only) -------------------------
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] > 1:
        plt.figure(figsize=(8, 6))
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
        plt.title("Correlation Heatmap (numeric features)")
        savefig("07_correlation_heatmap.png")


# --------------------------------------------------------------------------- #
# 3. Feature engineering
# --------------------------------------------------------------------------- #
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    section("3. FEATURE ENGINEERING")
    df = df.copy()

    if {"oldbalanceOrg", "newbalanceOrig"}.issubset(df.columns):
        df["balanceDiffOrig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    if {"newbalanceDest", "oldbalanceDest"}.issubset(df.columns):
        df["balanceDiffDest"] = df["newbalanceDest"] - df["oldbalanceDest"]

    # Flags: did the balances behave the way they "should" for a transfer?
    if "balanceDiffOrig" in df.columns:
        df["origBalanceMismatch"] = (df["balanceDiffOrig"] < 0).astype(int)
    if "balanceDiffDest" in df.columns:
        df["destBalanceMismatch"] = (df["balanceDiffDest"] < 0).astype(int)

    # Encode categorical transaction type
    if "type" in df.columns:
        le = LabelEncoder()
        df["type_encoded"] = le.fit_transform(df["type"])
        print(f"Encoded 'type' classes: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Drop high-cardinality identifier columns that don't generalize
    drop_cols = [c for c in ["nameOrig", "nameDest", "type"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    print(f"Final feature columns: {[c for c in df.columns if c != TARGET]}")
    return df


# --------------------------------------------------------------------------- #
# 4. Sampling strategies (training data only!)
# --------------------------------------------------------------------------- #
def apply_sampling(X_train: pd.DataFrame, y_train: pd.Series, strategy: str):
    section(f"4. SAMPLING — strategy = '{strategy}'")
    print(f"Before sampling: {y_train.value_counts().to_dict()}")

    if strategy == "none":
        X_res, y_res = X_train, y_train

    elif strategy == "undersample":
        df_train = pd.concat([X_train, y_train], axis=1)
        fraud = df_train[df_train[TARGET] == 1]
        legit = df_train[df_train[TARGET] == 0]
        legit_down = resample(legit, replace=False, n_samples=len(fraud), random_state=RANDOM_STATE)
        df_bal = pd.concat([fraud, legit_down])
        X_res, y_res = df_bal.drop(columns=[TARGET]), df_bal[TARGET]

    elif strategy == "oversample":
        df_train = pd.concat([X_train, y_train], axis=1)
        fraud = df_train[df_train[TARGET] == 1]
        legit = df_train[df_train[TARGET] == 0]
        fraud_up = resample(fraud, replace=True, n_samples=len(legit), random_state=RANDOM_STATE)
        df_bal = pd.concat([legit, fraud_up])
        X_res, y_res = df_bal.drop(columns=[TARGET]), df_bal[TARGET]

    elif strategy == "smote":
        if not IMBLEARN_AVAILABLE:
            raise ImportError(
                "imbalanced-learn is not installed. Run `pip install imbalanced-learn` "
                "or choose --sampling none/undersample/oversample."
            )
        smote = SMOTE(random_state=RANDOM_STATE)
        X_res, y_res = smote.fit_resample(X_train, y_train)

    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")

    print(f"After sampling:  {pd.Series(y_res).value_counts().to_dict()}")
    return X_res, y_res


# --------------------------------------------------------------------------- #
# 5. Model training
# --------------------------------------------------------------------------- #
def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train) -> Optional["XGBClassifier"]:
    if not XGBOOST_AVAILABLE:
        print("  xgboost is not installed — skipping XGBoost model "
              "(run `pip install xgboost` to enable it).")
        return None

    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / max(pos, 1)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


# --------------------------------------------------------------------------- #
# 6. Evaluation
# --------------------------------------------------------------------------- #
@dataclass
class EvalResult:
    name: str
    y_true: np.ndarray
    y_proba: np.ndarray
    threshold: float = 0.5

    @property
    def y_pred(self) -> np.ndarray:
        return (self.y_proba >= self.threshold).astype(int)

    def metrics(self) -> dict:
        return {
            "precision": precision_score(self.y_true, self.y_pred, zero_division=0),
            "recall": recall_score(self.y_true, self.y_pred, zero_division=0),
            "f1": f1_score(self.y_true, self.y_pred, zero_division=0),
            "roc_auc": roc_auc_score(self.y_true, self.y_proba),
            "pr_auc": average_precision_score(self.y_true, self.y_proba),
        }


def evaluate_model(name: str, model, X_test, y_test) -> EvalResult:
    section(f"5. EVALUATION — {name}")
    y_proba = model.predict_proba(X_test)[:, 1]
    result = EvalResult(name=name, y_true=y_test.values, y_proba=y_proba)

    m = result.metrics()
    print(f"Precision : {m['precision']:.4f}")
    print(f"Recall    : {m['recall']:.4f}")
    print(f"F1-score  : {m['f1']:.4f}")
    print(f"ROC-AUC   : {m['roc_auc']:.4f}")
    print(f"PR-AUC    : {m['pr_auc']:.4f}")
    print("\nClassification report (threshold=0.5):")
    print(classification_report(result.y_true, result.y_pred, digits=4))

    # Confusion matrix
    cm = confusion_matrix(result.y_true, result.y_pred)
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    plt.title(f"Confusion Matrix — {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    savefig(f"confusion_matrix_{name.replace(' ', '_').lower()}.png")

    return result


def plot_roc_pr_curves(results: list[EvalResult]) -> None:
    section("6. ROC & PRECISION-RECALL CURVES")

    plt.figure(figsize=(6, 5))
    for r in results:
        fpr, tpr, _ = roc_curve(r.y_true, r.y_proba)
        plt.plot(fpr, tpr, label=f"{r.name} (AUC={roc_auc_score(r.y_true, r.y_proba):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    savefig("08_roc_curve.png")

    plt.figure(figsize=(6, 5))
    for r in results:
        prec, rec, _ = precision_recall_curve(r.y_true, r.y_proba)
        ap = average_precision_score(r.y_true, r.y_proba)
        plt.plot(rec, prec, label=f"{r.name} (PR-AUC={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    savefig("09_precision_recall_curve.png")


# --------------------------------------------------------------------------- #
# 7. Threshold sweep / business decision framing
# --------------------------------------------------------------------------- #
def threshold_business_report(result: EvalResult, fraud_cost: float, review_cost: float) -> pd.DataFrame:
    """
    Build a table that translates precision/recall tradeoffs into a rough
    business cost model so a non-technical stakeholder can pick a threshold.

    fraud_cost   : average money lost when a fraud is MISSED (false negative)
    review_cost  : average analyst/ops cost of manually reviewing ONE
                   transaction flagged as fraud (false positive + true positive)
    """
    section(f"7. THRESHOLD SWEEP & BUSINESS COST — {result.name}")

    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    rows = []
    for t in thresholds:
        y_pred = (result.y_proba >= t).astype(int)
        cm = confusion_matrix(result.y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        flagged = tp + fp

        # Business cost: missed frauds cost `fraud_cost` each, every flagged
        # transaction (TP or FP) costs `review_cost` in analyst time.
        total_cost = fn * fraud_cost + flagged * review_cost

        rows.append({
            "threshold": t, "precision": round(precision, 3), "recall": round(recall, 3),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "flagged_for_review": flagged, "estimated_cost": round(total_cost, 2),
        })

    report = pd.DataFrame(rows)
    best_row = report.loc[report["estimated_cost"].idxmin()]
    print(report.to_string(index=False))
    print(f"\n>>> Lowest estimated business cost at threshold = {best_row['threshold']} "
          f"(precision={best_row['precision']}, recall={best_row['recall']}, "
          f"estimated_cost={best_row['estimated_cost']})")

    plt.figure(figsize=(7, 4))
    plt.plot(report["threshold"], report["estimated_cost"], marker="o", color="purple")
    plt.axvline(best_row["threshold"], color="red", linestyle="--",
                label=f"min-cost threshold = {best_row['threshold']}")
    plt.xlabel("Decision Threshold")
    plt.ylabel(f"Estimated Cost  (fraud_cost={fraud_cost}, review_cost={review_cost})")
    plt.title(f"Business Cost vs Threshold — {result.name}")
    plt.legend()
    savefig(f"10_business_cost_{result.name.replace(' ', '_').lower()}.png")

    out_path = REPORTS_DIR / f"threshold_report_{result.name.replace(' ', '_').lower()}.csv"
    report.to_csv(out_path, index=False)
    print(f"  [saved table]  {out_path.relative_to(ROOT_DIR)}")

    return report


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Credit Card / Online Payments Fraud Detection pipeline")
    p.add_argument("--data", type=str, default=str(ROOT_DIR / "data" / "AIML_Dataset.csv"),
                    help="Path to the fraud dataset CSV.")
    p.add_argument("--sampling", type=str, default="smote",
                    choices=["none", "undersample", "oversample", "smote"],
                    help="Resampling strategy applied to the TRAINING split only.")
    p.add_argument("--model", type=str, default="both",
                    choices=["rf", "xgb", "both"],
                    help="Which model(s) to train.")
    p.add_argument("--test-size", type=float, default=0.2, help="Held-out test set fraction.")
    p.add_argument("--fraud-cost", type=float, default=500.0,
                    help="Estimated $ cost of a missed fraud (false negative), for the business report.")
    p.add_argument("--review-cost", type=float, default=5.0,
                    help="Estimated $ cost of manually reviewing one flagged transaction.")
    p.add_argument("--sample-frac", type=float, default=1.0,
                    help="Optionally subsample the dataset (0-1] for quick local runs.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()

    if not Path(args.data).exists():
        raise FileNotFoundError(
            f"Could not find dataset at '{args.data}'.\n"
            f"Download the PaySim-style fraud dataset and place it under 'data/', "
            f"or pass a custom path with --data."
        )

    df = load_data(args.data)

    if 0 < args.sample_frac < 1.0:
        df = df.sample(frac=args.sample_frac, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"Subsampled dataset to {len(df):,} rows (sample_frac={args.sample_frac})")

    run_eda(df)
    df_feat = engineer_features(df)

    X = df_feat.drop(columns=[TARGET])
    y = df_feat[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

    X_train_res, y_train_res = apply_sampling(X_train, y_train, args.sampling)

    results = []

    if args.model in ("rf", "both"):
        section("5. TRAINING — Random Forest")
        rf = train_random_forest(X_train_res, y_train_res)
        joblib.dump(rf, MODELS_DIR / "random_forest.joblib")
        results.append(evaluate_model("Random Forest", rf, X_test, y_test))

    if args.model in ("xgb", "both"):
        section("5. TRAINING — XGBoost")
        xgb = train_xgboost(X_train_res, y_train_res)
        if xgb is not None:
            joblib.dump(xgb, MODELS_DIR / "xgboost.joblib")
            results.append(evaluate_model("XGBoost", xgb, X_test, y_test))

    if not results:
        raise RuntimeError("No model was trained. Check --model and your installed packages.")

    plot_roc_pr_curves(results)

    business_reports = {}
    for r in results:
        business_reports[r.name] = threshold_business_report(r, args.fraud_cost, args.review_cost)

    # --- Final summary -------------------------------------------------------
    section("SUMMARY")
    summary = {r.name: r.metrics() for r in results}
    for name, m in summary.items():
        print(f"{name:>14}: " + ", ".join(f"{k}={v:.4f}" for k, v in m.items()))

    with open(REPORTS_DIR / "summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[saved] {REPORTS_DIR / 'summary_metrics.json'}")
    print(f"\nDone in {time.time() - start:.1f}s. See the 'outputs/' folder for figures, "
          f"trained models and reports.")


if __name__ == "__main__":
    main()
