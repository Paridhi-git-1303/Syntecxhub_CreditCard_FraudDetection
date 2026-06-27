# Credit Card / Online Payments Fraud Detection

End-to-end machine learning project for detecting fraudulent transactions in
a highly imbalanced payments dataset (PaySim-style schema: `step`, `type`,
`amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`,
`newbalanceDest`, `isFraud`, `isFlaggedFraud`).

Built to satisfy the project brief:

- [x] Explore the imbalanced fraud dataset, EDA and visualize class imbalance
- [x] Apply sampling approaches (undersampling / oversampling / SMOTE)
- [x] Train a Random Forest / XGBoost model and evaluate with precision,
      recall, ROC-AUC
- [x] Discuss tradeoffs (precision vs recall) and present business decision
      thresholds

## Project structure

```
fraud-detection-project/
├── data/
│   └── README.md              # where to get the dataset
├── notebooks/
│   └── fraud_detection_analysis.ipynb   # exploratory, narrative version
├── src/
│   └── fraud_detection.py     # the full pipeline as a runnable script
├── outputs/
│   ├── figures/                # EDA plots, ROC/PR curves, confusion matrices
│   ├── models/                  # saved .joblib models
│   └── reports/                 # threshold sweep CSVs + summary_metrics.json
├── requirements.txt
└── README.md
```

## Quickstart

```bash
# 1. Create an environment and install dependencies
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Get the dataset (see data/README.md) and place it in data/
#    e.g. data/AIML_Dataset.csv

# 3. Run the full pipeline
python src/fraud_detection.py --data data/AIML_Dataset.csv --sampling smote --model both
```

All figures, trained models, and reports are written to `outputs/`.

### CLI options

| flag             | default                  | description                                            |
|------------------|---------------------------|---------------------------------------------------------|
| `--data`         | `data/AIML_Dataset.csv`   | path to the dataset CSV                                  |
| `--sampling`     | `smote`                   | `none` \| `undersample` \| `oversample` \| `smote`        |
| `--model`        | `both`                    | `rf` \| `xgb` \| `both`                                   |
| `--test-size`    | `0.2`                      | held-out test fraction                                    |
| `--fraud-cost`   | `500`                      | est. $ lost per missed fraud (business report)            |
| `--review-cost`  | `5`                        | est. $ cost per manually reviewed flagged transaction      |
| `--sample-frac`  | `1.0`                      | optionally subsample the data for a quick local run        |

Example — quick local smoke test on 10% of the data with undersampling:

```bash
python src/fraud_detection.py --sample-frac 0.1 --sampling undersample --model rf
```

## What the pipeline does

1. **EDA** — class balance, transaction-type breakdown, fraud rate by type,
   amount distribution (log scale), amount vs. fraud boxplot, frauds over
   time, and a correlation heatmap. All saved as PNGs in
   `outputs/figures/`.
2. **Feature engineering** — derives `balanceDiffOrig` / `balanceDiffDest`
   and flags transactions whose balances moved in a suspicious direction;
   label-encodes `type`; drops high-cardinality ID columns
   (`nameOrig`, `nameDest`) that don't generalize.
3. **Stratified train/test split** — keeps the same fraud ratio in both
   splits.
4. **Sampling (training data only)** — random undersampling, random
   oversampling, or SMOTE. The test set is **never** resampled, since doing
   so would leak information and overstate performance.
5. **Models** — Random Forest (`class_weight="balanced_subsample"`) and
   XGBoost (`scale_pos_weight` tuned to the class ratio). XGBoost is
   optional — the script degrades gracefully to Random Forest only if
   `xgboost` isn't installed.
6. **Evaluation** — precision, recall, F1, ROC-AUC, PR-AUC, confusion
   matrices, and combined ROC / precision-recall curves across models.
7. **Business decision thresholds** — sweeps the classification threshold
   from 0.05 to 0.95 and converts each one into an estimated dollar cost
   using two inputs you control: the cost of a **missed fraud** and the
   cost of a **manual review**. The threshold that minimizes total
   estimated cost is highlighted — this is the number a fraud-ops team
   would actually plug into production, not the default 0.5.

## Precision vs. recall — why it matters here

Fraud datasets are extremely imbalanced (often <1% positive class), so a
model that maximizes plain accuracy can simply predict "not fraud" for
everything and still score >99%. That's why this project leans on
**precision, recall, ROC-AUC, and PR-AUC** instead:

- **Recall** (catch rate) matters because every missed fraud is a direct
  financial loss.
- **Precision** matters because every false alarm costs analyst time and
  can frustrate legitimate customers.
- Pushing the decision threshold down raises recall but tanks precision,
  and vice versa — there is no threshold that maximizes both
  simultaneously. The `outputs/reports/threshold_report_*.csv` files (and
  the matching cost-vs-threshold plot) make that tradeoff concrete in
  dollar terms so the choice becomes a business decision rather than a
  default left at 0.5.

## Notes

- `outputs/figures`, `outputs/models`, and `outputs/reports` are kept
  empty (with `.gitkeep`) in version control — they're regenerated by
  running the script.
- The raw dataset CSV is excluded from git via `.gitignore`; see
  `data/README.md` for how to obtain it.
