# Dataset

This project expects a PaySim-style mobile-money transactions dataset with
(at least) the following columns:

| column            | description                                              |
|--------------------|-----------------------------------------------------------|
| `step`             | unit of time (1 step = 1 hour)                            |
| `type`             | transaction type: PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN |
| `amount`           | transaction amount                                         |
| `nameOrig`         | originating customer ID                                    |
| `oldbalanceOrg`    | originator's balance before the transaction                |
| `newbalanceOrig`   | originator's balance after the transaction                 |
| `nameDest`         | destination customer/merchant ID                           |
| `oldbalanceDest`   | recipient's balance before the transaction                 |
| `newbalanceDest`   | recipient's balance after the transaction                  |
| `isFraud`          | target label — 1 if the transaction is fraudulent          |
| `isFlaggedFraud`   | flag raised by the existing rule-based system               |

The notebook this project was built from used a file named
`AIML Dataset.csv`. This is the same schema as the popular Kaggle
**"Online Payments Fraud Detection Dataset"** / **PaySim1** dataset.

## How to get it

1. Download the dataset from Kaggle (search "Online Payments Fraud
   Detection Dataset" or "PaySim1").
2. Place the CSV in this `data/` folder.
3. Rename it to `AIML_Dataset.csv`, or point the script at it directly:

   ```bash
   python src/fraud_detection.py --data "data/AIML Dataset.csv"
   ```

The raw CSV is intentionally excluded from version control (see
`.gitignore`) — it's a few hundred MB and isn't something you want to
push to GitHub.
