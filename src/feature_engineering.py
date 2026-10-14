
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"


def save_features(features, name):
    """Save applicant-level features and validate unique applicant IDs."""
    if not features["SK_ID_CURR"].is_unique:
        raise ValueError(f"{name} contains duplicate applicant IDs")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / f"{name}.csv"
    features.to_csv(output_path, index=False)

    print(
        f"{name}: {len(features):,} applicants, "
        f"{features.shape[1] - 1} features -> {output_path.name}"
    )


def aggregate_bureau_data():
    bureau = pd.read_csv(DATA_DIR / "bureau.csv")
    balance = pd.read_csv(DATA_DIR / "bureau_balance.csv")

    # Encode monthly statuses as numeric indicators.
    status_dummies = pd.get_dummies(
        balance["STATUS"], prefix="BB_STATUS", dtype="int8"
    )
    balance = pd.concat(
        [balance[["SK_ID_BUREAU", "MONTHS_BALANCE"]], status_dummies],
        axis=1,
    )

    balance_agg = balance.groupby("SK_ID_BUREAU").agg(
        {"MONTHS_BALANCE": ["count", "min", "max"],
         **{col: "mean" for col in status_dummies.columns}}
    )
    balance_agg.columns = [
        f"BB_{col}_{stat}".upper()
        for col, stat in balance_agg.columns
    ]
    balance_agg = balance_agg.reset_index()

    bureau = bureau.merge(
        balance_agg,
        on="SK_ID_BUREAU",
        how="left",
        validate="one_to_one",
    )

    numeric_cols = bureau.select_dtypes(include="number").columns.tolist()
    numeric_cols = [
        c for c in numeric_cols
        if c not in {"SK_ID_CURR", "SK_ID_BUREAU"}
    ]

    features = bureau.groupby("SK_ID_CURR")[numeric_cols].agg(
        ["mean", "max", "min", "sum"]
    )
    features.columns = [
        f"BUREAU_{col}_{stat}".upper()
        for col, stat in features.columns
    ]
    return features.reset_index()


def aggregate_previous_applications():
    df = pd.read_csv(DATA_DIR / "previous_application.csv")

    features = df.groupby("SK_ID_CURR").agg(
        PREV_APP_COUNT=("SK_ID_PREV", "count"),
        PREV_APP_CREDIT_MEAN=("AMT_CREDIT", "mean"),
        PREV_APP_CREDIT_MAX=("AMT_CREDIT", "max"),
        PREV_APP_APPLICATION_MEAN=("AMT_APPLICATION", "mean"),
        PREV_APP_APPLICATION_MAX=("AMT_APPLICATION", "max"),
        PREV_APP_ANNUITY_MEAN=("AMT_ANNUITY", "mean"),
        PREV_APP_GOODS_PRICE_MEAN=("AMT_GOODS_PRICE", "mean"),
    )

    statuses = pd.crosstab(df["SK_ID_CURR"], df["NAME_CONTRACT_STATUS"])
    for status in ["Approved", "Refused"]:
        if status in statuses.columns:
            features[f"PREV_APP_{status.upper()}_COUNT"] = statuses[status]

    return features.reset_index()


def aggregate_pos_cash():
    df = pd.read_csv(DATA_DIR / "POS_CASH_balance.csv")

    features = df.groupby("SK_ID_CURR").agg(
        POS_RECORD_COUNT=("SK_ID_PREV", "count"),
        POS_MONTHS_MEAN=("MONTHS_BALANCE", "mean"),
        POS_MONTHS_MIN=("MONTHS_BALANCE", "min"),
        POS_MONTHS_MAX=("MONTHS_BALANCE", "max"),
        POS_SK_DPD_MEAN=("SK_DPD", "mean"),
        POS_SK_DPD_MAX=("SK_DPD", "max"),
        POS_SK_DPD_DEF_MEAN=("SK_DPD_DEF", "mean"),
        POS_SK_DPD_DEF_MAX=("SK_DPD_DEF", "max"),
    )
    return features.reset_index()


def aggregate_installments():
    df = pd.read_csv(DATA_DIR / "installments_payments.csv")

    # Positive delay means payment was made after the due date.
    df["PAYMENT_DELAY"] = (
        df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]
    )
    df["PAYMENT_SHORTFALL"] = (
        df["AMT_INSTALMENT"] - df["AMT_PAYMENT"]
    )

    features = df.groupby("SK_ID_CURR").agg(
        INSTALLMENT_COUNT=("SK_ID_PREV", "count"),
        PAYMENT_DELAY_MEAN=("PAYMENT_DELAY", "mean"),
        PAYMENT_DELAY_MAX=("PAYMENT_DELAY", "max"),
        PAYMENT_SHORTFALL_MEAN=("PAYMENT_SHORTFALL", "mean"),
        PAYMENT_AMOUNT_MEAN=("AMT_PAYMENT", "mean"),
        INSTALLMENT_AMOUNT_MEAN=("AMT_INSTALMENT", "mean"),
    )
    return features.reset_index()


def aggregate_credit_card():
    df = pd.read_csv(DATA_DIR / "credit_card_balance.csv")

    # Summarize credit-card activity per applicant.
    features = df.groupby("SK_ID_CURR").agg(
        CC_RECORD_COUNT=("SK_ID_PREV", "count"),
        CC_BALANCE_MEAN=("AMT_BALANCE", "mean"),
        CC_BALANCE_MAX=("AMT_BALANCE", "max"),
        CC_CREDIT_LIMIT_MEAN=("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
        CC_DRAWINGS_MEAN=("AMT_DRAWINGS_CURRENT", "mean"),
        CC_PAYMENT_TOTAL_MEAN=("AMT_PAYMENT_TOTAL_CURRENT", "mean"),
        CC_DPD_MEAN=("SK_DPD", "mean"),
        CC_DPD_MAX=("SK_DPD", "max"),
    )
    return features.reset_index()


def main():
    aggregations = [
        ("bureau_features", aggregate_bureau_data),
        ("previous_application_features", aggregate_previous_applications),
        ("pos_cash_features", aggregate_pos_cash),
        ("installment_features", aggregate_installments),
        ("credit_card_features", aggregate_credit_card),
    ]

    for name, function in aggregations:
        print(f"\nAggregating {name}...")
        features = function()
        save_features(features, name)
        del features

    print("\nAll auxiliary feature tables saved successfully.")


if __name__ == "__main__":
    main()


