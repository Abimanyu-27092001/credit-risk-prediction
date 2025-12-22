import pandas as pd
import urllib.request


def load_german_credit_data() -> pd.DataFrame:
    """
    Downloads and parses the UCI German Credit dataset.
    Returns a pandas DataFrame with proper column names.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

    raw = urllib.request.urlopen(url).read().decode("latin-1")
    rows = [r.strip() for r in raw.splitlines() if r.strip()]

    cols = [
        "Status_checking_account",
        "Duration_months",
        "Credit_history",
        "Purpose",
        "Credit_amount",
        "Savings_account_bonds",
        "Present_employment_since",
        "Installment_rate",
        "Personal_status_sex",
        "Other_debtors_guarantors",
        "Present_residence_since",
        "Property",
        "Age_years",
        "Other_installment_plans",
        "Housing",
        "Number_existing_credits",
        "Job",
        "Number_people_liable",
        "Telephone",
        "Foreign_worker",
        "Class",
    ]

    data = [r.split() for r in rows]
    return pd.DataFrame(data, columns=cols)


def clean_german_credit_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and types the German Credit dataset.
    - Casts numeric columns
    - Creates binary default target
    - Drops original class column
    """
    df = df.copy()

    numeric_columns = [
        "Duration_months",
        "Credit_amount",
        "Installment_rate",
        "Present_residence_since",
        "Age_years",
        "Number_existing_credits",
        "Number_people_liable",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Target: 1 = default (bad), 0 = good
    df["default"] = (df["Class"] == "2").astype(int)

    df.drop(columns=["Class"], inplace=True)

    return df
