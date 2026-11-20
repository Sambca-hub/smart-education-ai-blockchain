import pandas as pd
import numpy as np


def load_dataset(file):
    """Load and validate an educational CSV dataset."""
    df = pd.read_csv(file, sep=None, engine="python")

    if df.empty:
        raise ValueError("The uploaded dataset is empty.")
    if len(df.columns) < 2:
        raise ValueError("Dataset needs at least two columns.")

    df.columns = df.columns.astype(str).str.strip()

    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names detected.")

    return df


def validate_dataset(df):
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "column_names": list(df.columns),
    }


def clean_dataset(df, remove_duplicates=True):
    cleaned = df.copy()
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)

    if remove_duplicates:
        cleaned = cleaned.drop_duplicates()

    return cleaned


def get_column_information(df):
    return pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isnull().sum().values,
        "Unique Values": df.nunique().values,
    })
