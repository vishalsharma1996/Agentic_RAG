import numpy as np
import pandas as pd
import re

def is_repeating_row(row):
    values = [str(v).strip() for v in row if np.array(pd.notna(v)).any() and str(v).strip()]
    if not len(values):
        return False
    return len(set(values)) == 1


def is_null(val):
    if pd.isna(val):
        return True
    if isinstance(val, str) and val.strip() in {"", "-", "--"}:
        return True
    return False

def merge_duplicate_columns(df):
    df = df.copy()
    df = df.fillna("").astype(str)
    col_counts = df.columns.value_counts()
    duplicate_names = col_counts[col_counts > 1].index.tolist()
    for col_name in duplicate_names:
        # Get column indices instead of names
        col_indices = [i for i, c in enumerate(df.columns) if c == col_name]

        def merge_row(row):
            values = [row.iloc[i].strip() for i in col_indices]
            non_null_values = [v for v in values if v not in {"", "-", "--"}]

            if not non_null_values:
                return ""

            if len(non_null_values) == 1:
                return non_null_values[0]

            unique_vals = list(dict.fromkeys(non_null_values))
            return " ".join(unique_vals)

        df[col_name] = df.apply(merge_row, axis=1)

        # Drop duplicate columns except first occurrence
        to_drop = set(col_indices[1:])
        df = df.iloc[:, [i for i in range(df.shape[1]) if i not in to_drop]]

    return df
