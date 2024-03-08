from typing import List

import numpy as np
import pandas as pd


def continuous_selector(df: pd.DataFrame, ordinal_cutoff: int = 10) -> List[str]:
    columns = df.select_dtypes(include=np.number).columns.tolist()
    return [c for c in columns if df[c].nunique() > ordinal_cutoff]


def ordinal_selector(df: pd.DataFrame, ordinal_cutoff: int = 10) -> List[str]:
    columns = df.select_dtypes(include=np.number).columns.tolist()
    return [c for c in columns if df[c].nunique() <= ordinal_cutoff]


def nominal_selector(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=object).columns.tolist()
