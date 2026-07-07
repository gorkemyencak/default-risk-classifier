import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

def get_dataframe_overview(df: pd.DataFrame) -> pd.DataFrame:
    """ Return high-level information for each column """
    df_overview = pd.DataFrame({
        'dtype': df.dtypes.astype(str),
        'missing_count': df.isna().sum(),
        'missing_rate': df.isna().mean(),
        'unique_count': df.nunique(dropna = True),
        'unique_rate': df.nunique(dropna = True) / len(df)
    })

    df_overview[['missing_rate', 'unique_rate']] = (
        df_overview[['missing_rate', 'unique_rate']].round(4)
    )

    return df_overview.sort_values(
        by = [
            'missing_rate',
            'unique_count'
        ],
        ascending = [
            False,
            False
        ]
    )

def get_duplicate_summary(
        df: pd.DataFrame,
        id_column: str | None = None
) -> dict:
    """ Return duplicate information at row level and ID level """
    duplicate_dict = {
        'n_rows': len(df),
        'duplicate_rows': int(df.duplicated().sum())
    }

    # append id_column duplicated row information to the dictionary if it is passed
    if id_column is not None and id_column in df.columns:
        duplicate_dict[f'duplicate_{id_column}'] = int(df[id_column].duplicated().sum())
    
    return duplicate_dict


def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """ Return dataframe columns with missing values only, sorted by missing ratio in a descending order """
    missing = (
        df
        .isna()
        .sum()
        .to_frame(name = 'missing_count')
        .assign(missing_ratio = lambda x: x['missing_count'] / df.shape[0])
        .query('missing_count > 0')
        .sort_values('missing_ratio', ascending = False)
    )

    missing[['missing_ratio']] = missing[['missing_ratio']].round(4)

    return missing

def get_constant_columns(df: pd.DataFrame) -> list[str]:
    """ Return columns wth only one unique non-missing value """
    constant_cols = [col for col in df.columns if df[col].nunique(dropna = True) <= 1]

    return constant_cols

def get_target_summary(
        df: pd.DataFrame,
        target_column: str
) -> pd.DataFrame:
    """ Return target count, ratio and percentage """
    target_summary = (
        df[target_column]
        .value_counts(dropna = False)
        .to_frame(name = 'count')
        .assign(
            ratio = lambda x: x['count'] / x['count'].sum(),
            percentage = lambda x: 100 * x['ratio']
        )
    )

    return target_summary