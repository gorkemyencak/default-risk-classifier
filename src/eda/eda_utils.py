import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

def compare_supervised_vs_unsupervised_columns(
        supervised_df: pd.DataFrame,
        unsupervised_df: pd.DataFrame
) -> dict:
    """ Return column structure of supervised and unsupervised datasets """
    supervised_cols = set(supervised_df.columns)
    unsupervised_cols = set(unsupervised_df.columns)

    return {
        'columns_only_in_supervised_dataset': sorted(supervised_cols - unsupervised_cols),
        'columns_only_in_unsupervised_dataset': sorted(unsupervised_cols - supervised_cols),
        'common_columns': sorted(supervised_cols | unsupervised_cols)
    }

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

def get_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """ Return descriptive statistics for numeric columns """
    df_numeric = df.select_dtypes(
        include = [
            'int64',
            'float64'
        ]
    )

    if df_numeric.empty:
        return pd.DataFrame()
    
    return df_numeric.describe(
        percentiles = [
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99
        ]
    ).T

def get_negative_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Return negative values in numeric columns 
    
    If non-negative numeric column, then exclude them from the summary report
    """
    df_numeric = df.select_dtypes(
        include = [
            'int64',
            'float64'
        ]
    )

    negative_summary = []
    
    for col in df_numeric.columns:
        negative_count = int((df_numeric[col] < 0).sum())
        negative_rate = negative_count / df.shape[0]

        if negative_count > 0:

            negative_summary.append({
                'column': col,
                'negative_count': negative_count,
                'negative_rate': negative_rate,
                'min_value': df_numeric[col].min()
            })
    
    return pd.DataFrame(negative_summary).sort_values(
        'negative_count',
        ascending = False
    ) if negative_summary else pd.DataFrame()


def get_categorical_summary(
        df: pd.DataFrame,
        top_n: int = 10
) -> pd.DataFrame:
    """ Return count and ratio for categorical columns with top_n categorical values """
    categorical_cols = df.select_dtypes(
        exclude = [
            'int64',
            'float64'
        ]
    ).columns

    categorical_summaries: list[pd.DataFrame] = []

    for col in categorical_cols:
        row = (
            df[col]
            .value_counts(dropna = False)
            .head(top_n)
            .reset_index()
        )

        row.columns = ['value', 'count']
        row.insert(
            0,
            'column',
            str(col)
        )
        row['ratio'] = (
            row['count'] 
            / df.shape[0]
        ).round(4)

        categorical_summaries.append(row)
    
    if not categorical_summaries:
        return pd.DataFrame(
            columns = [
                'column',
                'value',
                'count',
                'ratio'
            ]
        )

    return pd.concat(
        categorical_summaries, 
        ignore_index = True
    )




