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

def get_default_rate_by_category(
        df: pd.DataFrame,
        target_column: str,
        max_n: int = 10
) -> pd.DataFrame:
    """ Return default rate by category for all low-cardinality categorical features """
    categorical_cols = (
        df
        .select_dtypes(
            exclude = [
                'int64',
                'float64'
            ]
        )
        .columns
        .drop(target_column, errors = 'ignore')
    )

    results: list[pd.DataFrame] = []

    for col in categorical_cols:
        
        if df[col].nunique(dropna = True) <= max_n:

            res = (
                df
                .groupby(col, dropna = False)
                .agg(
                    customer_count = (target_column, 'size'),
                    default_count = (target_column, 'sum'),
                    default_rate = (target_column, 'mean')
                )
                .assign(
                    default_rate_percentage = lambda x: 100 * x['default_rate']
                )
                .reset_index()
                .rename(
                    columns = {
                        col: 'category_value'
                    }
                )
            )

            res.insert(
                0,
                'feature',
                col
            )

            res['default_rate'] = res['default_rate'].round(4)
            res['default_rate_percentage'] = res['default_rate_percentage'].round(2)

            results.append(res)
    
    return (
        pd.concat(results, ignore_index = True)
        .sort_values(
            by = ['feature', 'default_rate'],
            ascending = [True, False]
        )
        .reset_index(drop = True)
    ) if results else pd.DataFrame()

def get_default_rate_by_numeric_bin(
        df: pd.DataFrame,
        target_column: str,
        n_bins: int = 5,
        strategy: str = 'quantile'
) -> pd.DataFrame:
    """ Return default rate by numetic bins for all numeric features """
    # sanity check
    if strategy not in ['uniform', 'quantile']:
        raise ValueError(f'Provided {strategy} is not a valid strategy. Please choose either uniform or quantile for strategy parameter!')
    
    # derive numeric columns
    numeric_cols = (
        df
        .select_dtypes(
            include = [
                'int64',
                'float64'
            ]
        )
        .columns
        .drop(target_column, errors = 'ignore')
    )

    df_temp = df[[target_column]].copy()

    skipped_cols: list[str] = []

    for col in numeric_cols:

        temp_bin_column = f'{col}_bin'

        series = (
            df[col]
            .replace([np.inf, -np.inf], np.nan)
        )

        valid_series = series.dropna()

        if valid_series.empty:
            skipped_cols.append(col)
            continue

        if valid_series.nunique(dropna = True) <= 1:
            skipped_cols.append(col)
            continue

        if strategy == 'quantile':
            df_temp[temp_bin_column] = pd.qcut(
                df[col],
                q = n_bins,
                duplicates = 'drop'
            )
        
        elif strategy == 'uniform':
            df_temp[temp_bin_column] = pd.cut(
                series,
                bins = n_bins,
                include_lowest = True,
                duplicates = 'drop'
            )
    
    results = get_default_rate_by_category(
        df = df_temp,
        target_column = target_column,
        max_n = n_bins
    )

    return results

def get_outlier_summary(
        df: pd.DataFrame,
        lower_quantile: float = 0.05,
        upper_quantile: float = 0.95
) -> pd.DataFrame:
    """ Return outlier summary using lower and upper quantiles """
    # derive numeric dataframe
    df_numeric = (
        df
        .select_dtypes(
            include = [
                'int64',
                'float64'
            ]
        )
    )

    results = []

    for col in df_numeric.columns:

        series = df_numeric[col].dropna()

        if series.empty:
            continue

        # define lower- and upper-bounds
        lb = series.quantile(q = lower_quantile)
        ub = series.quantile(q = upper_quantile)

        outlier_mask = (
            (series < lb) | (series > ub)
        )
        outlier_count = int(outlier_mask.sum())

        results.append({
            'column': col,
            'lower_quantile': lower_quantile,
            'upper_quantile': upper_quantile,
            'lower_bound': lb,
            'upper_bound': ub,
            'outlier_count': outlier_count,
            'outlier_rate': outlier_count / df_numeric[col].shape[0],
            'min_value': series.min(),
            'max_value': series.max()
        })
    
    return pd.DataFrame(results).sort_values(
        'outlier_rate',
        ascending = False
    ).reset_index(drop = True)

def get_highly_correlated_features(
        df: pd.DataFrame,
        target_column: str,
        threshold: float = 0.80
) -> pd.DataFrame:
    """ Return highly correlated numeric feature pairs """
    # derive numeric features excluding the target variable
    df_numeric_target_excluded = (
        df
        .select_dtypes(
            include = [
                'int64',
                'float64'
            ]
        )
        .drop(
            columns = [target_column],
            errors = 'ignore'
        )
    )

    # correlation matrix in absolute terms
    corr_matrix = (
        df_numeric_target_excluded
        .corr()
        .abs()
    )

    # keeping upper triangle to avoid duplicate correlated pairs
    upper_triangle = corr_matrix.where(
        np.triu(
            np.ones(corr_matrix.shape),
            k = 1
        ).astype(bool)
    )

    # correlated pairs based on absolute correlation matrix
    correlated_pairs = (
        upper_triangle
        .stack()
        .reset_index()
    )

    correlated_pairs.columns = [
        'feature_1',
        'feature_2',
        'correlation'
    ]

    # filter correlated pairs rxceeding the threshold for linear dependency, and then sort them in non-increasing order
    correlated_pairs = (
        correlated_pairs
        .query('correlation >= @threshold')
        .sort_values(
            'correlation', 
            ascending = False
        )
        .reset_index(drop = True)
    )

    return correlated_pairs

def get_numeric_target_correlation(
        df: pd.DataFrame,
        target_column: str
) -> pd.DataFrame:
    """ Return absolute correlation of numeric features with target variable """
    # sanity checks
    if target_column not in df.columns:
        raise ValueError(f'{target_column} not found in the dataset!')
    
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        raise TypeError(f'{target_column} must be numeric!')

    # derive numeric features excluding the target variable
    numeric_cols_target_excluded = (
        df
        .select_dtypes(
            include = [
                'int64',
                'float64'
            ]
        )
        .drop(
            columns = [target_column],
            errors = 'ignore'
        )
        .columns
    )

    valid_cols: list[str] = []

    for col in numeric_cols_target_excluded:

        series = df[col].replace([np.inf, -np.inf], np.nan)

        if series.dropna().empty:
            continue

        if series.nunique(dropna = True) <= 1:
            continue

        valid_cols.append(col)
    
    if not valid_cols:
        return pd.DataFrame()

    # target correlation
    target_corr = (
        df[valid_cols]
        .replace([np.inf, -np.inf], np.nan)
        .corrwith(df[target_column])
        .abs()
        .sort_values(ascending = False)
        .reset_index()
    )

    target_corr.columns = [
        'feature',
        'abs_target_correlation'
    ]

    return target_corr

def get_highly_correlated_features_with_target_contribution(
        df: pd.DataFrame,
        target_column: str,
        threshold: float = 0.80
) -> pd.DataFrame:
    """ Return highly correlated feature pairs with target correlation comparison """
    # highly linearly correlated pairs
    high_corr_pairs = get_highly_correlated_features(
        df = df,
        target_column = target_column,
        threshold = threshold
    )

    # target correlation for all feature set
    target_corr = get_numeric_target_correlation(
        df = df,
        target_column = target_column
    )

    target_corr_dict = target_corr.set_index('feature')['abs_target_correlation']

    # feature with target correlations
    high_corr_pairs['feature_1_target_corr'] = (
        high_corr_pairs['feature_1']
        .map(target_corr_dict)
    )

    high_corr_pairs['feature_2_target_corr'] = (
        high_corr_pairs['feature_2']
        .map(target_corr_dict)
    )

    high_corr_pairs['suggested_feature_to_drop'] = np.where(
        high_corr_pairs['feature_1_target_corr'] >= high_corr_pairs['feature_2_target_corr'],
        high_corr_pairs['feature_2'],
        high_corr_pairs['feature_1']
    )

    return high_corr_pairs

# visualizers
def plot_missing_values(
        df: pd.DataFrame,
        top_n: int = 10
) -> None:
    """ Plot missing-value rates by column """
    missing_rate = (
        df
        .isna()
        .mean()
        .sort_values(ascending = False)
    )

    missing_rate = missing_rate[missing_rate > 0].head(top_n)

    # sanity check
    if missing_rate.empty:
        print(('No missing values found!'))
        return None
    
    plt.figure(figsize = (8, max(4, len(missing_rate) * 0.4)))

    missing_rate.sort_values().plot(kind = 'barh')
    plt.title('Missing values by column', fontweight = 'bold')
    plt.xlabel('Missing rate')
    plt.ylabel('Column')
    plt.tight_layout()
    plt.show()

def plot_target_distribution(
        df: pd.DataFrame,
        target_column: str
) -> None:
    """ Plot target distribution for supervised classification """
    # sanity check
    if target_column not in df.columns:
        raise ValueError(f'{target_column} not found in the dataset!')
    
    target_counts = (
        df[target_column]
        .value_counts(dropna = False)
        .sort_index()
    )

    target_rates = (
        df[target_column]
        .value_counts(
            normalize = True,
            dropna = False
        )
        .sort_index()
    )

    plt.figure(figsize = (8, 5))
    target_counts.plot(kind = 'bar')
    plt.title('Target Distribution', fontweight = 'bold')
    plt.xlabel(target_column)
    plt.ylabel('Count')
    plt.tight_layout()

    for index, value in enumerate(target_counts):

        rate = target_rates.iloc[index]
        plt.text(
            index,
            value,
            f'{rate:.1%}',
            ha = 'center',
            va = 'bottom'
        )
    
    plt.show()

def plot_numeric_boxplots(
        df: pd.DataFrame,
        columns: list[str]
) -> None:
    """ Plot boxplots for selected numeric columns to detect outliers """
    # sanity check
    for col in columns:
        # sanity checks
        if col not in df.columns:
            print(f'Skipping {col}: not found')
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f'Skipping {col}: not numeric column')
        
        plt.figure(figsize = (7, 3))
        sns.boxplot(x = df[col])
        plt.title(f'Boxplot of {col}')
        plt.xlabel(col)
        plt.tight_layout()

        plt.show()
    
def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """ Plot correlation heatmap for numetic features """
    # derive numeric dataframe
    df_numeric = (
        df
        .select_dtypes(
            include = [
                'int64',
                'float64'
            ]
        )
    )

    # Pearson's correlation matrix
    corr_matrix = df_numeric.corr()

    plt.figure(figsize = (12, 9))
    sns.heatmap(
        corr_matrix,
        cmap = 'coolwarm',
        center = 0,
        square = True
    )
    plt.title('Correlation heatmap')
    plt.tight_layout()

    plt.show()



