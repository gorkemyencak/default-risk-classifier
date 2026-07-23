import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer

from config import METRICS_DIR

class NumericClusterSeparationAnalyzer:
    """ 
    NumericClusterSeparationAnalyzer class analyzes which numeric features explain separation between clusters 
    
    This class is a pre-interpretation diagnostic layer, and is applied after assinging the cluster labels deploying the clustering models

    The scores represented in this module should be interpreted in response to:
        - which numeric features vary the most across clusters?
        - which features explain the largest share of between-cluster variance?
        - which features should be prioritized in cluster profiling?

    Key metrics:
        * η^{2}: share of total feature variance explained by cluster membership
        * anova-f-score: ANOVA F-statistic using cluster labels as groups -> measures whether cluster means statistically differ
        * max absolute standardized cluster mean: largest standardized distance between any cluster mean an the portfolio average
        * cluster mean min / cluster mean max: minimum and maximum cluster means for the feature
        * cluster wtih min mean / cluster with max mean: clusters where the feature is lowest/highest
    """
    def __init__(
            self,
            metrics_dir: Path = METRICS_DIR
    ) -> None:
        # attributes
        self.metrics_dir = metrics_dir
    
    # helper functions - data preparation
    @staticmethod
    def _prepare_labels(
        labels,
        index: pd.Index
    ) -> pd.Series:
        """ Convert labels into a Series aligned with the dataframe index """
        label_series = pd.Series(
            labels,
            index = index,
            name = 'cluster'
        )

        # check missing labels in the label series
        if label_series.isna().any():
            raise ValueError('Cluster labels contain missing values!')

        return label_series
    
    @staticmethod
    def _get_numeric_features(
        df: pd.DataFrame,
        exclude_columns: list[str] | None = None
    ) -> pd.DataFrame:
        """ Extract numeric and boolean features from dataframe """
        # get excluded columns
        excluded_cols = (
            exclude_columns or []
        )

        # numeric features
        numeric_df = (
            df
            .select_dtypes(
                include = [
                    'number',
                    'bool'
                ]
            )
            .drop(
                columns = excluded_cols,
                errors = 'ignore'
            )
            .replace([np.inf, -np.inf], np.nan)
        )

        return numeric_df
    
    @staticmethod
    def _remove_unusable_features(
        numeric_df: pd.DataFrame
    ) -> pd.DataFrame:
        """ Remove fully missing or constant numeric features """
        usable_features: list[str] = []

        for col in numeric_df.columns:

            series = numeric_df[col]

            # missing series
            if series.notna().sum() == 0:
                continue
            
            # constant series
            if series.nunique(dropna = True) <= 1:
                continue

            usable_features.append(col)
        
        return numeric_df[usable_features]
    
    @staticmethod
    def _impute_numeric_features(
        numeric_df: pd.DataFrame
    ) -> pd.DataFrame:
        """ Median-impute numeric features for statistical calculations """
        # median imputes
        imputer = SimpleImputer(strategy = 'median')

        # fit_transform median imputer
        imputed_array = imputer.fit_transform(numeric_df)

        return pd.DataFrame(
            imputed_array,
            columns = numeric_df.columns,
            index = numeric_df.index
        )
    
    # helper functions - core statistics
    @staticmethod
    def _calculate_eta_squared(
        feature: pd.Series,
        labels: pd.Series
    ) -> float:
        """
        Calculate eta-squared for one feature

        If cluster membership explains a higher feature variation, then we can safely claim that such a feature is a strong cluster-separation feature  

        Formula:
            η^{2} = between_cluster_sum_of_squares / total_sum_of_squares
        """
        # global mean
        global_mean = feature.mean()

        # cluster means and counts
        cluster_means = (
            feature
            .groupby(labels)
            .mean()
        )
        cluster_counts = (
            feature
            .groupby(labels)
            .size()
        )

        # between_cluster_sum_of_squares and total_sum_of_squares
        ss_between = (
            cluster_counts * (cluster_means - global_mean) ** 2
        ).sum()
        ss_total = (
            (feature - global_mean) ** 2
        ).sum()

        if ss_total == 0:
            return np.nan
        
        return float(
            ss_between / ss_total
        )
    
    @staticmethod
    def _calculate_standardized_cluster_mean_spread(
        feature: pd.Series,
        labels: pd.Series
    ) -> dict:
        """ 
        Calculate standardized cluster mean deviation from portfolio average 
        
        Standardized cluster mean deviation simply responds to how different the average values are across clusters, helps to identify the cluster with
        lowest and highest average, and how far the most extreme cluster average is from the global average for a given feature
        """
        # global mean & global std
        global_mean = feature.mean()
        global_std = feature.std()

        # cluster means & min/max cluster means
        cluster_means = (
            feature
            .groupby(labels)
            .mean()
        )
        cluster_mean_min = cluster_means.min()
        cluster_mean_max = cluster_means.max()

        # clusters with min/max means
        cluster_with_min_mean = cluster_means.idxmin()
        cluster_with_max_mean = cluster_means.idxmax()

        # checking whether global standard deviation is usable
        if global_std == 0 or np.isnan(global_std):
            # the feature cannot be standardized 
            max_abs_standardized_cluster_mean = np.nan
        else:
            # calculating the cluster mean in terms of relative std distance below/above the global average
            standardized_cluster_means = (
                (cluster_means - global_mean)
                / global_std
            )

            # maximum standardized cluster mean 
            max_abs_standardized_cluster_mean = (
                standardized_cluster_means
                .abs()
                .max()
            )
        
        return {
            'cluster_mean_min': cluster_mean_min,
            'cluster_mean_max': cluster_mean_max,
            'cluster_mean_range': cluster_mean_max - cluster_mean_min,
            'cluster_with_min_mean': cluster_with_min_mean,
            'cluster_with_max_mean': cluster_with_max_mean,
            'max_abs_standardized_cluster_mean': max_abs_standardized_cluster_mean
        }
    
    # cluster separation engine
    def analyze(
            self,
            df: pd.DataFrame,
            labels,
            exclude_columns: list[str] | None = None
    ) -> pd.DataFrame:
        """ 
        Calculate numeric cluster-separation importance for one clustering solution 
        
        higher f-score -> stronger separation across clusters
        lower p-value -> stronger statistical evidence of different cluster means
        """
        # prepare aligned labels
        labels_series = self._prepare_labels(
            labels = labels,
            index = df.index
        )

        # get numeric features
        numeric_df = self._get_numeric_features(
            df = df,
            exclude_columns = exclude_columns
        )

        # clean numeric features
        numeric_df = self._remove_unusable_features(
            numeric_df = numeric_df
        )

        # sanity check
        if numeric_df.empty:
            raise ValueError('No usable numeric features found!')
        
        # median-imputation for numeric features
        X_imputed = self._impute_numeric_features(
            numeric_df = numeric_df
        )

        # statistical importance of cluster means
        f_scores, p_values = f_classif(
            X = X_imputed,
            y = labels_series
        )

        anova_df = pd.DataFrame({
            'feature': X_imputed.columns,
            'anova_f_score': f_scores,
            'anova_p_value': p_values
        })

        results: list[dict] = []

        for col in X_imputed.columns:

            feature = X_imputed[col]

            # calculate η^{2}
            eta_squared = self._calculate_eta_squared(
                feature = feature,
                labels = labels_series
            )

            # calculate standardized cluster mean spread
            spread_stats = self._calculate_standardized_cluster_mean_spread(
                feature = feature,
                labels = labels_series
            )

            result = {
                'feature': col,
                'eta_squared': eta_squared
            }
            result.update(spread_stats)

            results.append(result)
        
        separation_df = pd.DataFrame(results)

        separation_df = separation_df.merge(
            anova_df,
            on = 'feature',
            how = 'left'
        )

        return (
            separation_df
            .sort_values(
                by = [
                    'eta_squared',
                    'anova_f_score',
                    'max_abs_standardized_cluster_mean'
                ],
                ascending = False
            )
            .reset_index(drop = True)
        )
    
    def analyze_multiple_models(
            self,
            df: pd.DataFrame,
            cluster_labels: pd.DataFrame,
            model_names: list[str] | None = None,
            exclude_columns: list[str] | None = None
    ) -> pd.DataFrame:
        """ 
        Calculate numeric cluster-separation importance for multiple clustering solutions, 
        and return a single combined interpretation table
        """
        # check whether a list of model_names are provided, and analyze all clustering models if model_names are not passed
        if model_names is None:
            model_names = (
                cluster_labels
                .columns
                .tolist()
            )
        
        results: list[pd.DataFrame] = []

        for model_name in model_names:
            # check if user-sprecified model is found in cluster_labels
            if model_name not in cluster_labels.columns:
                raise ValueError(f'{model_name} not found in cluster_labels columns, please train the {model_name} first!')
            
            # compute numeric cluster-separation importance of a given model_name 
            model_result = self.analyze(
                df = df,
                labels = cluster_labels[model_name],
                exclude_columns = exclude_columns
            )

            model_result.insert(0, 'model', model_name)

            results.append(model_result)
        
        return pd.concat(
            results,
            ignore_index = True
        )
    
    # pre-interpretation feature selection
    @staticmethod
    def get_top_features(
        separation_df: pd.DataFrame,
        top_n: int = 15,
        model_name: str | None = None
    ) -> pd.DataFrame:
        """ 
        Return top cluster-separating features filtered by user-specified model_name 
        
        Top features are sorted in the following order:
            1) η^{2} -> Higher eta_squared implies that cluster labels explain more of this feature's variance
            2) Anova F-Score -> Higher anova f-score implies that cluster means are more separated relative to within-cluster variation
            3) Standardized Spread -> implies that at least one cluster mean is far from global average
        """
        df = separation_df.copy()

        # filter the separation_df by model_name
        if model_name is not None:
            df = df.query('model == @model_name') 
        
        return(
            df
            .sort_values(
                by = [
                    'eta_squared',
                    'anova_f_score',
                    'max_abs_standardized_cluster_mean'
                ],
                ascending = False
            )
            .head(top_n)
            .reset_index(drop = True)
        )
    
    @staticmethod
    def select_profile_features(
        separation_df: pd.DataFrame,
        domain_features: list[str] | None = None,
        top_n_data_driven: int = 15,
        model_name: str | None = None
    ) -> list[str]:
        """ Combine domain-driven and data-driven profile features, so that it provides a defensible feature list for cluster profiling """
        # business domain knowledge / expert opinion
        domain_features = (
            domain_features or []
        )

        # statistical cluster-separation importances
        top_features_df = NumericClusterSeparationAnalyzer.get_top_features(
            separation_df = separation_df,
            top_n = top_n_data_driven,
            model_name = model_name
        )

        data_driven_features = top_features_df['feature'].tolist()

        # combine domain-driven & data-driven features while preserving the order of numeric features
        combined_features = list(
            dict.fromkeys(
                domain_features + data_driven_features
            )
        )

        return combined_features
    
    def save_separation_table(
        self,
        separation_df: pd.DataFrame,
        file_name: str = 'numeric_cluster_separation_importance.csv'
    ) -> None:
        """ Save cluster-separation importance table locally """
        self.metrics_dir.mkdir(
            parents = True,
            exist_ok = True
        )

        separation_df.to_csv(
            self.metrics_dir / file_name,
            index = False
        )