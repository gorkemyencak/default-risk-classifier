import pandas as pd
import numpy as np

from pathlib import Path

from config import METRICS_DIR

class ClusterInterpreter:
    """
    ClusterInterpreter class interprets customer clusters from a business and credit-risk perspective, the goal is to translate 
    statistical clusters into portfolio-risk segments

    Key functions:
        * attach selected cluster labels to the original unsupervised dataframe
        * calculate cluster size summaries
        * calculate numeric cluster profiles
        * compare cluster means against global averages
        * calculate categorical composition by cluster
        * identify top distinguishing features per cluster
        * create preliminary business labels for clusters
        * save interpretation outputs    
    """
    def __init__(
            self,
            metrics_dir: Path = METRICS_DIR,
            cluster_column: str = 'cluster'
    ) -> None:
        # attributes
        self.metrics_dir = metrics_dir
        self.cluster_column = cluster_column
    
    # helper functions - data preparation
    @staticmethod
    def _prepare_labels(
        labels,
        index: pd.Index,
        cluster_column: str = 'cluster'
    ) -> pd.Series:
        """ Converting raw cluster labels into a clean pandas Series aligned with dataframe index """
        # create label Series matching with dataframe index
        label_series = pd.Series(
            labels,
            index = index,
            name = cluster_column
        )

        # check missing values in label Series
        if label_series.isna().any():
            raise ValueError('cluster labels contain missing values!')
        
        return label_series
    
    def _add_cluster_labels(
            self,
            df: pd.DataFrame,
            labels,
            cluster_column: str | None = None
    ) -> pd.DataFrame:
        """ Return the copy of original dataframe with labels assigned as a new column """
        # ensure cluster_column being deployed
        cluster_column = (
            cluster_column or self.cluster_column
        )

        # copy original unsupervised df
        df_with_clusters = df.copy()

        # generate label Series
        label_series = self._prepare_labels(
            labels = labels,
            index = df.index,
            cluster_column = cluster_column
        )

        # assign label Series into df
        df_with_clusters[cluster_column] = label_series

        return df_with_clusters
    
    @staticmethod
    def _validate_features(
        df: pd.DataFrame,
        features: list[str]
    ) -> list[str]:
        """ Return the list of features that exists in the dataframe """
        return [
            feature
            for feature in features
            if feature in df.columns
        ]
    
    # helper functions - business interpretation
    @staticmethod
    def _add_business_interpretation_notes(
        profile_long: pd.DataFrame,
        feature_column: str = 'feature'
    ):
        """ Business interpretation notes based on feature names that has to be reviewed before used in the final reporting """
        # create a copy of numeric profile long dataframe
        profile = profile_long.copy()

        def map_note(feature: str) -> str:

            feature_lower = feature.lower()

            if 'fico' in feature_lower:
                return 'Credit bureau quality / creditworthiness signal'
            if 'bureau excluded' in feature_lower:
                return 'Customer may have insufficient or excluded bureau history'
            if 'loan to value' in feature_lower or 'ltv' in feature_lower:
                return 'Loan exposure relative to asset value'
            if 'overdue' in feature_lower:
                return 'Historical repayment stress'
            if 'delinquency' in feature_lower:
                return 'Recent repayment stress signal'
            if 'inquir' in feature_lower:
                return 'Recent credit seeking behavior'
            if 'balance' in feature_lower:
                return 'Outstanding balance / leverage signal'
            if 'disbursed' in feature_lower:
                return 'Exposure size signal'
            if 'instalment' in feature_lower or 'installment' in feature_lower:
                return 'Repayment burden signal'
            if 'active account' in feature_lower or 'active_account' in feature_lower:
                return 'Current credit activity signal'
            if 'number of accounts' in feature_lower or 'accountsper' in feature_lower:
                return 'Credit history depth'
            if 'recentlyopened' in feature_lower or 'opened last 6 months' in feature_lower:
                return 'Recent credit activity signal'
            if 'age' in feature_lower:
                return 'Customer maturity / lifecycle signal'
            if 'state' in feature_lower:
                return 'Geographic / portfolio mix signal'
            if 'employment' in feature_lower:
                return 'Income stability / employment profile signal'
            
            return 'Requires manual business review'
        
        profile['business_interpretation'] = profile[feature_column].map(map_note)

        return profile
    
    @staticmethod
    def _map_note_to_cluster_label(
        top_notes: list[str],
        top_features: list[str]
    ) -> str:
        """ Map dominant business notes into preliminary business labels """
        # create the set of business notes as lowercase
        notes = {note.lower() for note in top_notes}

        if 'repayment stress' in notes:
            return 'Repayment stress / delinquency segment'
        if 'credit bureau quality' in notes:
            return 'Credit bureau quality segment'
        if 'loan exposure' in notes:
            return 'High collateral / LTV-driven segment'
        if 'exposure size' in notes or 'outstanding balance' in notes:
            return 'High exposure / balance-driven segment'
        if 'recent credit seeking' in notes or 'recent credit activity' in notes:
            return 'Recent credit activity segment'
        if 'credit history depth' in notes:
            return 'Credit history depth segment'
        if 'credit maturity' in notes:
            return 'Customer lifecycle segment'
        
        return 'General portfolio segment'


    # clustering profiler engine - cluster size
    def get_cluster_size_summary(
            self,
            df_with_clusters: pd.DataFrame,
            cluster_column: str | None = None
    ) -> pd.DataFrame:
        """ Return count and share of customers in each cluster """
        # ensure cluster_column being deployed
        cluster_column = (
            cluster_column or self.cluster_column
        )

        # sanity check for cluster_column
        if cluster_column not in df_with_clusters.columns:
            raise ValueError(f'{cluster_column} not found in the dataframe!')
        
        # cluster size summary with count and share of customers at each portfolio cluster
        cluster_size_summary = (
            df_with_clusters[cluster_column]
            .value_counts()
            .sort_index()
            .to_frame(name = 'customer_count')
            .assign(
                customer_share = lambda x: x['customer_count'] / x['customer_count'].sum(),
                customer_share_pct = lambda x: 100 * x['customer_share']
            )
            .reset_index()
            .rename(
                columns = {
                    'index': cluster_column
                }
            )
        )

        # round the digits for customer_share and customer_share_pct
        cluster_size_summary['customer_share'] = cluster_size_summary['customer_share'].round(4)
        cluster_size_summary['customer_share_pct'] = cluster_size_summary['customer_share_pct'].round(2)

        return cluster_size_summary
    
    # clustering profiler engine - numeric profiles
    def get_numeric_cluster_profile(
            self,
            df_with_clusters: pd.DataFrame,
            numeric_features: list[str],
            cluster_column: str | None = None,
            agg_functions: list[str] | None = None
    ) -> pd.DataFrame:
        """ Return numeric cluster profile using selected numeric features """
        # ensure cluster_column & agg_functions being deployed
        cluster_column = (
            cluster_column or self.cluster_column
        )
        agg_functions = (
            agg_functions or ['mean', 'median']
        )

        # sanity check for cluster_column
        if cluster_column in df_with_clusters.columns:
            raise ValueError(f'{cluster_column} not found in the dataframe!')
        
        # validate numeric features that exist in the dataframe
        numeric_features = self._validate_features(
            df = df_with_clusters,
            features = numeric_features
        )

        # sanity check on numeric feature validation
        if not numeric_features:
            raise ValueError('No valid numeric features provided!')
        
        # numeric profiler
        numeric_profile = (
            df_with_clusters
            .groupby(cluster_column)[numeric_features]
            .agg(agg_functions)
        )

        # flatteninng MultiIndex columns
        numeric_profile.columns = [
            f'{feature}_{agg_stat}'
            for feature, agg_stat in numeric_profile.columns
        ]

        return numeric_profile.reset_index()
    
    def get_numeric_mean_profile(
            self,
            df_with_clusters: pd.DataFrame,
            numeric_features: list[str],
            cluster_column: str | None = None
    ) -> pd.DataFrame:
        """ Return cluster-level numeric means """
        # ensure cluster_column being deployed
        cluster_column = (
            cluster_column or self.cluster_column
        )

        # validate numeric features that exist in the dataframe
        numeric_features = self._validate_features(
            df = df_with_clusters,
            features = numeric_features
        )

        # sanity check on numeric feature validation
        if not numeric_features:
            raise ValueError('No valid numeric features provided!')
        
        # numeric mean profiler
        numeric_mean_profile = (
            df_with_clusters
            .groupby(cluster_column)[numeric_features]
            .mean()
        )

        return numeric_mean_profile.reset_index()
    
    def get_relative_numeric_profile(
            self,
            df_with_clusters: pd.DataFrame,
            numeric_features: list[str],
            cluster_column: str | None = None
    ) -> pd.DataFrame:
        """  
        Return cluster means divided by portfolio averages
            - if value > 1 -> implies that cluster average is above portfolio average
            - if value < 1. -> implies that cluster average is below portfolio average
        """
        # ensure cluster_column being deployed
        cluster_column = (
            cluster_column or self.cluster_column
        )

        # validate numeric features that exist in the dataframe
        numeric_features = self._validate_features(
            df = df_with_clusters,
            features = numeric_features
        )

        # sanity check on numeric feature validation
        if not numeric_features:
            raise ValueError('No valid numeric features provided!')
        
        # cluster-level mean profiler
        cluster_mean_profile = (
            df_with_clusters
            .groupby(cluster_column)[numeric_features]
            .mean()
        )

        # portfolio-level mean profiler
        portfolio_mean_profile = df_with_clusters[numeric_features].mean()

        # relative cluster mean profiler against portfolio mean profiler
        relative_mean_profile = cluster_mean_profile.divide(
            portfolio_mean_profile.replace(
                to_replace = 0,
                value = np.nan
            )
        )

        return (
            relative_mean_profile
            .replace(
                to_replace = [np.inf, -np.inf],
                value = np.nan
            )
            .reset_index()
        )
    
    def get_numeric_cluster_profile_long(
            self,
            df_with_clusters: pd.DataFrame,
            numeric_features: list[str],
            cluster_column: str | None = None            
    ) -> pd.DataFrame:
        """ Return long-format numeric cluster profile with portfolio comparison using selected features """
        # ensure cluster_column being deployed
        cluster_column = (
            cluster_column or self.cluster_column
        )

        # validate numeric features that exist in the dataframe
        numeric_features = self._validate_features(
            df = df_with_clusters,
            features = numeric_features
        )

        # sanity check on numeric feature validation
        if not numeric_features:
            raise ValueError('No valid numeric features provided!')
        
        # cluster-level mean profiler
        cluster_mean_profile = (
            df_with_clusters
            .groupby(cluster_column)[numeric_features]
            .mean()
        )

        # portfolio-level mean profiler
        portfolio_mean_profile = df_with_clusters[numeric_features].mean()

        results: list[dict] = []

        for cluster_label, row in cluster_mean_profile.iterrows():

            for feature in numeric_features:

                # get cluster-level mean of a given feature
                cluster_mean = row[feature]

                # get portfolio mean of a given feature
                portfolio_mean = portfolio_mean_profile[feature]

                # safe division check
                if portfolio_mean == 0 or pd.isna(portfolio_mean):
                    relative_to_portfolio = np.nan
                else:
                    relative_to_portfolio = cluster_mean / portfolio_mean
                
                results.append({
                    cluster_column: cluster_label,
                    'feature': feature,
                    'cluster_mean': cluster_mean,
                    'portfolio_mean': portfolio_mean,
                    'signed_difference': cluster_mean - portfolio_mean,
                    'absolute_difference': abs(cluster_mean - portfolio_mean),
                    'relative_to_portfolio': relative_to_portfolio
                })
        
        return pd.DataFrame(results)
    
    # clustering profiler engine - categorical profiles
    def get_categorical_cluster_profile(
            self,
            df_with_clusters: pd.DataFrame,
            categorical_features: list[str],
            cluster_column: str | None = None,
            top_n: int = 5
    ) -> pd.DataFrame:
        """ Return categorical cluster profile """
        # ensure cluster_column being deployed
        cluster_column = (
            cluster_column or self.cluster_column
        )

        # validate categorical features that exist in the dataframe
        categorical_features = self._validate_features(
            df = df_with_clusters,
            features = categorical_features
        )

        # sanity check on categorical feature validation
        if not categorical_features:
            raise ValueError(f'No valid categorical features provided!')
        
        results: list[pd.DataFrame] = []

        for feature in categorical_features:

            # profile table for a given categorical feature
            profile_table = (
                df_with_clusters
                .groupby(
                    [cluster_column, feature],
                    dropna = False
                )
                .size()
                .to_frame(name = 'count')
                .reset_index()
            )

            # cluster-level total counts
            profile_table['cluster_total'] = (
                profile_table
                .groupby(cluster_column)['count']
                .transform('sum')
            )

            # witthin-cluster share of features
            profile_table['cluster_share'] = (
                profile_table['count'] / profile_table['cluster_total']
            )

            # portfolio-level total counts
            profile_table['portfolio_count'] = (
                profile_table[feature]
                .map(
                    df_with_clusters[feature]
                    .value_counts(dropna = False)
                )
            )

            # overall portfolio-share of each feature
            profile_table['portfolio_share'] = (
                profile_table['portfolio_count'] / len(df_with_clusters)
            )

            # compute if a feature is over- or under-represented in a cluster compared with the portfolio, i.e. >1 -> over-represented, <1 -> under-represented
            profile_table['share_lift_vs_portfolio'] = (
                profile_table['cluster_share'] 
                / profile_table['portfolio_share'].replace(0, np.nan)
            )

            # assign feature name to the profile table
            profile_table.insert(0, 'feature', feature)

            # top_n categorical features per cluster based on within-cluster share
            profile_table = (
                profile_table
                .sort_values(
                    by = [
                        cluster_column, 
                        'cluster_share'
                    ],
                    ascending = [
                        True,
                        False
                    ]
                )
                .groupby(cluster_column)
                .head(top_n)
                .reset_index(drop = True)
            )

            # rename feature column
            profile_table = (
                profile_table
                .rename(
                    columns = {
                        feature: 'category_value'
                    }
                )
            )

            results.append(profile_table)
        
        return pd.concat(
            results,
            ignore_index = True
        )
    
    # clustering profiler engine - top distinguishing cluster-level feature profiles
    def get_top_distinguishing_features_by_cluster(
            self,
            numeric_profile_long: pd.DataFrame,
            top_n: int = 8,
            cluster_column: str | None = None,
            min_relative_distance: float = 0.20
    ) -> pd.DataFrame:
        """
        Return the most distinguishing numeric features for each cluster

        A feature is distinguishing if its relative value is greater than 1 + min_relative_distance or lower than 1 - min_relative_distance
        """
        # ensure cluster_column being deployed
        cluster_column = (
            cluster_column or self.cluster_column
        )

        # create a copy of numeric profile long dataframe
        profile = numeric_profile_long.copy()

        # signed relative difference from portfolio average
        profile['signed_relative_distance_from_portfolio'] = profile['relative_to_portfolio'] - 1

        # absolute relative difference from portfolio average
        profile['abs_relative_distance_from_portfolio'] = profile['signed_relative_distance_from_portfolio'].abs()

        # distinguishing features by cluster
        profile = profile.query('abs_relative_distance_from_portfolio > @min_relative_distance')

        # return if profile table is empty
        if profile.empty:
            return pd.DataFrame()
        
        # top_n distinguishing numeric features per cluster
        top_features = (
            profile
            .sort_values(
                by = [
                    cluster_column,
                    'abs_relative_distance_from_portfolio'
                ],
                ascending = [
                    True,
                    False
                ]
            )
            .groupby(cluster_column)
            .head(top_n)
            .reset_index(drop = True)
        )

        return top_features
    
    # clustering profiler engine - business interpretation methods
    def generate_preliminary_cluster_labels(
            self,
            distinguishing_features: pd.DataFrame,
            cluster_column: str | None = None
    ) -> pd.DataFrame:
        """ Generate preliminary business labels per cluster using the dominant notes from cluster's most distinguished features """
        # ensure cluster_column being deployed
        cluster_column = (
            cluster_column or self.cluster_column
        )

        # return empty dataframe if distinguished features are empty
        if distinguishing_features.empty:
            return pd.DataFrame(
                columns = [
                    cluster_column,
                    'preliminary_cluster_label',
                    'label_rationale'
                ]
            )
        
        # assign business interpretation notes based on features
        profile_table = self._add_business_interpretation_notes(
            profile_long = distinguishing_features,
            feature_column = 'feature'
        )

        results: list[dict] = []

        for cluster_label, cluster_df in profile_table.groupby(cluster_column):

            # top 3 notes per cluster -> dominating business notes
            top_notes = (
                cluster_df['business_interpretation']
                .value_counts()
                .head(3)
                .index
                .tolist()
            )

            # top 5 most distinguishing feature names per cluster
            top_features = (
                cluster_df
                .sort_values(
                    'abs_relative_distance_from_portfolio',
                    ascending = False
                )
                .head(5)['feature']
                .tolist()
            )

            # map into preliminary business labels
            business_label = self._map_note_to_cluster_label(
                top_notes = top_notes,
                top_features = top_features
            )

            # rationale for the generated business label
            rationale = (
                'Dominant notes: '
                + ', '.join(top_notes)
                + '. Top distinguishing features: '
                + ', '.join(top_features)
            )

            results.append({
                cluster_column: cluster_column,
                'preliminary_cluster_label': business_label,
                'label_rationale': rationale
            })
        
        return pd.DataFrame(results)
    
    def apply_manual_cluster_labels(
            self,
            cluster_size_summary: pd.DataFrame,
            manual_labels: dict,
            cluster_column: str | None = None
    ) -> pd.DataFrame:
        """ Attach manually reviewed business labels to cluster size table after business review """
        # ensure cluster_column being deployed
        cluster_column = (
            cluster_column or self.cluster_column
        )

        # create a copy of cluster size profiler
        cluster_size_summary = cluster_size_summary.copy()

        # assign business label
        cluster_size_summary['business_label'] = (
            cluster_size_summary[cluster_column]
            .map(manual_labels)
            .fillna('Unlabeled cluster')
        )

        return cluster_size_summary
    
    # save profiler tables locally
    def save_interpretation_table(
            self,
            interpretation_df: pd.DataFrame,
            file_name: str = 'business_interpretation_table.csv'
    ) -> None:
        """ Save cluster interpretation table locally """
        self.metrics_dir.mkdir(
            parents = True,
            exist_ok = True
        )

        interpretation_df.to_csv(
            self.metrics_dir / file_name,
            index = False
        )