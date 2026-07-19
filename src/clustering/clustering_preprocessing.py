import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """  
    FrequencyEncoder encode high-cardinality categorical features, since we cannot apply Target Encoding on unsupervised dataset

    In frequency encoding, each category is replaced by its relative frequency in the training data. Unknown categories are
    mapped to 0
    """
    def __init__(
            self,
            missing_label: str = 'Unknown'
    ) -> None:
        # attributes
        self.missing_label = missing_label
    
    # helper function
    @staticmethod
    def _to_dataframe(X) -> pd.DataFrame:
        """ Convert input into dataframe """
        if isinstance(X, pd.DataFrame):
            return X.copy()
        
        return pd.DataFrame(X)
    
    def fit(
            self,
            X,
            y = None
    ):
        """ Fit FrequencyEncoder to learn category frequencies for each column """
        X_df = self._to_dataframe(X = X)

        self.frequency_maps_ = {}

        for col in X_df.columns:

            series = (
                X_df[col]
                .astype('object')
                .fillna(self.missing_label)
            )

            self.frequency_maps_[col] = (
                series
                .value_counts(normalize = True)
                .to_dict()
            )
        
        self.feature_names_in_ = (
            X_df
            .columns
            .tolist()
        )

        return self
    
    def transform(
            self,
            X
    ):
        """ Apply learned frequency encoding """
        X_df = self._to_dataframe(X = X)

        encoded_columns = []

        for col in X_df.columns:

            frequency_map = self.frequency_maps_.get(col, {})

            encoded_col = (
                X_df[col]
                .astype('object')
                .fillna(self.missing_label)
                .map(frequency_map)
                .fillna(0.0)
                .astype(float)
            )

            encoded_columns.append(encoded_col.to_numpy())
        
        return np.vstack(encoded_columns).T
    

class ClusteringPreprocessingBuilder:
    """ 
    Build preprocessing pipeline for unsupervised clustering

    Key functions:
        - identify numeric, low-cardinality and high-cardinality categorical features
        - apply median imputation and robust scaling to numeric features
        - apply one-hot encoding to low-cardinality categorical features
        - apply frequency encoding to high-cardinality categorical features
        - drop target column if it is accidentally present 
        - keep all preprocessing leakage-safe inside sklearn pipelines    
    """
    def __init__(
            self,
            low_cardinality_threshold: int = 10,
            high_cardinality_categorical_features: list[str] | None = None,
            columns_to_exclude: list[str] | None = None
    ) -> None:
        # attributes
        self.low_cardinality_threshold = low_cardinality_threshold
        self.high_cardinality_categorical_features = (
            high_cardinality_categorical_features or ['State']
        )
        self.columns_to_exclude = (
            columns_to_exclude or ['Target']
        )
    
    # helper functions
    def clean_clustering_input(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """ Remove target column if it is mistakenly present in the unsupervised dataset """
        df = df.copy()

        cols_to_drop = [
            col
            for col in self.columns_to_exclude
            if col in df.columns
        ]

        return (
            df
            .drop(
                columns = cols_to_drop,
                errors = 'ignore'
            )
        )
    
    def _get_numeric_features(
            self,
            X: pd.DataFrame
    ) -> list[str]:
        """ Return numeric features based on dtype """
        return (
            X
            .select_dtypes(
                include = [
                    'number',
                    'bool'
                ]
            )
            .columns
            .tolist()
        )
    
    def _get_categorical_features(
            self,
            X: pd.DataFrame
    ) -> list[str]:
        """ Return categorical features based on dtype """
        return (
            X.
            select_dtypes(
                exclude = [
                    'number',
                    'bool'
                ]
            )
            .columns
            .tolist()
        )
    
    def _split_categorical_features(
            self,
            X: pd.DataFrame
    ) -> tuple[list[str], list[str]]:
        """ Split categorical features into low- and high-cardinality groups """
        # extract categorical features
        categorical_features = self._get_categorical_features(X = X)

        # initialize low- and high-cardinality features
        low_cardinality_features: list[str] = []
        high_cardinality_features: list[str] = []

        for col in categorical_features:

            # number of unique values in the categorical feature
            n_unique = X[col].nunique(dropna = True)

            # if high_cardinality feature is explicitly stated
            if col in self.high_cardinality_categorical_features:
                high_cardinality_features.append(col)
            # check the number of unique values to decide whether low- or high-cardinality column
            elif n_unique >= self.low_cardinality_threshold:
                high_cardinality_features.append(col)
            else:
                low_cardinality_features.append(col)
        
        return low_cardinality_features, high_cardinality_features
    
    # clustering preprocessing builder engine
    def get_feature_groups(
            self,
            X: pd.DataFrame
    ) -> dict[str, list[str]]:
        """ Return feature groups used by the clustering preprocessing pipeline """
        # get numeric features
        numeric_features = self._get_numeric_features(X = X)

        # get categorical features
        low_cardinality_features, high_cardinality_features = self._split_categorical_features(X = X)

        return {
            'numeric_features': numeric_features,
            'low_cardinality_categorical_features': low_cardinality_features,
            'high_cardinality_categorical_features': high_cardinality_features
        }
    
    def build_clustering_preprocessor(
            self,
            X: pd.DataFrame
    ) -> ColumnTransformer:
        """ 
        Build clustering preprocessing ColumnTransformer 
        
        ColumnTransformer allows different transformations for different column groups
        """
        # get feature groups
        feature_groups = self.get_feature_groups(X = X)

        numeric_features = feature_groups['numeric_features']
        low_cardinality_features = feature_groups['low_cardinality_categorical_features']
        high_cardinality_features = feature_groups['high_cardinality_categorical_features']

        # clustering preprocessing pipeline
        numeric_pipeline = Pipeline(
            steps = [
                ('imputer', SimpleImputer(strategy = 'median')),
                ('scaler', RobustScaler())
            ]
        )

        low_cardinality_pipeline = Pipeline(
            steps = [
                ('imputer', SimpleImputer(strategy = 'most_frequent')),
                (
                    'onehot',
                    OneHotEncoder(
                        handle_unknown = 'ignore',
                        sparse_output = False
                    )
                )
            ]
        )

        high_cardinality_pipeline = Pipeline(
            steps = [
                ('frequency_encoder', FrequencyEncoder(missing_label = 'Unknown')),
                ('scaler', RobustScaler())
            ]
        )

        # column transformation pipeline
        transformers = []

        if numeric_features:
            transformers.append(
                (
                    'numeric',
                    numeric_pipeline,
                    numeric_features
                )
            )
        
        if low_cardinality_features:
            transformers.append(
                (
                    'low_cardinality_categorical',
                    low_cardinality_pipeline,
                    low_cardinality_features
                )
            )
        
        if high_cardinality_features:
            transformers.append(
                (
                    'high_cardinality_categorical',
                    high_cardinality_pipeline,
                    high_cardinality_features
                )
            )
        
        column_transformer = ColumnTransformer(
            transformers = transformers,
            remainder = 'drop',
            verbose_feature_names_out = True
        )

        return column_transformer