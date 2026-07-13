import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, TargetEncoder

class LoanToValueOutlierImputer(BaseEstimator, TransformerMixin):
    """
    Replace extreme Loan To Value values with the training median. LoanToValueOutlierImputer is fitted only on the training data
    when used inside a sklearn Pipeline that avoids data leakage

    Transformer rule:
        - treat values below lower_quantile as outliers
        - treat values above upper_quantile as outliers
        - replace outliers with training median    
    """
    def __init__(
            self,
            column: str = 'Loan To Value',
            lower_quantile: float = 0.01,
            upper_quantile: float = 0.99
    ) -> None:
        # attributes
        self.column = column
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(
            self,
            X: pd.DataFrame,
            y: pd.Series | None = None
    ):
        """ 
        Fit LoanToValueOutlierImputer to learn:
            - lower bound
            - upper bound
            - median
        using the train data only     
        """
        X = X.copy()

        # sanity check
        if self.column not in X.columns:
            self._lower_bound = None
            self._upper_bound = None
            self._median = None
            
            return self
        
        # LTV serie
        LTV_serie = pd.to_numeric(
            X[self.column],
            errors = 'coerce'
        )

        self._lower_bound = LTV_serie.quantile(q = self.lower_quantile)
        self._upper_bound = LTV_serie.quantile(q = self.upper_quantile)
        self._median = LTV_serie.median()

        return self
    
    def transform(
            self,
            X: pd.DataFrame
    ) -> pd.DataFrame:
        """ Transform applies the learned rule extracted from the training data only """
        X = X.copy()

        # sanity checks
        if self.column not in X.columns:
            return X
        
        if self._lower_bound is None or self._upper_bound is None:
            return X
        
        # LTV serie
        LTV_serie = pd.to_numeric(
            X[self.column],
            errors = 'coerce'
        )

        # outliers
        outlier_mask = (
            (LTV_serie > self._upper_bound)
            | 
            (LTV_serie < self._lower_bound)
        )

        # replace with training median
        X.loc[outlier_mask, self.column] = self._median

        return X
    

class ModelPreprocessingBuilder:
    """
    Build preprocessing objects for supervised classification models

    Key functions:
        - identify numeric, low-cardinality categorical, and high-cardinality categorical features
        - apply median imputation and robust scaling to numeric features
        - apply one-hot encoding to low-cardinaility categorical features
        - apply target encoding to high-cardinality categorical features
        - keep all preprocessing leakage-safe inside sklearn pipelines    
    """
    def __init__(
            self,
            target_column: str = 'Target',
            low_cardinality_threshold: int = 10,
            high_cardinality_categorical_features: list[str] | None = None,
            apply_ltv_outlier_imputation: bool = True,
            ltv_column: str = 'Loan To Value',
            lower_quantile: float = 0.01,
            upper_quantile: float = 0.99,
            random_state: int = 2
    ) -> None:
        # attributes
        self.target_column = target_column
        self.low_cardinality_threshold = low_cardinality_threshold
        self.high_cardinality_categorical_features = (
            high_cardinality_categorical_features or ['State']
        )
        self.apply_ltv_outlier_imputation = apply_ltv_outlier_imputation
        self.ltv_column = ltv_column
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.random_state = random_state
    
    # helpers
    def _get_numeric_features(
            self,
            X: pd.DataFrame
    ) -> list[str]:
        """ Return numeric features based on dtype """
        return X.select_dtypes(
            include = [
                'number',
                'bool'
            ]
        ).columns.tolist()
    
    def _get_categorical_features(
            self,
            X: pd.DataFrame
    ) -> list[str]:
        return X.select_dtypes(
            exclude = [
                'number',
                'bool'
            ]
        ).columns.tolist()
    
    def _split_categorical_features(
            self,
            X: pd.DataFrame
    ) -> tuple[list[str], list[str]]:
        """ 
        Return low-cardinality categorical features that exist in X 
        
        Low-cardinality features should be one-hot encoded, while high-cardinality features are excluded
        from this list, since they will be target encoded

        A categorical feature is considered high-cardinality if:
            - it is explicitly stated in self.high_cardinality_categorical_features, or
            - its number of unique values is greater than self.low_cardinality_threshold
        """
        # extract categorical features
        categorical_features = self._get_categorical_features(X = X)

        # initialize low- and high-cardinality features
        low_cardinality_features: list[str] = []
        high_cardinality_features: list[str] = []

        for col in categorical_features:
            
            # number of unique values inside the categorical feature
            n_unique = X[col].nunique(dropna = True)

            # if high_cardinality feature is explicitly stated
            if col in self.high_cardinality_categorical_features:
                high_cardinality_features.append(col)
            # check number of unique values to decide whether low- or high-cardinality column
            elif n_unique >= self.low_cardinality_threshold:
                high_cardinality_features.append(col)
            else:
                low_cardinality_features.append(col)
        
        return low_cardinality_features, high_cardinality_features
    

    # model preprocessing builder engine
    def split_target(
            self,
            df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """ Split processed supervised dataset into predictors X and target y """
        df = df.copy()

        # sanity check
        if self.target_column not in df.columns:
            raise ValueError(f'{self.target_column} not found in the supervised dataframe')
        
        # split into predictors and target
        X = df.drop(columns = [self.target_column])
        y = df[self.target_column]

        return X, y
    
    def get_feature_groups(
            self,
            X: pd.DataFrame
    ) -> dict[str, list[str]]:
        """ Return feature groups used by the model preprocessing pipeline """
        # get numeric features
        numeric_features = self._get_numeric_features(X = X)

        # get categorical features
        low_cardinality_features, high_cardinality_features = self._split_categorical_features(X = X)

        return {
            'numeric_features': numeric_features,
            'low_cardinality_categorical_features': low_cardinality_features,
            'high_cardinality_categorical_features': high_cardinality_features
        }
    
    # sklearn pipeline methods
    def build_column_transformer(
            self,
            X: pd.DataFrame
    ) -> ColumnTransformer:
        """
        Build ColumnTransformer for model preprocessing

        ColumnTransformer allows different transformations for different column groups
        """
        # get feature groups
        feature_groups = self.get_feature_groups(X = X)

        numeric_features = feature_groups['numeric_features']
        low_cardinality_features = feature_groups['low_cardinality_categorical_features']
        high_cardinality_features = feature_groups['high_cardinality_categorical_features']

        # model preprocessing pipeline
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
                ('imputer', SimpleImputer(strategy = 'most_frequent')),
                (
                    'target_encoder',
                    TargetEncoder(
                        target_type = 'binary',
                        smooth = 'auto',
                        cv = 5,
                        shuffle = True,
                        random_state = self.random_state
                    )
                )
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
    
    def build_model_preprocessing_pipeline(
            self,
            X: pd.DataFrame
    ) -> Pipeline:
        """  
        Build full model preprocessing pipeline method

        LoanToValueOutlierImputer is placed befor the ColumnTransformer to make sure that outlier handling will be 
        treated before column transformation methods
        """
        # initialize model preprocessing pipeline steps
        steps = []

        # check if LoanToValueOutlierImputer is required
        if self.apply_ltv_outlier_imputation:
            steps.append(
                (
                    'ltv_outlier_imputer',
                    LoanToValueOutlierImputer(
                        column = self.ltv_column,
                        lower_quantile = self.lower_quantile,
                        upper_quantile = self.upper_quantile
                    )
                )
            )
        
        steps.append(
            (
                'column_transformer',
                self.build_column_transformer(X = X)
            )
        )

        return Pipeline(steps = steps)