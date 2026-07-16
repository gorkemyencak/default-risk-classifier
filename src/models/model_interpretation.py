import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

from config import METRICS_DIR

class SupervisedModelInterpreter:
    """
    Supervised model interpretation class for supervised classification pipelines

    This class supports:
        - Logistic Regression coefficient interpretation
        - Tree-based native feature importance
        - Gradient Boosting native feature importance
        - Model-agnostic permutation importance
    """
    def __init__(
            self,
            metrics_dir: Path = METRICS_DIR,
            random_state: int = 2
    ) -> None:
        # attributes
        self.metrics_dir = metrics_dir
        self.random_state = random_state
    
    ### helper functions
    @staticmethod
    def _get_classifier(
        pipeline: Pipeline
    ):
        """ Return fitted classifier from full model pipeline """
        return pipeline.named_steps['classifier']
    
    @staticmethod
    def _get_column_transformer(
        pipeline: Pipeline
    ):
        """ Return fitted ColumnTransformer from nested preprocessing pipeline """
        preprocessor = pipeline.named_steps['preprocessor']

        if 'column_transformer' not in preprocessor.named_steps:
            raise KeyError("column_transformer not found inside pipeline['preprocessor']")
        
        return preprocessor.named_steps['column_transformer']
    
    @staticmethod
    def _clean_transformed_feature_name(
        feature_name: str
    ) -> str:
        """ 
        Clean sklearn-generated transformed feature names 
        
        e.g.
            - numeric__Loan To Value                                  -> Loan To Value
            - low_cardinality_categorical__FICO Rating_Exceptional    -> FICO Rating_Exceptional 
            - high_cardinality_categorical__State                     -> State
        """
        # define prefixes
        prefixes_to_drop = [
            'numeric__',
            'low_cardinality_categorical__',
            'high_cardinality_categorical__'
        ]

        clean_feature_name: str = ''

        for prefix in prefixes_to_drop:
            clean_feature_name = feature_name.replace(prefix, '')
        
        return clean_feature_name
    
    def _get_transformed_feature_names(
            self,
            pipeline: Pipeline,
            clean_names: bool = True
    ) -> list[str]:
        """ Return feature names after model preprocessing """
        # column transformer steps
        column_transformer = self._get_column_transformer(pipeline = pipeline)

        # sklearn-generated transformed feature names inside column transformer pipeline
        feature_names = (
            column_transformer
            .get_feature_names_out()
            .tolist()
        )

        # drop sklearn-generated prefixes from transformed feature names
        if clean_names:
            feature_names = [
                self._clean_transformed_feature_name(feature_name = feature)
                for feature in feature_names
            ]

        return feature_names
    
    ### logistic regression interpretation
    def get_logistic_regression_coefficients(
            self,
            pipeline: Pipeline,
            model_name: str = 'logistic_regression_baseline'
    ) -> pd.DataFrame:
        """ Return logistic regression coefficients """
        # get classifier
        classifier = self._get_classifier(pipeline = pipeline)

        # sanity check
        if not isinstance(classifier, LogisticRegression):
            raise TypeError('The classifier is not LogisticRegression!')
        
        # get cleaned feature names
        feature_names = self._get_transformed_feature_names(
            pipeline = pipeline,
            clean_names = True
        )

        # logistic regression coefficients
        coefficients = (
            classifier
            .coef_
            .ravel()
        )

        coefficients_df = pd.DataFrame({
            'model': model_name,
            'feature': feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients),
            'odds_ratio': np.exp(coefficients)
        })

        coefficients_df['effect_direction'] = np.where(
            coefficients_df['coefficient'] > 0,
            'increases_deafult_risk',
            'decreases_default_risk'
        )

        sorted_coefficient_df = (
            coefficients_df
            .sort_values(
                'abs_coefficient',
                ascending = False
            )
            .reset_index(drop = True)
        )

        return sorted_coefficient_df
    
    ### tree/ensemble interpretation
    def get_native_feature_importance(
            self,
            pipeline: Pipeline,
            model_name: str
    ) -> pd.DataFrame:
        """ 
        Return native feature importance for tree-based classifiers 
        
        This method works for estimators exposing feature_importances_, including:
            - DecisionTreeClassifier
            - RandomForestClassifier
            - GradientBoostingClassifier
        """
        # get classifier
        classifier = self._get_classifier(pipeline = pipeline)

        # sanity check
        if not hasattr(classifier, 'feature_importances_'):
            raise TypeError(f'{model_name} does not expose feature_importances_')
        
        # get cleaned feature names
        feature_names = self._get_transformed_feature_names(
            pipeline = pipeline,
            clean_names = True
        )

        # tree-based classifier - feature importances
        feature_importances = classifier.feature_importances_

        feature_importances_df = pd.DataFrame({
            'model': model_name,
            'feature': feature_names,
            'importance': feature_importances
        })

        sorted_feature_importances_df = (
            feature_importances_df
            .sort_values(
                'importance',
                ascending = False
            )
            .reset_index(drop = True)
        )

        return sorted_feature_importances_df
    
    def get_all_native_feature_importance(
            self,
            trained_pipelines: dict[str, Pipeline]
    ) -> pd.DataFrame:
        """ Return native feature importances for all compatible classifier models """
        results: list[pd.DataFrame] = []

        for model_name, pipeline in trained_pipelines.items():

            # get classifier
            classifier = self._get_classifier(pipeline = pipeline)

            # sanity check
            if hasattr(classifier, 'feature_importances_'):
                sorted_feature_importances_df = self.get_native_feature_importance(
                    pipeline = pipeline,
                    model_name = model_name
                )
            
                results.append(sorted_feature_importances_df)
        
        if not results:
            return pd.DataFrame()
        
        return pd.concat(
            results,
            ignore_index = True
        )
    
    # model-agnostic interpretation
    def get_permutation_importance(
            self,
            pipeline: Pipeline,
            X: pd.DataFrame,
            y: pd.Series,
            model_name: str,
            scoring: str = 'average_precision',
            n_repeats: int = 10,
            n_jobs: int = -1
    ) -> pd.DataFrame:
        """ 
        Return permutation importance on raw input features 
        
        Permutation importance interpretation is a business-friendly raw feature importance that evaluates the full fitted 
        pipeline directly on the original X columns
        """
        # permutation importance
        result = permutation_importance(
            estimator = pipeline,
            X = X,
            y = y,
            scoring = scoring,
            n_repeats = n_repeats,
            n_jobs = n_jobs,
            random_state = self.random_state
        )

        permutation_importance_df = pd.DataFrame({
            'model': model_name,
            'feature': X.columns,
            'importance_mean': result['importances_mean'],
            'importance_std': result['importances_std'],
            'scoring': scoring
        })

        sorted_permutation_importance_df = (
            permutation_importance_df
            .sort_values(
                'importance_mean',
                ascending = False
            )
            .reset_index(drop = True)
        )

        return sorted_permutation_importance_df
    
    def get_all_permutation_importances(
            self,
            trained_pipelines: dict[str, Pipeline],
            X: pd.DataFrame,
            y: pd.Series,
            scoring: str = 'average_precision',
            n_repeats: int = 10,
            n_jobs: int = -1
    ) -> pd.DataFrame:
        """ Return permutation importances for all trained model pipelines """
        results: list[pd.DataFrame] = []

        for model_name, pipeline in trained_pipelines.items():

            sorted_permutation_importance_df = self.get_permutation_importance(
                pipeline = pipeline,
                X = X,
                y = y,
                model_name = model_name,
                scoring = scoring,
                n_repeats = n_repeats,
                n_jobs = n_jobs
            )

            results.append(sorted_permutation_importance_df)
        
        if not results:
            pd.DataFrame()
        
        return pd.concat(
            results,
            ignore_index = True
        )
    
    # business summary reports
    @staticmethod
    def get_top_features(
        importance_df: pd.DataFrame,
        value_column: str,
        top_n: int = 10
    ) -> pd.DataFrame:
        """ Return top_n features from a feature importance dataframe """
        return (
            importance_df
            .sort_values(
                value_column,
                ascending = False
            )
            .head(top_n)
            .reset_index(drop = True)
        )
    
    @staticmethod
    def add_business_interpretation_notes(
        df: pd.DataFrame,
        feature_column: str = 'feature'
    ) -> pd.DataFrame:
        """ Business interpretation notes based on feature names that has to be reviewed before used in the final reporting """
        df = df.copy()

        def map_note(feature: str) -> str:

            feature_lower = feature.lower()

            if 'fico' in feature_lower:
                return 'Credit bureau quality / creditworthiness signal'
            if 'bureau excluded' in feature_lower:
                return 'Customer may have insufficient or excluded bureau history'
            if 'loan to value' in feature_lower:
                return 'Loan exposure relative to asset value'
            if 'overdue' in feature_lower:
                return 'Historical repayment stress / delinquency signal'
            if 'delinquency' in feature_lower:
                return 'Recent repayment stress signal'
            if 'inquiries' in feature_lower:
                return 'Recent credit-seeking behavior'
            if 'balance' in feature_lower:
                return 'Outstanding balance / leverage signal'
            if 'disbursed' in feature_lower:
                return 'Exposure size signal'
            if 'instalment' in feature_lower or 'installment' in feature_lower:
                return 'Repayment burden signal'
            if 'active_account' in feature_lower:
                return 'Current credit activity signal'
            if 'state' in feature_lower:
                return 'Geographic / portfolio mix signal'
            if 'employment' in feature_lower:
                return 'Income stability / employment profile signal'
            if 'age' in feature_lower:
                return 'Customer maturity / lifecycle signal'
            
            return 'Requires manual business review'
        
        df['business_interpretation'] = df[feature_column].map(map_note)

        return df
    
    def save_interpretation_table(
            self,
            df: pd.DataFrame,
            file_name: str
    ) -> None:
        """ Save business interpretation dataframe into metics directory """
        self.metrics_dir.mkdir(
            parents = True,
            exist_ok = True
        )

        df.to_csv(
            self.metrics_dir / file_name,
            index = False
        )