import pandas as pd
import joblib

from pathlib import Path

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from config import TARGET_COLUMN, MODELS_DIR, METRICS_DIR

from src.data.data_loader import load_supervised_features

from src.models.model_preprocessing import ModelPreprocessingBuilder
from src.models.model_evaluation import SupervisedEvaluator

class SupervisedModelTrainer:
    """
    Train and evaluate supervised default-risk classification models

    SupervisedModelTrainer class trains logistic regression as baseline and several benchmark classes using the same
    model preprocessing pipeline

    Key functions:
        - load processed supervised feature dataset
        - split predictors and target
        - perform stratified train-test split
        - build sklearn pipelines
        - train baseline and becnchmark classifiers
        - evaluate classification performance
        - save fitted model pipelines and metrics    
    """
    def __init__(
            self,
            target_column: str = TARGET_COLUMN,
            test_size: float = 0.20,
            random_state: int = 2,
            threshold: float = 0.50,
            model_dir: Path = MODELS_DIR,
            metrics_dir: Path = METRICS_DIR,
            preprocessing_builder: ModelPreprocessingBuilder | None = None,
            evaluator: SupervisedEvaluator | None = None            
    ) -> None:
        # attributes
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.threshold = threshold
        self.model_dir = model_dir
        self.metrics_dir = metrics_dir
        self.preprocessing_builder = (
            preprocessing_builder 
            or ModelPreprocessingBuilder(
                target_column = self.target_column,
                random_state = self.random_state
            )
        )
        self.evaluator = (
            evaluator
            or SupervisedEvaluator()
        )

        # learned attributes inside the SupervisedModelTrainer class
        self.trained_pipelines_: dict[str, Pipeline] = {}
        self.metrics_: pd.DataFrame | None = None

        self.X_train_: pd.DataFrame | None = None
        self.X_test_: pd.DataFrame | None = None
        self.y_train_: pd.Series | None = None
        self.y_test_: pd.Series | None = None

    # helpers
    def _model_uses_sample_weight(
            self,
            model_name: str
    ) -> bool:
        """
        Return True for models requiring sample-weight handling

        LR and tree models already receive class_weight, while we will pass balanced sample weights during fitting 
        for GradientBoostingClassifier as it does not support class_weight directly
        """
        return model_name in ['gradient_boosting_light']
    
    def _save_model(
            self,
            model_name: str,
            pipeline: Pipeline
    ) -> None:
        """ Save trained model pipeline into model_dir """
        self.model_dir.mkdir(
            parents = True,
            exist_ok = True
        )

        model_path = self.model_dir / f'{model_name}.joblib'

        joblib.dump(
            pipeline,
            model_path
        )
    
    def _save_metrics(
            self,
            metrics_df: pd.DataFrame
    ) -> None:
        """ Save model comparison metrics """
        self.metrics_dir.mkdir(
            parents = True,
            exist_ok = True
        )

        csv_safe_metrics = self.make_classification_metrics_csv_safe(
            metrics_df = metrics_df
        )

        csv_safe_metrics.to_csv(
            self.metrics_dir / 'supervised_model_metrics.csv',
            index = False
        )
    
    # supervised model trainer engine    
    def get_candidate_models(
            self
    ) -> dict:
        """ Return baseline and benchmark binary classifiers """
        classifier_dict = {
            'logistic_regression_baseline': LogisticRegression(
                max_iter = 5000,
                class_weight = 'balanced',
                solver = 'lbfgs',
                C = 0.01,
                tol = 1e-3,
                random_state = self.random_state
            ),
            'decision_tree_depth4': DecisionTreeClassifier(
                max_depth = 4,
                min_samples_leaf = 100,
                class_weight = 'balanced',
                random_state = self.random_state
            ),
            'random_forest_light': RandomForestClassifier(
                n_estimators = 300,
                max_depth = 8,
                min_samples_leaf = 50,
                class_weight = 'balanced_subsample',
                random_state = self.random_state,
                n_jobs = -1
            ),
            'gradient_boosting_light': GradientBoostingClassifier(
                n_estimators = 150,
                learning_rate = 0.05,
                max_depth = 3,
                random_state = self.random_state
            )
        }
        
        return classifier_dict
    
    def load_data(
            self
    ) -> pd.DataFrame:
        """ Load processed supervised feature dataset """
        return load_supervised_features()
    
    def split_predictors_and_target(
            self,
            df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """ Split supervised dataframe into predictors X. and target Y """
        X, y = self.preprocessing_builder.split_target(df = df)

        return X, y
    
    def create_train_test_split(
            self,
            X: pd.DataFrame,
            y: pd.Series
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """ Create stratified train-test split, where stratification keeps the minority class distribution similar in train and test sets """
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size = self.test_size,
            stratify = y,
            random_state = self.random_state
        )

        return X_train, X_test, y_train, y_test
    
    # model pipeline methods
    def build_model_pipeline(
            self,
            X_train: pd.DataFrame,
            model
    ) -> Pipeline:
        """  
        Build full sklearn pipeline:
            model preprocessing + classifier
        """
        preprocessing_pipeline = (
            self.preprocessing_builder
            .build_model_preprocessing_pipeline(X = X_train)
        ) 
        
        model_pipeline = Pipeline(
            steps = [
                ('preprocessor', preprocessing_pipeline),
                ('classifier', clone(model))
            ]
        )

        return model_pipeline
    
    def fit_single_model(
            self,
            model_name: str,
            model,
            X_train: pd.DataFrame,
            y_train: pd.Series
    ) -> Pipeline:
        """ Fit one full model pipeline """
        # model preprocessing + classifier pipeline
        pipeline = self.build_model_pipeline(
            X_train = X_train,
            model = model
        )

        # fitting pipeline method; use balanced sample weights if class_weight is not supported
        if self._model_uses_sample_weight(model_name = model_name):

            sample_weight = compute_sample_weight(
                class_weight = 'balanced',
                y = y_train
            )

            pipeline.fit(
                X = X_train,
                y = y_train,
                classifier__sample_weight = sample_weight
            )

        else:
            pipeline.fit(
                X = X_train,
                y = y_train
            )
        
        return pipeline
    
    def evaluate_single_model(
            self,
            model_name: str,
            pipeline: Pipeline,
            X_test: pd.DataFrame,
            y_test: pd.Series
    ) -> dict:
        """ Evaluate one fitted model pipeline using SupervisedEvaluator """
        classification_metrics = (
            self.evaluator.evaluate(
                model = pipeline,
                model_name = model_name,
                X_test = X_test,
                y_test = y_test,
                threshold = self.threshold
            )
        )

        return classification_metrics
    
    def train_all_models(
            self,
            save_models: bool = True,
            save_metrics: bool = True
    ) -> pd.DataFrame:
        """ Train all candidate (baseline + benchmark) models and evaluate them on the test set """
        # load supervised data
        df = self.load_data()

        # Split supervised dataframe into predictors X. and target Y 
        X, y = self.split_predictors_and_target(df = df)

        # train-test split
        X_train, X_test, y_train, y_test = self.create_train_test_split(
            X = X,
            y = y
        )

        self.X_train_ = X_train
        self.X_test_ = X_test
        self.y_train_ = y_train
        self.y_test_ = y_test

        metrics: list[dict] = []

        for model_name, model in self.get_candidate_models().items():

            print(f'Training {model_name}')

            # fit each model individually
            pipeline = self.fit_single_model(
                model_name = model_name,
                model = model,
                X_train = X_train,
                y_train = y_train
            )

            # classification model metrics
            model_metrics = self.evaluate_single_model(
                model_name = model_name,
                pipeline = pipeline,
                X_test = X_test,
                y_test = y_test
            )

            metrics.append(model_metrics)

            self.trained_pipelines_[model_name] = pipeline

            if save_models:
                self._save_model(
                    model_name = model_name,
                    pipeline = pipeline
                )
        
        metrics_df = self.evaluator.to_dataframe(metrics = metrics)

        self.metrics_ = metrics_df

        if save_metrics:
            self._save_metrics(metrics_df = metrics_df)
        
        return metrics_df
    
    # reports
    def get_classification_report(
            self,
            model_name: str
    ) -> str:
        """ Return classification report for a trained model """
        # sanity checks
        if self.metrics_ is None:
            raise ValueError('No metrics found, please call train_all_models() first!')
        
        if model_name not in self.metrics_['model'].values:
            raise ValueError(f'{model_name} not found in evaluated models!')
        
        report = (
            self.metrics_
            .loc[
                self.metrics_['model'] == model_name,
                'classification_report'
            ]
            .iloc[0]
        )

        return report
    
    @staticmethod
    def make_classification_metrics_csv_safe(
        metrics_df: pd.DataFrame
    ) -> pd.DataFrame:
        """ Convert metrics dataframe into a csv friendly format """
        # drop confusion matrix and classification_report 
        cols_to_drop = [
            'confusion_matrix',
            'classification_report'
        ]

        csv_friendly_metrics = metrics_df.drop(
            columns = [
                col
                for col in cols_to_drop
                if col in metrics_df.columns
            ],
            errors = 'ignore'
        )

        return (
            csv_friendly_metrics
            .sort_values(
                by = 'pr_auc',
                ascending = False
            )
            .reset_index(drop = True)
        )