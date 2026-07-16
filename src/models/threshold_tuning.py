import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)

from config import METRICS_DIR

class ThresholdTuner:
    """
    ThresholdTuner class adjusts binary classification thresholds in favor of associated business costs; therefore it supports
    business-driven threshold selection

    Threshol choice should depend on:
        - false negatives: risky/default customers missed by the classifier
        - false positives: may cause losing financially strong customers incorrectly flagges as risky by the classifier
    """
    def __init__(
            self,
            thresholds: np.ndarray | None = None,
            metrics_dir: Path = METRICS_DIR
    ) -> None:
        # attributes
        self.thresholds = (
            thresholds
            if thresholds is not None
            else np.arange(0.05, 0.91, 0.05)
        )
        self.metrics_dir = metrics_dir
    
    # helper function
    @staticmethod
    def _evaluate_threshold(
        y_true,
        y_probability,
        threshold: float
    ) -> dict:
        """ Evaluate binary predictions at a given threshold """
        # get predictions
        y_prediction = (y_probability >= threshold).astype(int)

        # evaluation metrics
        tn, fp, fn, tp = confusion_matrix(
            y_true = y_true,
            y_pred = y_prediction
        ).ravel()

        predicted_positive_rate = float(np.mean(y_prediction))

        result_dict = {
            'threshold': threshold,
            'accuracy': accuracy_score(
                y_true = y_true,
                y_pred = y_prediction
            ),
            'balanced_accuracy': balanced_accuracy_score(
                y_true = y_true,
                y_pred = y_prediction
            ),
            'precision': precision_score(
                y_true = y_true,
                y_pred = y_prediction,
                zero_division = 0
            ),
            'recall': recall_score(
                y_true = y_true,
                y_pred = y_prediction,
                zero_division = 0
            ),
            'f1_score': f1_score(
                y_true = y_true,
                y_pred = y_prediction ,
                zero_division = 0
            ),
            'true_positive': tp,
            'true_negative': tn,
            'false_positive': fp,
            'false_negative': fn,
            'predicted_positive_rate': predicted_positive_rate
        }

        return result_dict
    
    # threshold tuning engine
    def tune_single_model(
            self,
            model,
            model_name: str,
            X_test: pd.DataFrame,
            y_test: pd.Series
    ) -> pd.DataFrame:
        """ Evaluate one fitted model across all candidate thresholds """
        # get predicted class probabilities
        y_probability = model.predict_proba(X_test)[:, 1]

        threshold_results = []

        for threshold in self.thresholds:

            result = self._evaluate_threshold(
                y_true = y_test,
                y_probability = y_probability,
                threshold = threshold
            )

            result['model'] = model_name

            threshold_results.append(result)
        
        return pd.DataFrame(threshold_results)
    
    def tune_all_models(
            self,
            trained_pipelines: dict[str, Pipeline],
            X_test: pd.DataFrame,
            y_test: pd.Series
    ) -> pd.DataFrame:
        """ Evaluate all fitted model pipelines across all candidate thresholds """
        results: list[pd.DataFrame] = []

        for model_name, pipeline in trained_pipelines.items():

            model_threshold_results = self.tune_single_model(
                model = pipeline,
                model_name = model_name,
                X_test = X_test,
                y_test = y_test
            )

            results.append(model_threshold_results)
        
        return pd.concat(
            results,
            ignore_index = True
        )
    
    def select_best_threshold(
            self,
            threshold_results: pd.DataFrame,
            model_name: str,
            selection_strategy: str = 'max_f1',
            min_recall: float | None = None,
            min_precision: float | None = None,
            max_predicted_positive_rate: float | None = None
    ) -> pd.DataFrame:
        """  
        Selecting the best threshold for the model using the selection strategy defined by business units

        Available selection strategies:
            - max_f1
            - max_precision
            - max_recall
            - max_balanced_accuracy
        
        Optional constraints:
            - min_recall
            - min_precision
            - max_predicted_positive_rate        
        """
        # get evaluation result of a fitted model across candidate threshold values
        model_results = (
            threshold_results
            .query('model == @model_name')
            .copy()
        )

        # sanity check
        if model_results.empty:
            raise ValueError(f'No threshold results found for the {model_name}')
        
        # filtering evaluation results based on optional constraints
        if min_recall is not None:
            model_results = (
                model_results
                .query('recall >= @min_recall')
            )
        
        if min_precision is not None:
            model_results = (
                model_results
                .query('precision >= @min_precision')
            )
        
        if max_predicted_positive_rate is not None:
            model_results = (
                model_results
                .query('predicted_positive_rate <= @max_predicted_positive_rate')
            )
        
        # sanity check after filtering
        if model_results.empty:
            raise ValueError('No threshold satisfies the provided optional business constraints')
        
        # strategy selection
        if selection_strategy == 'max_f1':
            sort_columns = [
                'f1_score',
                'precision',
                'recall'
            ]
        elif selection_strategy == 'max_balanced_accuracy':
            sort_columns = [
                'balanced_accuracy',
                'f1_score',
                'precision'
            ]
        elif selection_strategy == 'max_precision':
            sort_columns = [
                'precision',
                'recall',
                'f1_score'
            ]
        elif selection_strategy == 'max_recall':
            sort_columns = [
                'recall',
                'precision',
                'f1_score'
            ]
        else:
            raise ValueError(f'Invalid selection strategy: {selection_strategy}. Please choose from max_f1, max_balanced_accuracy, max_precision, or max_recall')
        
        sorted_model_results = (
            model_results
            .sort_values(
                by = sort_columns,
                ascending = False
            )
            .head(1)
            .reset_index(drop = True)
        )

        return sorted_model_results
    
    def select_best_thresholds_for_all_models(
            self,
            threshold_results: pd.DataFrame,
            selection_strategy: str = 'max_f1',
            min_recall: float | None = None,
            min_precision: float | None = None,
            max_predicted_positive_rate: float | None = None
    ) -> pd.DataFrame:
        """ Selecting the best threshold for each model using the selection strategy """
        results: list[pd.DataFrame] = []

        for model_name in threshold_results['model'].unique().tolist():

            selected_threshold = self.select_best_threshold(
                threshold_results = threshold_results,
                model_name = model_name,
                selection_strategy = selection_strategy,
                min_recall = min_recall,
                min_precision = min_precision,
                max_predicted_positive_rate = max_predicted_positive_rate
            )

            results.append(selected_threshold)
        
        return (
            pd.concat(
                results,
                ignore_index = True
            )
            .sort_values(
                'f1_score',
                ascending = False
            )
            .reset_index(drop = True)
        )
    
    # utility function
    def save_threshold_results(
            self,
            threshold_results: pd.DataFrame,
            file_name: str = 'threshold_tuning_results.csv'
    ) -> None:
        """ Save threshold tuning results into self.metrics_dir directory """
        self.metrics_dir.mkdir(
            parents = True,
            exist_ok = True
        )

        threshold_results.to_csv(
            self.metrics_dir / file_name,
            index = False
        )