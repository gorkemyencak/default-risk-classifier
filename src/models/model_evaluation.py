import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)

class SupervisedEvaluator:
    """ Supervised model evaluator for binary classification models """
    def __init__(self) -> None:
        pass

    def evaluate(
            self,
            model,
            model_name: str,
            X_test: pd.DataFrame,
            y_test: pd.DataFrame,
            threshold: float = 0.5
    ) -> dict:
        """ Return classification metrics for a binary classifier """        
        # model predictions
        y_probability = model.predict_proba(X_test)[:, 1]
        y_prediction = (y_probability >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true = y_test,
            y_pred = y_prediction
        ).ravel()

        results = {
            'model': model_name,
            'threshold': threshold,
            'accuracy': accuracy_score(
                y_true = y_test,
                y_pred = y_prediction
            ),
            'balanced_accuracy': balanced_accuracy_score(
                y_true = y_test,
                y_pred = y_prediction
            ),
            'precision': precision_score(
                y_true = y_test,
                y_pred = y_prediction,
                zero_division = 0
            ),
            'recall': recall_score(
                y_true = y_test,
                y_pred = y_prediction,
                zero_division = 0
            ),
            'f1_score': f1_score(
                y_true = y_test,
                y_pred = y_prediction,
                zero_division = 0
            ),
            'roc_auc': roc_auc_score(
                y_true = y_test,
                y_score = y_probability
            ),
            'pr_auc': average_precision_score(
                y_true = y_test,
                y_score = y_probability
            ),
            'true_positive': tp,
            'true_negative': tn,
            'false_positive': fp,
            'false_negative': fn,
            'predicted_positive_rate': np.mean(y_prediction),
            'confusion_matrix': confusion_matrix(
                y_true = y_test,
                y_pred = y_prediction
            ),
            'classification_report': classification_report(
                y_true = y_test,
                y_pred = y_prediction
            )
        }

        return results
    
    @staticmethod
    def to_dataframe(
        metrics: list[dict]
    ) -> pd.DataFrame:
        """ Convert list of metric dictionaries into a sorted dataframe by PR-AUC score """
        return (
            pd.DataFrame(metrics)
            .sort_values(
                by = 'pr_auc',
                ascending = False
            )
            .reset_index(drop = True)
        )