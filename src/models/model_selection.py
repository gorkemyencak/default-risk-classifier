import pandas as pd

from pathlib import Path

from config import METRICS_DIR

class SupervisedModelSelector:
    """
    Selecting final classification model using both statistical performance and business opinion constraints

    SupervisedModelSelector class combines the following metrics when deciding upon the final classifier:
        - threshold-independent metrics: ROC-AUC & PR-AUC
        - selected threshold metrics based on business opinion and investigation capacity: recall, precision, f1-score
        - operational metrics such as predicted positive rate    
    """
    def __init__(
            self,
            metrics_dir: Path = METRICS_DIR
    ) -> None:
        # attributes
        self.metrics_dir = metrics_dir
    
    # helper functions
    @staticmethod
    def _drop_complex_metric_columns(
        metrics_df: pd.DataFrame
    ) -> pd.DataFrame:
        """ Drop columns that are not useful for model selection tables """
        cols_to_drop = [
            'confusion_matrix',
            'classification_report'
        ]

        return (
            metrics_df
            .drop(
                columns = [
                    col
                    for col in cols_to_drop
                    if col in metrics_df.columns
                ],
                errors = 'ignore'
            )
        )
    
    @staticmethod
    def _add_model_role(
        model_selection_table: pd.DataFrame
    ) -> pd.DataFrame:
        """ Add model role labels for business communication """
        model_selection_table = model_selection_table.copy()

        def map_role(model_name: str) -> str:
            
            if 'logistic_regression' in model_name:
                return 'Explainable baseline'
            if 'decision_tree' in model_name:
                return 'Interpretable non-linear benchmark'
            if 'random_forest' in model_name:
                return 'Tree ensemble benchmark'
            if 'gradient_boosting' in model_name:
                return 'Boosting benchmark'
            return 'Candidate model'
        
        # map model role
        model_selection_table['model_role'] = model_selection_table['model'].map(map_role)

        return model_selection_table
    
    
    # supervised model selector engine
    def combine_model_metrics_and_thresholds(
            self,
            model_metrics: pd.DataFrame,
            selected_thresholds: pd.DataFrame
    ) -> pd.DataFrame:
        """  
        Combine model-level metrics and selected threshold metrics

        model_metrics:
            Output from SupervisedModelTrainer.train_all_models()
        
        selected_thresholds:
            Output from ThresholdTuner.select_best_thresholds_for_all_models()
        """
        # drop complex columns from model_metrics
        model_metrics_clean = self._drop_complex_metric_columns(metrics_df = model_metrics)

        # rename threshold-dependent metrics based on business opinion, capacity & constraints
        threshold_cols_to_rename = {
            col: f'selected_{col}'
            for col in selected_thresholds.columns
            if col != 'model'
        }

        selected_thresholds_renamed = selected_thresholds.rename(
            columns = threshold_cols_to_rename
        )

        # merge model metrics with selected threshold metrics
        model_selection_table = model_metrics_clean.merge(
            right = selected_thresholds_renamed,
            how = 'left',
            on = 'model'
        )

        return model_selection_table
    
    def apply_business_constraints(
            self,
            model_selection_table: pd.DataFrame,
            min_pr_auc: float | None = None,
            min_recall: float | None = None,
            min_precision: float | None = None,
            max_predicted_positive_rate: float | None = None
    ) -> pd.DataFrame:
        """ Filter candidate models based on business constraints """
        filtered_table = model_selection_table.copy()

        # filter model selection tables based on business constraints
        if min_pr_auc is not None:
            filtered_table = filtered_table.query('pr_auc >= @min_pr_auc')
        
        if min_recall is not None:
            filtered_table = filtered_table.query('selected_recall >= @min_recall')
        
        if min_precision is not None:
            filtered_table = filtered_table.query('selected_precision >= @min_precision')
        
        if max_predicted_positive_rate is not None:
            filtered_table = filtered_table.query('selected_predicted_positive_rate <= @max_predicted_positive_rate')
        
        return filtered_table
    
    def rank_candidate_models(
            self,
            model_selection_table: pd.DataFrame,
            sort_columns: list[str] | None = None
    ) -> pd.DataFrame:
        """  
        Rank candidate models

        Default ranking policy:
            1) PR-AUC
            2) selected F1-Score
            3) selected Recall
            4) selected Precision
            5) ROC-AUC        
        """
        # define default ranking policy if sort_columns is not provided
        if sort_columns is None:
            sort_columns = [
                'pr_auc',
                'selected_f1_score',
                'selected_recall',
                'selected_precision',
                'roc_auc'
            ]
        
        # sanity check
        existing_sort_cols = [
            col
            for col in sort_columns
            if col in model_selection_table.columns
        ]

        return (
            model_selection_table
            .sort_values(
                by = existing_sort_cols,
                ascending = False
            )
            .reset_index(drop = True)
        )
    
    def select_final_model(
            self,
            model_selection_table: pd.DataFrame,
            min_pr_auc: float | None = None,
            min_recall: float | None = None,
            min_precision: float | None = None,
            max_predicted_positive_rate: float | None = None,
            sort_columns: list[str] | None = None
    ) -> pd.DataFrame:
        """ Select final model after applying optional business constraints """
        # filter model selection table using business contraints
        filtered_table = self.apply_business_constraints(
            model_selection_table = model_selection_table,
            min_pr_auc = min_pr_auc,
            min_recall = min_recall,
            min_precision = min_precision,
            max_predicted_positive_rate = max_predicted_positive_rate
        )

        # sanity check
        if filtered_table.empty:
            raise ValueError('No candidate model satisfies the provided business constraints')
        
        # extract ranked model selection table
        ranked_filtered_table = self.rank_candidate_models(
            model_selection_table = filtered_table,
            sort_columns = sort_columns
        )

        return (
            ranked_filtered_table
            .head(1)
            .reset_index(drop = True)
        )
    
    def save_model_selection_table(
            self,
            model_selection_table: pd.DataFrame,
            file_name: str = 'supervised_model_selection_table.csv'
    ) -> None:
        """ Save model selection table """
        self.metrics_dir.mkdir(
            parents = True,
            exist_ok = True
        )

        model_selection_table.to_csv(
            self.metrics_dir / file_name,
            index = False
        )