import pandas as pd
import joblib

from pathlib import Path

from sklearn.cluster import KMeans, MiniBatchKMeans, BisectingKMeans
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline

from config import (
    METRICS_DIR,
    MODELS_DIR
)

from src.data.data_loader import load_unsupervised_features
from src.clustering.clustering_preprocessing import ClusteringPreprocessingBuilder
from src.clustering.clustering_evaluation import ClusteringEvaluator

class ClusteringModelTrainer:
    """
    Train and evaluate unsupervised customer segmentation models

    The baseline clustering model is KMeans. MiniBatchKMeans is included as a scalable benchmark

    Key functions:
        - load processed unsupervised feature dataset
        - prepare clustering input
        - build preprocessing + clustering pipelines
        - train baseline and benchmark clustering models
        - evaluate internal clustering quality
        - save fitted clustering pipelines, metrics, and cluster labels
    
    Candidate models:
        - KMeans: centroid-based baseline
        - MiniBatchKMeans: scalable centriod-based benchmark
        - GaussianMixture: probabilistic/distributional bechmark
        - Birch: scalable hierarchical benchmark
    """
    def __init__(
            self,
            cluster_range: range = range(2, 9),
            random_state: int = 2,
            model_dir: Path = MODELS_DIR,
            metrics_dir: Path = METRICS_DIR,
            preprocessing_builder: ClusteringPreprocessingBuilder | None = None,
            evaluator: ClusteringEvaluator | None = None,
            algorithms: list[str] | None = None
    ) -> None:
        # attributes
        self.cluster_range = cluster_range
        self.random_state = random_state
        self.model_dir = model_dir
        self.metrics_dir = metrics_dir
        self.preprocessing_builder = (
            preprocessing_builder or ClusteringPreprocessingBuilder()
        )
        self.evaluator = (
            evaluator or ClusteringEvaluator(random_state = self.random_state)
        )
        self.algorithms = (
            algorithms or
            [
                'kmeans',
                'minibatch_kmeans',
                'gaussian_mixture',
                'birch'
            ]
        )

        # learned attributes inside the ClusteringModelTrainer class
        self.trained_pipelines_: dict[str, Pipeline] = {}
        self.metrics_: pd.DataFrame | None = None
        self.cluster_labels_: pd.DataFrame | None = None
        self.X_: pd.DataFrame | None = None
    
    # helper functions
    def _load_data(
            self
    ) -> pd.DataFrame:
        """ Load processed unsupervised feature dataset """
        return load_unsupervised_features()
    
    def _prepare_clustering_data(
            self,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """ Prepare dataframe for clustering """
        return (
            self
            .preprocessing_builder
            .clean_clustering_input(df = df)
        )
    
    def _model_name_matches_algorithms(
            self,
            model_name: str
    ) -> bool:
        """ Check whether model_name matches the requested algorithms """
        if 'minibatch_kmeans' in model_name:
            return 'minibatch_kmeans' in self.algorithms
        
        if 'bisecting_kmeans' in model_name:
            return 'bisecting_kmeans' in self.algorithms
        
        if 'kmeans' in model_name:
            return 'kmeans' in self.algorithms
        
        if 'gaussian_mixture' in model_name:
            return 'gaussian_mixture' in self.algorithms
        
        return False
    
    @staticmethod
    def _get_transformed_features(
        pipeline: Pipeline,
        X: pd.DataFrame
    ):
        """ Return transformed features from fitted clustering pipeline """
        return pipeline.named_steps['preprocessor'].transform(X)
    
    @staticmethod
    def _get_inertia(
        pipeline: Pipeline
    ) -> float | None:
        """ Return inertia if available """
        # clusterer pipeline
        clusterer = pipeline.named_steps['clusterer']

        if hasattr(clusterer, 'inertia_'):
            return float(clusterer.inertia_)
        
        return None
    
    @ staticmethod
    def _get_model_diagnostics(
        pipeline: Pipeline
    ) -> dict:
        """ Return model-specific diagnostics when available """
        # clusterer pipeline
        clusterer = pipeline.named_steps['clusterer']

        diagnostics = {}

        if hasattr(clusterer, 'lower_bound_'):
            diagnostics['gmm_lower_bound'] = float(clusterer.lower_bound_)
        
        if hasattr(clusterer, 'n_iter_'):
            diagnostics['n_iter'] = int(clusterer.n_iter_)
        
        return diagnostics
    
    def _save_model(
            self,
            model_name: str,
            pipeline: Pipeline
    ) -> None:
        """ Save trained clusterer pipeline into model_dir """
        self.model_dir.mkdir(
            parents = True,
            exist_ok = True
        )

        model_path = self.model_dir / f'{model_name}.joblib'

        joblib.dump(
            value = pipeline,
            filename = model_path
        )
    
    def _save_metrics(
            self,
            metrics_df: pd.DataFrame
    ) -> None:
        """ Save clustering metrics """
        self.metrics_dir.mkdir(
            parents = True,
            exist_ok = True
        )

        metrics_path = self.metrics_dir / 'clustering_model_metrics.csv'

        metrics_df.to_csv(
            metrics_path,
            index = False
        )
    
    def _save_cluster_labels(
            self,
            cluster_labels_df: pd.DataFrame
    ) -> None:
        """ Save cluster labels for all fitted clustering models """
        self.metrics_dir.mkdir(
            parents = True,
            exist_ok = True
        )

        labels_path = self.metrics_dir / 'cluster_labels.csv'

        cluster_labels_df.to_csv(
            labels_path,
            index = False
        )


    # clustering model trainer engine
    def get_candidate_clusterers(
            self,
            n_clusters: int
    ) -> dict:
        """ Return baseline and benchmark clusterers for a given number of clusters """
        # define all clusterers
        all_clusterers = {
            f'kmeans_k{n_clusters}': KMeans(
                n_clusters = n_clusters,
                random_state = self.random_state,
                n_init = 'auto',
                max_iter = 300
            ),
            f'minibatch_kmeans_k{n_clusters}': MiniBatchKMeans(
                n_clusters = n_clusters,
                random_state = self.random_state,
                n_init = 'auto',
                batch_size = 1024,
                max_iter = 300
            ),
            f'gaussian_mixture_k{n_clusters}': GaussianMixture(
                n_components = n_clusters,
                covariance_type = 'diag',
                random_state = self.random_state,
                n_init = 3,
                max_iter = 300
            ),
            f'bisecting_kmeans_k{n_clusters}': BisectingKMeans(
                n_clusters = n_clusters,
                random_state = self.random_state,
                n_init = 1,
                max_iter = 150
            ) 
        }

        # selected clusterers
        selected_clusterers = {
            model_name: model
            for model_name, model in all_clusterers.items()
            if self._model_name_matches_algorithms(model_name = model_name)
        }

        return selected_clusterers
    
    # clustering pipeline methods
    def build_clustering_pipeline(
            self,
            X: pd.DataFrame,
            clusterer
    ) -> Pipeline:
        """ 
        Build full clustering pipeline  
            clustering preprocessing + clusterer
        """
        # preprocessor
        clustering_preprocessor = (
            self
            .preprocessing_builder
            .build_clustering_preprocessor(X = X)
        )

        # preprocessor + clusterer pipeline
        clustering_pipeline = Pipeline(
            steps = [
                ('preprocessor', clustering_preprocessor),
                ('clusterer', clusterer)
            ]
        )

        return clustering_pipeline
    
    def fit_single_model(
            self,
            model_name: str,
            clusterer,
            X: pd.DataFrame
    ) -> tuple[Pipeline, pd.Series]:
        """ Fit one clustering pipeline and return cluster labels """
        # build preprocessing + clusterer pipeline
        clusterer_pipeline = self.build_clustering_pipeline(
            X = X,
            clusterer = clusterer
        )

        # predict labels
        labels = clusterer_pipeline.fit_predict(X = X)

        # get label series
        label_series = pd.Series(
            labels,
            name = model_name,
            index = X.index
        )

        return clusterer_pipeline, label_series
    
    # clustering model training
    def train_all_models(
            self,
            save_models: bool = True,
            save_metrics: bool = True,
            save_cluster_labels: bool = True
    ) -> pd.DataFrame:
        """ Train and evaluate all candidate clustering models across all cluster counts """
        # load unsupervised features
        df = self._load_data()

        # clean features
        X = self._prepare_clustering_data(df = df)
        self.X_ = X.copy()

        metrics: list[dict] = []
        cluster_label_frames: list[pd.Series] = []

        for n_clusters in self.cluster_range:
            
            # baseline + benchmark clusterers 
            candidate_clusterers = self.get_candidate_clusterers(n_clusters = n_clusters)

            for model_name, clusterer in candidate_clusterers.items():

                print(f'Training {model_name}')

                # fit each model individually
                pipeline, labels = self.fit_single_model(
                    model_name = model_name,
                    clusterer = clusterer,
                    X = X
                )

                # get transformed features
                X_transformed = self._get_transformed_features(
                    pipeline = pipeline,
                    X = X
                )

                # evaluate clustering metrics
                clusterer_metrics = self.evaluator.evaluate(
                    model_name = model_name,
                    n_clusters = n_clusters,
                    X_transformed = X_transformed,
                    labels = labels,
                    inertia = self._get_inertia(pipeline = pipeline)
                )

                # append model diagnostics
                clusterer_metrics.update(
                    self._get_model_diagnostics(
                        pipeline = pipeline
                    )
                )

                metrics.append(clusterer_metrics)
                cluster_label_frames.append(labels)

                self.trained_pipelines_[model_name] = pipeline

                # save fitted clusterers locally
                if save_models:
                    self._save_model(
                        model_name = model_name,
                        pipeline = pipeline
                    )
        
        # convert metrics into dataframe
        metrics_df = self.evaluator.to_dataframe(metrics = metrics)
        self.metrics_ = metrics_df

        # save metrics locally
        if save_metrics:
            self._save_metrics(metrics_df = metrics_df)

        # convert cluster labels into dataframe
        cluster_labels_df = pd.concat(
            cluster_label_frames,
            axis = 1
        )
        self.cluster_labels_ = cluster_labels_df

        # save cluster labels locally
        if save_cluster_labels:
            self._save_cluster_labels(cluster_labels_df = cluster_labels_df)
        
        return metrics_df