import pandas as pd
import numpy as np

from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score
)

class ClusteringEvaluator:
    """ 
    Clustering model evaluator for unsupervised clustering models 
    
    Clustering quality is evaluated using internal validation metrics:
        - Silhouette Score: higher is better
        - Calinski-Harabasz score: higher is better
        - Davies-Boulding score: lower is better
        - inertia: lower is better
    """
    def __init__(
            self,
            silhouette_sample_size: int | None = 10_000,
            random_state: int = 2,
            min_cluster_size: int = 50,
            min_cluster_share: float = 0.005
    ) -> None:
        # attributes
        self.silhouette_sample_size = silhouette_sample_size
        self.random_state = random_state
        self.min_cluster_size = min_cluster_size
        self.min_cluster_share = min_cluster_share

    # helper functions
    @staticmethod
    def _get_number_of_assigned_clusters(
        labels
    ) -> int:
        """ Return number of unique assigned clusters """
        return len(np.unique(labels))
    
    @staticmethod
    def _to_numpy(
        X
    ):
        """ Convert matrix object to numpy array when needed """
        if hasattr(X, 'toarray'):
            return X.toarray()
        
        return np.asarray(X)
    
    def _get_stratified_sample_indices(
            self,
            labels,
            sample_size: int
    ) -> np.ndarray:
        """ 
        Return stratified sample of row indices from clustering labels, which guarantees that each asigned cluster is represented in the sample,
        for silhouette score calculation 
        """
        # convert labels into numpy array
        labels = np.asarray(labels)
        
        # randomness
        rng = np.random.default_rng(seed = self.random_state)

        # get unique cluster labels
        unique_labels = np.unique(labels)

        # get number of observations
        n_samples = len(labels)

        # checking whether requested sample size is greater than or equal to full observed dataset; if so, the function returns all indices
        if sample_size >= n_samples:
            return np.arange(n_samples)
        
        # checking whether requested sample_size is greater than or equal to unqiue labels to make sure each cluster is represented in the sample
        if sample_size < len(unique_labels):
            raise ValueError(
                f'sample_size: {sample_size} must be at least the number of unique labels '
                'to guarantee every cluster is represented'
            )
        
        sampled_indices: list[int] = []

        # guarantee one observation per cluster
        for label in unique_labels:

            # returning label indices inside labels array
            label_indices = np.where(labels == label)[0]

            # selecting indices of sample size 1 without replacement
            selected_index = rng.choice(
                label_indices,
                size = 1,
                replace = False
            )
        
            sampled_indices.append(int(selected_index[0]))
        
        sampled_indices_array = np.asarray(
            sampled_indices,
            dtype = int
        )

        # remaining slots in requested sample size when each cluster is guaranteed to be represented
        remaining_size = sample_size - len(sampled_indices_array)

        # check if no slack available
        if remaining_size == 0:
            return sampled_indices_array
        
        # get remaining indices that are not still represented in the sample
        all_indices = np.arange(n_samples)
        remaining_pool = np.setdiff1d(
            all_indices,
            sampled_indices_array
        )

        # determine additional indices that will be assigned to the sample
        additional_indices = rng.choice(
            remaining_pool,
            size = remaining_size,
            replace = False
        )

        # concatenate sampled_indices and additional_indices to return the final sample
        final_sampled_indices = np.concatenate([
            sampled_indices_array,
            additional_indices
        ])

        return final_sampled_indices
    
    def _calculate_silhouette_score(
            self,
            X_transformed,
            labels
    ) -> float:
        """ 
        Calculate silhouette score safely 
        
        Instead of passing sample_size directly to sklearn.metrics.silhouette_score, we perform stratified sampling as a private method to avoid
        one-cluster sampled subsets; hence, each unique label is represented in the sampled subsets
        """
        # convert labels into numpy array
        labels = np.asarray(labels)

        # convert features into numpy array
        X_array = self._to_numpy(X_transformed)

        # get the total number of observations
        n_observations = len(labels)

        # check whether sample_size is already provided; if not, then any sampling is not required 
        if self.silhouette_sample_size is None:
            
            return float(
                silhouette_score(
                    X = X_array,
                    labels = labels
                )
            )
        
        # make sure that provided sample size is less than or equal to the tota number of observations
        sample_size = min(
            self.silhouette_sample_size,
            n_observations
        )

        # get stratified sample indices
        sampled_indices = self._get_stratified_sample_indices(
            labels = labels,
            sample_size = sample_size
        )

        # get sample labels and features
        sampled_labels = labels[sampled_indices]
        sampled_X = X_array[sampled_indices]

        # number of unique assigned clusters
        n_sampled_clusters = self._get_number_of_assigned_clusters(labels = sampled_labels)

        if not 1 < n_sampled_clusters < len(sampled_labels):
            return float('nan')
        
        return float(
            silhouette_score(
                X = sampled_X,
                labels = sampled_labels
            )
        )
    
    def _get_cluster_size_summary(
            self,
            labels
    ) -> dict:
        """ Return cluster size diagnostics """
        # convert labels into numpy array
        labels = np.asarray(labels)

        # get unique labels
        unique_labels, counts = np.unique(
            labels,
            return_counts = True
        )

        # cluster size
        cluster_sizes = dict(
            zip(
                unique_labels.tolist(),
                counts.tolist()
            )
        )

        min_cluster_count = int(counts.min())
        min_cluster_share = float(
            counts.min() / len(labels)
        )

        return {
            'cluster_sizes': cluster_sizes,
            'min_cluster_count': min_cluster_count,
            'min_cluster_share': min_cluster_share
        }


    # clustering evaluator engine
    def evaluate(
            self,
            model_name: str,
            n_clusters: int,
            X_transformed,
            labels,
            inertia: float | None = None
    ) -> dict:
        """ Return clustering evaluation metrics """
        labels = np.asarray(labels)

        # total number of observations
        n_observations = len(labels)

        # get number of unique assigned cluster labels
        n_assigned_clusters = self._get_number_of_assigned_clusters(labels = labels)

        # get cluster size summary
        cluster_size_summary = self._get_cluster_size_summary(labels = labels)

        result = {
            'model': model_name,
            'n_clusters': n_clusters,
            'n_observations': n_observations,
            'n_assigned_clusters': n_assigned_clusters,
            'inertia': inertia,
            'davies_bouldin_score': np.nan,
            'calinski_harabasz_score': np.nan,
            'silhouette_score': np.nan,
            'is_valid_clustering': False,
            'evaluation_note': None,
            'cluster_sizes': cluster_size_summary['cluster_sizes'],
            'min_cluster_count': cluster_size_summary['min_cluster_count'],
            'min_cluster_share': cluster_size_summary['min_cluster_share']
        }

        # check n_assigned_clusters that ranges between 2 and n_observations
        if not 1 < n_assigned_clusters < n_observations:
            result['evaluation_note'] = (
                f'Invalid clustering solution: assigned {n_assigned_clusters} cluster(s). '
                f'Internal metrics require at least 2 clusters.'
            )
            
            return result
        
        # check degenerate solution
        if cluster_size_summary['min_cluster_count'] < self.min_cluster_size:
            result['evaluation_note'] = (
                f'Degenerate clustering solution: smallest cluster share is only {cluster_size_summary['min_cluster_share']:.4f}'
            )
        
        # if cluster labels are assigned properly
        result['is_valid_clustering'] = True
        result['evaluation_note'] = 'Valid clustering solution'

        result['davies_bouldin_score'] = davies_bouldin_score(
            X = X_transformed,
            labels = labels
        )

        result['calinski_harabasz_score'] = calinski_harabasz_score(
            X = X_transformed,
            labels = labels
        )

        result['silhouette_score'] = self._calculate_silhouette_score(
            X_transformed = X_transformed,
            labels = labels
        )

        return result
    
    @staticmethod
    def to_dataframe(metrics: list[dict]) -> pd.DataFrame:
        """ Convert clustering metrics into dataframe """
        return (
            pd.DataFrame(metrics)
            .sort_values(
                by = [
                    'is_valid_clustering',
                    'silhouette_score',
                    'davies_bouldin_score'
                ],
                ascending = [
                    False,
                    False,
                    True
                ]
            )
            .reset_index(drop = True)
        )