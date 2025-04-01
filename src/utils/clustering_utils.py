from typing import Generator, Dict, Optional, Iterable, Union, Any
from tqdm.auto import tqdm
import torch
import numpy as np
from sklearn.metrics import silhouette_score




#----------------- Clustering utilities for feature embeddings -----------------#


#& NEW - helper for `incremental_find_optimal_cluster_count``
def create_kmeans_dict(method: str, batch_size=16, largest_k=32, k_values=None) -> Dict[int, Any]:
    if k_values is None:
        k_values = range(2, largest_k + 1)
    if method == "sklearn":
        # TODO: look into arguments to the constructor: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        from sklearn.cluster import MiniBatchKMeans
        return {k: MiniBatchKMeans(n_clusters=k, batch_size=batch_size, init='k-means++') for k in k_values}
    elif method == "torch":
        import torch_kmeans
        # TODO: look into arguments to the constructor: https://github.com/jokofa/torch_kmeans/blob/master/src/torch_kmeans/clustering/kmeans.py#L19
        return {k: torch_kmeans.KMeans(init_method='k-means++', n_clusters=k) for k in k_values}
    else:
        raise ValueError(f"Unknown method: {method}")

#& NEW - helper for `incremental_find_optimal_cluster_count``
def run_kmeans_prediction(kmeans_obj, features, method="sklearn"):
    if method == "sklearn":
        labels = kmeans_obj.predict(features)
    else:
        labels = kmeans_obj(features).labels.cpu().numpy()
    return labels


#& NEW - didn't exist in the old project, but I think this is how we'll need to handle streaming feature data
def incremental_find_optimal_cluster_count(
    method: str = "sklearn", max_clusters: int = 32, batch_size: int = 16
) -> Generator[Dict[int, Optional[float]], Union[torch.Tensor, np.ndarray], None]:
    """ Single-pass incremental approach over all k-values in parallel.
        :param method: "sklearn" or "torch"
        :param max_clusters: The largest k (not strictly needed if k_values is given)
        :param batch_size: Batch size for MiniBatchKMeans (sklearn)
        :return: A generator that yields a dictionary {k: silhouette_score} after each batch update.
                It expects .send(...) with a new batch of features (torch.Tensor or np.ndarray). Send None to terminate.
    """
    k_values = list(range(max_clusters))
    kmeans_dict = create_kmeans_dict(method, batch_size=batch_size, largest_k=max_clusters)
    # TODO: decide if I want to keep returning all silhouette scores or just the best K each iteration
    silhouette_scores: Dict[int, Optional[float]] = {k: None for k in k_values}
    while True:
        # generator suspends here and resumes when new data is sent
        features = (yield silhouette_scores)
        if features is None:
            break
        # convert to numpy if using sklearn
        if method == "sklearn" and isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        # partial fit or just call forward for each k
        for k in k_values:
            if method == "sklearn":
                kmeans_dict[k].partial_fit(features)
            else:
                # For torch_kmeans, you might have to handle incremental updates manually
                # e.g., run kmeans_obj(features) and update cluster centers as needed
                _ = kmeans_dict[k](features)
            # compute silhouette scores each k
            labels = run_kmeans_prediction(kmeans_dict[k], features, method=method)
            # avoid actually updating features and moving from device if not necessary
            feature_ptr = features.cpu().numpy() if isinstance(features, torch.Tensor) else features
            silhouette_scores[k] = silhouette_score(feature_ptr, labels)
        #best_k = max(silhouette_scores, key=silhouette_scores.get)  # Get the k with the highest silhouette score





#? NOTE: GPT thinks I called this `convex_hull_intersection` for some reason
# TODO: need to update this to use a streaming approach for input features as well
def calculate_convex_hulls(features, n_clusters):
    from sklearn.cluster import MiniBatchKMeans
    from scipy.spatial import ConvexHull
    kmeans = MiniBatchKMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(features)
    cluster_hulls = {}
    for cluster_id in tqdm(range(n_clusters), desc="Calculating convex hulls for each cluster"):
        cluster_points = features[labels == cluster_id]
        # Compute convex hull only if there are enough points
        if len(cluster_points) >= 3:
            hull = ConvexHull(cluster_points)
            cluster_hulls[cluster_id] = hull
        else:
            print(f"Cluster {cluster_id} skipped due to insufficient points.")
    return cluster_hulls



#& incremental_clustering() doesn't exist anymore