from typing import Callable, Optional, Any, List, Union
from tqdm.auto import tqdm
import torch
import numpy as np




def compute_adaptive_bandwidth(features):
    """ Compute adaptive bandwidth using average distance to nearest neighbors. """
    if features.size(0) == 1:
        return 1.0
    pairwise_distances = torch.cdist(features, features, p=2)
    bandwidth = pairwise_distances.topk(2, largest=False, dim=1).values[:, 1].mean().item()
    return bandwidth


def is_singular(A: torch.Tensor, TOL=1e-8):
    det = torch.linalg.det(A)
    return torch.isclose(det, torch.tensor(0.), atol=TOL)


def gaussian_kernel(X: torch.Tensor, Y: torch.Tensor, sigma=1.0):
    """ Computes the Gaussian (RBF) kernel between two tensors.
        Args:
            X: Tensor of shape [N, D]
            Y: Tensor of shape [M, D]
            sigma: Bandwidth parameter for the Gaussian kernel
        Returns:
            Kernel matrix of shape [N, M]
    """
    X = X.unsqueeze(1)  # [N, 1, D]
    Y = Y.unsqueeze(0)  # [1, M, D]
    dist = torch.cdist(X, Y, p=2) ** 2  # Pairwise squared distances
    return torch.exp(-dist / (2 * sigma ** 2))


######################################################################################
### Maximum Mean Discrepancy (MMD) code for feature space comparisons
######################################################################################

# TODO: might want to make this a torch.nn.Module for robustness and chaining in the pipelines later
class MMD:
    """ Implements Maximum Mean Discrepancy (MMD) for feature space comparison. """
    def __init__(self, kernel_fn: Callable[[torch.Tensor, torch.Tensor, Optional[Any]], torch.Tensor] = gaussian_kernel):
        # TODO: need a bandwidth parameter that can be computed adaptively if never set
        self.kernel_fn = kernel_fn

    def compute(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """ Computes the MMD between two feature distributions X and Y.
            Args:
                X: Feature embeddings from the in-domain dataset (shape: [N, D])
                Y: Feature embeddings from the OOD dataset (shape: [M, D])
            Returns:
                MMD score (scalar tensor)
        """
        K_XX = self.kernel_fn(X, X)  # Kernel matrix for X (shape: [N, N]) - compares every pair of samples in source features X
        K_YY = self.kernel_fn(Y, Y)  # Kernel matrix for Y (shape: [M, M]) - compares every pair of samples in OOD features Y
        K_XY = self.kernel_fn(X, Y)  # Cross-kernel matrix between X and Y (shape: [N, M])  - compares every pair of samples in source features X and OOD features Y
        mmd = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
        return mmd

def compute_mmd(X: torch.Tensor, Y: torch.Tensor, bandwidth=None, per_sample=False) -> Union[float, List[float]]:
    """ Computes the MMD between X and Y, with optional per-sample computation for single target (Y) samples
        Args:
            target_sample: Embedding of the target sample [D].
            source_clusters: List of density tensors for each source cluster.
        Returns:
            List of MMD scores between the target sample and each cluster.
    """
    if X.ndim > 2: X = X.flatten(start_dim=1)
    if Y.ndim > 2: Y = Y.flatten(start_dim=1)
    if bandwidth is None:
        bandwidth = compute_adaptive_bandwidth(torch.cat([X, Y], dim=0))
    # TODO: replace these with a new Gaussian kernel function that doesn't perform extra unsqueezing
    if not per_sample:
        dxx = torch.cdist(X, X) ** 2
        dyy = torch.cdist(Y, Y) ** 2
        dxy = torch.cdist(X, Y) ** 2
        # Compute RBF kernel for each distance matrix
        K_xx = torch.exp(-dxx / (2 * bandwidth ** 2)).mean()
        K_yy = torch.exp(-dyy / (2 * bandwidth ** 2)).mean()
        K_xy = torch.exp(-dxy / (2 * bandwidth ** 2)).mean()
        return (K_xx + K_yy - 2*K_xy).item()
    else:
        # in this mode, X is typically a tensor of feature density clusters and Y is a single target sample
        # compute the MMD between each cluster in X and the single target sample in Y
        return [compute_mmd(X, y.unsqueeze(0), bandwidth) for y in Y]


#& old code had `evaluate_mmd` as well, which created MoKDE models given entire dataloaders
# not very rigorous really:
    # source_density = torch.cat(mokde_source.get_density_tensors(), dim=0)
    # ood_density = torch.cat(mokde_ood.get_density_tensors(), dim=0)
    # mmd_score = torch.mean((source_density - ood_density) ** 2)

#& old code also had `compute_mmd_scores` which iterated over test sample points and called `evaluate_mmd` for each one separately




######################################################################################
### Mahalanobis Distance code for feature space comparisons
######################################################################################

#& formerly `calculate_mahalanobis_distances` in the old project - numpy-based
def compute_mahalanobis_distances_from_hulls(test_features: np.ndarray, cluster_hulls: np.ndarray, source_features: np.ndarray):
    from scipy.spatial.distance import mahalanobis
    distances = []
    for test_point in test_features:
        min_distance = float('inf')
        # Compute distance to each cluster's convex hull
        for cluster_id, hull in tqdm(cluster_hulls.items(), desc="Calculating Mahalanobis distances"):
            cluster_points = source_features[hull.vertices]
            cov_matrix = np.cov(cluster_points, rowvar=False)
            mean = np.mean(cluster_points, axis=0)
            # Mahalanobis distance to the mean of the cluster
            distance = mahalanobis(test_point, mean, np.linalg.inv(cov_matrix))
            min_distance = min(min_distance, distance)
        distances.append(min_distance)
    return distances


def compute_mahalanobis(source_features: torch.Tensor, test_features: torch.Tensor) -> float:
    """ Computes the Mahalanobis distance between two density tensors
        Args:
            source_features, test_features: Density tensors for source and test dataset of shape [K, N] for K clusters and N feature dimensions.
        Returns:
            Mahalanobis distance (scalar tensor).
    """
    mean_diff = (torch.mean(source_features, dim=0) - torch.mean(test_features, dim=0)).reshape(-1, 1)
    print("mean_diff shape: ", mean_diff.shape)
    # NOTE: default is unbiased with Bessel’s correction (N-1 in the denominator)
    cov_sum = torch.cov(source_features.T)
    cov_sum += torch.cov(test_features.T) # done in place to use less CUDA memory
    cov_sum = (cov_sum + cov_sum.T)/2  # Ensure covariance matrix is symmetric
    # ridge regression if the covariance matrix is singular (det(cov_sum) = 0)
    while is_singular(cov_sum):
        print("Singular covariance matrix detected. Adding regularization.")
        traces = torch.diagonal(cov_sum, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
        identity = torch.eye(cov_sum.shape[-1], device=cov_sum.device).expand_as(cov_sum)
        cov_sum += 1e-5*traces.unsqueeze(-1)*identity
    mahalanobis_distance = torch.sqrt(mean_diff.T @ torch.linalg.inv(cov_sum) @ mean_diff)
    return mahalanobis_distance.item()


#& older, unused code from the original project - might be necessary at times for large numbers of features:
# #! MIGHT DELETE - unused everywhere
# def compute_mahalanobis_reduced(source_features, test_features, n_components=256, regularization=1e-8):
#     """ Computes the Mahalanobis distance between two feature sets, using PCA for dimensionality reduction.
#         Args:
#             source_features (torch.Tensor): Feature tensor for the source dataset, shape [B, N].
#             test_features (torch.Tensor): Feature tensor for the test dataset, shape [B, N].
#             n_components (int): Number of PCA components to retain for dimensionality reduction.
#             regularization (float): Regularization factor for the covariance matrix to handle singularity.
#         Returns:
#             torch.Tensor: Mahalanobis distance (scalar tensor).
#     """
#     # Step 1: Dimensionality Reduction with PCA
#     source_mean = torch.mean(source_features, dim=0, keepdim=True)
#     source_centered = source_features - source_mean
#     u, s, vh = torch.linalg.svd(source_centered, full_matrices=False)
#     # Retain top `n_components` principal components
#     pca_projection = vh[:n_components]
#     source_reduced = source_centered @ pca_projection.T
#     test_reduced = (test_features - source_mean) @ pca_projection.T
#     # Step 2: Compute Mean Vector Difference
#     mean_diff = (source_reduced.mean(dim=0) - test_reduced.mean(dim=0)).reshape(-1, 1)
#     # Step 3: Calculate Combined Covariance Matrix with Regularization
#     cov_sum = torch.cov(source_reduced.T) + torch.cov(test_reduced.T)
#     cov_sum = (cov_sum + cov_sum.T) / 2  # Ensure symmetry
#     # Apply regularization scaled by the trace of the covariance matrix
#     trace_val = torch.trace(cov_sum)
#     cov_sum += regularization * trace_val * torch.eye(cov_sum.shape[0], device=cov_sum.device)
#     # Step 4: Mahalanobis Distance Calculation
#     cov_inv = torch.linalg.inv(cov_sum)
#     mahalanobis_distance = torch.sqrt((mean_diff.T @ cov_inv @ mean_diff).squeeze())
#     return mahalanobis_distance

