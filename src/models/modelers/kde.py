from typing import List, Tuple
import torch
#import torch_kmeans
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans

from src.models.modeler_base import ModelerBase



class MixtureOfKDEsModeler(ModelerBase):
    def __init__(self, feature_dim, n_clusters, bandwidth=1.0, adaptive=True, device="cuda"):
        self.kde_model = ClusteredKDEModel(feature_dim, n_clusters, bandwidth, adaptive, device)

    def fit(self, X: torch.Tensor):
        self.kde_model.update_full_fit(X)

    # def transform(self, data):
    #     # your density or embedding logic

    def score_samples(self, X: torch.Tensor) -> torch.Tensor:
        return torch.tensor([self.kde_model.compute_sample_density(x) for x in X])

    def save(self, path):
        self.kde_model.save_kde_model(path)

    def load(self, path):
        # Returns self, so chaining works
        self.kde_model = ClusteredKDEModel.load_kde_model(path)
        return self




# Adaptive bandwidth calculation
def compute_adaptive_bandwidth(features):
    #print("features shape:", features.shape)
    pairwise_distances = torch.cdist(features, features, p=2)
    #print("pairwise distances shape:", pairwise_distances.shape)
    # Use mean of the distance to nearest neighbors as bandwidth
    if features.size(0) == 1:
        return 1.0
    bandwidth = pairwise_distances.topk(2, largest=False, dim=1).values[:, 1].mean().item()
    return bandwidth


######################################################################################
### Mixture of KDEs (MoKDE) using adaptive, weighted KDEs within each cluster
######################################################################################
class ClusteredKDEModel:
    """ A Mixture of KDEs (MoKDE) using adaptive, weighted KDEs within each cluster.
        Incremental clustering is achieved via MiniBatch KMeans.
    """
    def __init__(self, feature_dim, n_clusters, bandwidth=1.0, adaptive=True, device='cuda'):
        # TODO: add more flexible bandwidth handling, e.g. using a list of bandwidths for each cluster, None for adaptive
        self.n_clusters = n_clusters
        self.feature_dim = feature_dim  # Dimensionality of the feature embeddings
        #self.bandwidth = bandwidth # commenting out in favor of adaptive bandwidth everywhere
        self.adaptive = adaptive
        self.device = device
        # Initialize KDE models for each cluster with full feature dimensionality
        self.kde_models = [[] for _ in range(n_clusters)]  # Store full embeddings
        # Initialize cluster-wise KDE models
        # self.kde_models = [EnhancedKDEModel(feature_dim, bandwidth, adaptive, device)
        #                    for _ in range(n_clusters)]
        self.bandwidths = [None] * n_clusters  # Store adaptive bandwidth for each cluster
        # TODO: might want to move to the torch_kmeans version soon
        self.clusterer = MiniBatchKMeans(n_clusters=n_clusters, batch_size=32, random_state=42)

    def update(self, batch_features):
        """ Updates the MoKDE with batch features by assigning them to clusters.
            Args:
                batch_features: Feature embeddings [B, D].
        """
        ### ChatGPT wants to use global average pooling to completely condense the spatial dimension; not sure about that yet
        ###batch_features = torch.mean(batch_features, dim=(2, 3))  # Pool to (B, N) shape
        batch_features = batch_features.reshape(batch_features.size(0), -1)  # Flatten to [B, D]
        batch_features_np = batch_features.cpu().numpy()  # Convert to numpy
        # Perform initial fit or incremental update with partial_fit
        self.clusterer.partial_fit(batch_features_np)
        # Assign each feature to a cluster (using CPU to save memory)
        cluster_assignments = self.clusterer.predict(batch_features_np)
        # Update the appropriate KDE model for each assigned cluster
        #batch_features = torch.tensor(batch_features).to(self.device)
        for i in range(self.n_clusters):
            cluster_indices = torch.where(torch.tensor(cluster_assignments) == i)[0]
            if len(cluster_indices) > 0:
                # update with cluster features
                #self.kde_models[i].update(batch_features[cluster_indices])
                cluster_features = batch_features[cluster_indices]
                if self.kde_models[i] == []:
                    # Initialize with the cluster features
                    self.kde_models[i] = cluster_features.mean(dim=0, keepdim=True)
                else:
                    # Update cluster mean features
                    self.kde_models[i] = (self.kde_models[i] + cluster_features.mean(dim=0, keepdim=True)) / 2
                # Compute adaptive bandwidth for the cluster if enabled
                if self.adaptive:
                    self.bandwidths[i] = compute_adaptive_bandwidth(cluster_features)

    def update_full_fit(self, full_features):
        """ Updates the MoKDE with the entire source feature tensor in one go.
            Args:
                full_features: Entire feature embeddings tensor [N, D].
        """
        full_features = full_features.reshape(full_features.size(0), -1)  # Flatten to [N, D]
        full_features_np = full_features.cpu().numpy()  # Convert to numpy
        # Perform full fit with all features
        cluster_assignments = self.clusterer.fit_predict(full_features_np)
        for i in range(self.n_clusters):
            cluster_indices = torch.where(torch.tensor(cluster_assignments) == i)[0]
            if len(cluster_indices) > 0:
                cluster_features = full_features[cluster_indices]
                if self.kde_models[i] == []:
                    # Initialize with the cluster features
                    self.kde_models[i] = cluster_features.mean(dim=0, keepdim=True)
                else:
                    # Update cluster mean features
                    self.kde_models[i] = (self.kde_models[i] + cluster_features.mean(dim=0, keepdim=True)) / 2
                # Compute adaptive bandwidth for the cluster if enabled
                if self.adaptive:
                    self.bandwidths[i] = compute_adaptive_bandwidth(cluster_features)

    def rbf_kernel_density(self, test_sample, cluster_index):
        """Compute the RBF kernel density of a test sample relative to a specific cluster KDE."""
        kde_cluster = self.kde_models[cluster_index]
        bandwidth = self.bandwidths[cluster_index] if self.adaptive else 1.0  # Default bandwidth if not adaptive
        scaled_diff = (test_sample - kde_cluster) / bandwidth
        density = torch.exp(-0.5 * torch.sum(scaled_diff ** 2)) / (bandwidth ** kde_cluster.size(1))
        return density

    def compute_sample_density(self, test_sample):
        """ Compute the density of a test sample using the mixture of KDEs """
        densities = []
        for i in range(self.n_clusters):
            if len(self.kde_models[i]) > 0:  # Check if cluster is initialized
                density = self.rbf_kernel_density(test_sample, i)
                densities.append(density)
        return torch.mean(torch.stack(densities))  # Average density over all clusters

    def save_kde_model(self, save_path):
        torch.save({'kde_models': self.kde_models, 'bandwidths': self.bandwidths}, save_path)

    @staticmethod
    def load_kde_model(load_path, device='cuda'):
        data = torch.load(load_path, map_location=device)
        kde_model = ClusteredKDEModel(feature_dim=data['kde_models'][0].size(1),
                                      n_clusters=len(data['kde_models']),
                                      adaptive=True, device=device)
        kde_model.kde_models = data['kde_models']
        kde_model.bandwidths = data['bandwidths']
        return kde_model

    def get_density_tensors(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """ Returns the average embedding for each cluster.
            Returns:
                List of average tensors for each cluster.
        """
        return self.kde_models, self.bandwidths






#---------------- deprecated functions from the old project --------------#
# For now, they're located in src/evaluate.kde_modeling.py in the old project
#& build_mokde_model - instantiated a ClusteredKDEModel and updated it with features or model + loader, then returned the fit model
#& build_kde_model_from_features - built a mokde_model from features read from the disk, then saved it if save_path was provided
#& build_mokde_model_with_optimal_k - iterated over fitting N clusters, found the K with the best silhouette score, then returned a model with K clusters


#& salvaging parts of `build_mokde_model` using only the feature input version:
def build_mokde_model(features: torch.Tensor, n_clusters=8, bandwidth=1.0, spatial_dims=(16,16), batch_size=16):
    # TODO: will probably need updating for streaming features if it's kept
    B, N, H, W = features.shape[:2]
    mokde_model = ClusteredKDEModel(N, n_clusters, bandwidth, device = features.device)
    if not all(dim1 == dim2 for dim1, dim2 in zip(spatial_dims, (H,W))):
        features = F.adaptive_avg_pool2d(features, spatial_dims)
    features = features.flatten(start_dim=1) if features.ndim > 2 else features
    for i in range(0, B, batch_size):
        end_index = min(i + batch_size, B)
        mokde_model.update(features[i:end_index])
    return mokde_model