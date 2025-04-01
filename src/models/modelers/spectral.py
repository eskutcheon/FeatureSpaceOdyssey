import numpy as np
import torch
# local imports
from ..modeler_base import ModelerBase



#& imported from the old project's src/evaluate/manifold_modeling.py


class DiffusionMapModeler(ModelerBase):
    def __init__(self, **kwargs):
        self.model = DiffusionMap(**kwargs)

    def fit(self, X: torch.Tensor):
        self.model.fit(X)

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        return self.model.transform(X)

    def score_samples(self, X: torch.Tensor) -> torch.Tensor:
        # Optional: use reconstruction error, density approximation, etc.
        raise NotImplementedError("Density scoring not implemented for DiffusionMap.")





def normalize_similarity_matrix(similarity_matrix):
    # Normalize to get the transition matrix for diffusion
    row_sums = similarity_matrix.sum(dim=1, keepdim=True)
    return similarity_matrix / row_sums

def diffusion_map(similarity_matrix, n_components=10, t=1):
    # Normalize the similarity matrix to obtain the transition matrix
    transition_matrix = normalize_similarity_matrix(similarity_matrix)
    # Perform eigendecomposition on the transition matrix
    eigenvalues, eigenvectors = torch.linalg.eigh(transition_matrix)
    # Select top eigenvectors corresponding to the largest eigenvalues
    sorted_indices = torch.argsort(eigenvalues, descending=True)[:n_components]
    selected_eigenvalues = eigenvalues[sorted_indices]
    selected_eigenvectors = eigenvectors[:, sorted_indices]
    # Compute the diffusion map: lambda^t * phi
    diffusion_map = selected_eigenvectors * (selected_eigenvalues ** t)
    return diffusion_map


#!!! FIXME: not currently functioning properly - deal with it later and use KDE for testing for now

class DiffusionMap:
    def __init__(self, gamma=None, alpha=0.5, n_components: int|None =None, diffusion_time=1, var_threshold=0.95, max_components=256):
        """ Initialize the DiffusionMap model.
            Parameters:
                gamma (float): Kernel bandwidth parameter; if None, it will be computed based on Welford's variance.
                alpha (float): Parameter for normalizing the Laplacian matrix; alpha = 0.5 gives normalized graph Laplacian.
                n_components (int or None): Number of diffusion map dimensions to keep. If None, `select_components` is used.
                diffusion_time (int): Diffusion time parameter, controls scale in the diffusion process.
                variance_threshold (float): Threshold for cumulative eigenvalue sum when selecting components (if n_components is None).
        """
        self.gamma = gamma
        self.alpha = alpha
        self.n_components = n_components
        self.max_components = max_components
        self.diffusion_time = diffusion_time
        self.variance_threshold = var_threshold
        self.embedding_ = None      # Store the diffusion map embedding
        self.eigenvalues_ = None
        self.eigenvectors_ = None
        self.degree_matrix_ = None  # Store degree for normalization in `transform`
        self.training_data_ = None  # Store training data for similarity calculations

    def _compute_gamma(self, X):
        """ Compute gamma based on Welford's variance for the entire dataset.
            Parameters:
                X (torch.Tensor): Input tensor of shape (n_samples, n_features).
        """
        variance = torch.var(X, unbiased=False)  # compute variance directly on the full dataset
        gamma_value = 1 / (2 * variance.sqrt().mean().item())  # Take the mean across features
        return gamma_value

    def _compute_affinity_matrix(self, X):
        """ Compute the affinity matrix using the RBF kernel.
            Parameters:
                X (torch.Tensor): Input tensor of shape (n_samples, n_features).
            Returns:
                torch.Tensor: Affinity matrix.
        """
        if self.gamma is None:
            self.gamma = self._compute_gamma(X)  # Compute gamma if not set
        pairwise_sq_dists = torch.cdist(X, X) ** 2
        K = torch.exp(-self.gamma * pairwise_sq_dists)
        return K

    def _compute_laplacian(self, K):
        """ Compute the normalized Laplacian matrix.
            Parameters:
                K (torch.Tensor): Affinity matrix.
            Returns:
                torch.Tensor: Laplacian matrix.
        """
        # Compute the degree matrix with alpha normalization
        d = K.sum(dim=1)
        d_alpha = torch.pow(d, -self.alpha)
        L_alpha = (d_alpha[:, None] * K) * d_alpha[None, :]
        # renormalize the Laplacian matrix for the diffusion process
        d_inv = torch.pow(L_alpha.sum(dim=1), -1)
        M = d_inv[:, None] * L_alpha
        return M, d  # Returning degree matrix d for use in transform

    def select_components(self, eigenvalues):
        """ Select the number of components to retain the specified variance threshold. """
        cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        num_components = np.searchsorted(cumulative_variance, self.variance_threshold) + 1
        return num_components

    def get_important_patches(self, patch_size=16):
        """ Get most important patches based on the selected principal components. """
        if not hasattr(self, 'selected_indices_'):
            raise ValueError("Principal components must be selected first. Fit the model before calling this method.")
        patch_area = 16**2  # Area of a 16x16 patch
        # Create a sparse Boolean tensor of shape (256, 16, 16)
        bool_tensor = torch.zeros((256, patch_size, patch_size), dtype=torch.bool)
        # Convert flat indices back to (256, 16, 16) shape
        for idx in self.selected_indices_:
            channel_idx = idx // patch_area
            spatial_idx = idx % patch_area
            row = spatial_idx // patch_size
            col = spatial_idx % patch_size
            bool_tensor[channel_idx, row, col] = True
        # Average across the channel dimension to create a (16, 16) visualization
        importance_map = bool_tensor.float().mean(dim=0)
        return importance_map

    def reduce_features(self, X):
        if self.n_components is None:
            X_reshaped = X.reshape(X.size(0), 256, 256)
            eigenvalues = torch.linalg.eigvals(X_reshaped).numpy()
            self.n_components = self.select_components(eigenvalues)
            print("number of components retained: ", self.n_components)
        X = X.transpose(0, 1)
        if X.dim() > 2:
            X = X.reshape(X.size(0), -1)
        return X

    def fit_transform(self, X):
        """ Compute the diffusion map embedding.
            Parameters:
                X (torch.Tensor): Input tensor of shape (n_samples, n_features).
            Returns:
                torch.Tensor: Diffusion map embedding of shape (n_samples, n_components).
        """
        #X = self.reduce_features(X)
        #print("X shape: ", X.shape)
        if X.ndim > 2:
            X = X.reshape(X.size(0), -1)  # Flatten spatial dimensions
        self.training_data_ = X.cpu().numpy()
        affinity_matrix = self._compute_affinity_matrix(X)
        laplacian_matrix, degree_matrix = self._compute_laplacian(affinity_matrix)
        #print("laplacian matrix shape: ", laplacian_matrix.shape)
        #print("degree matrix shape: ", degree_matrix.shape)
        #print("laplacian nans: ", torch.isnan(laplacian_matrix).sum())
        #print("laplacian infs: ", torch.isinf(laplacian_matrix).sum())
        #print("laplacian zeros: ", torch.sum(laplacian_matrix == 0))
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian_matrix)
        # laplacian_matrix_np = laplacian_matrix.cpu().numpy()
        # max_components = self.max_components if self.n_components is None else self.n_components + 1
        # eigenvalues, eigenvectors = eigsh(laplacian_matrix_np, k=max_components, which="LM")
        # # If n_components is None, select optimal number based on variance threshold
        eigenvalues = eigenvalues.numpy()
        eigenvectors = eigenvectors.numpy()
        # Select the top `n_components` eigenvalues
        if self.n_components is None:
            self.n_components = self.select_components(eigenvalues)
        idx = np.argsort(eigenvalues)[-self.n_components:]
        # Sort and select the appropriate eigenvalues/vectors
        idx = np.argsort(eigenvalues)[:self.n_components]
        self.eigenvalues_ = eigenvalues[idx]
        self.eigenvectors_ = eigenvectors[:, idx]
        diffusion_map = self.eigenvectors_ * (self.eigenvalues_ ** self.diffusion_time)
        self.embedding_ = torch.tensor(diffusion_map, dtype=torch.float32)
        #print("embedding shape: ", self.embedding_.shape)
        self.degree_matrix_ = degree_matrix  # Save degree matrix for transform
        #print("degree matrix shape: ", self.degree_matrix_.shape)
        return self.embedding_

    def fit(self, X):
        """ Fit the model to data without returning the transformed output.
            Parameters:
                X (torch.Tensor): Input feature tensor.
        """
        embedding = self.fit_transform(X)
        return embedding

    def transform(self, X_new):
        """ Project new data into the diffusion map space.
            Parameters:
                X (torch.Tensor): Input feature tensor.
            Returns:
                torch.Tensor: Projected data in diffusion map space.
        """
        if self.embedding_ is None:
            raise ValueError("The model must be fit before calling transform.")
        if X_new.ndim > 2:
            X_new = X_new.reshape(X_new.size(0), -1)  # Flatten spatial dimensions
        # Step 1: Compute similarities between X_new and training data
        X_new_np = X_new.cpu().numpy()
        affinity_new = torch.exp(-self.gamma * torch.cdist(X_new, torch.tensor(self.training_data_)))
        # Step 2: Normalize affinity matrix based on the degree matrix from the original embedding
        d_alpha_inv = torch.pow(self.degree_matrix_, -self.alpha)
        L_alpha_new = affinity_new * d_alpha_inv[None, :]
        # Step 3: Project the new data onto the diffusion map space
        diffusion_embedding_new = torch.matmul(L_alpha_new, torch.tensor(self.eigenvectors_, dtype=torch.float32))
        diffusion_embedding_new *= torch.tensor(self.eigenvalues_, dtype=torch.float32) ** self.diffusion_time
        return diffusion_embedding_new