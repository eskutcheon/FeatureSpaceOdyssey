import numpy as np
import torch

#& imported from the old project's src/evaluate/manifold_modeling.py
#!!! FIXME: not sure if I ever even got this working - need to update for the current codebase

class KernelPCA:
    def __init__(self, n_components, kernel='rbf', gamma=None, degree=3, coef0=1, device='cuda'):
        # Initialize kernel PCA parameters and placeholders for incremental fitting
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.device = device
        # Initialize placeholders for kernel matrix, eigenvectors, eigenvalues, and fitted data
        self.alphas = None
        self.lambdas = None
        self.X_fit_ = None
        self.K_accumulated = None
        self.fitted = False


    def compute_kernel_matrix(self, X, Y=None):
        if Y is None:
            Y = X
        if self.kernel == 'linear':
            K = torch.matmul(X, Y.transpose(-1, -2))
        elif self.kernel == 'rbf':
            gamma = self.gamma or 1.0 / X.size(1)
            # If batch sizes differ, compute the RBF kernel in smaller chunks
            if X.size(0) != Y.size(0):
                K_chunks = []
                # FIXME: needs to be nested in the reverse order with abstracted kernel functions
                for y_chunk in torch.split(Y, split_size_or_sections=1, dim=0):  # Split Y along batch dimension
                    K_chunk = torch.exp(-gamma * torch.cdist(X, y_chunk.squeeze(0)) ** 2)
                    K_chunks.append(K_chunk)
                    self.update_accumulated_kernel_matrix(K_chunk)
                K = torch.cat(K_chunks, dim=1)  # Concatenate along the column dimension
            else:
                # Compute RBF kernel when batch sizes match
                K = torch.exp(-gamma * torch.cdist(X, Y) ** 2)
        elif self.kernel == 'poly':
            K = (torch.matmul(X, Y.transpose(-1, -2)) + self.coef0) ** self.degree
        else:
            raise ValueError("Unsupported kernel type.")
        return K

    def center_kernel_matrix(self, K):
        H, W = K.shape[-2:]
        flat_dim = H*W
        M1 = torch.ones((H, W), device=self.device)/flat_dim # previously one_n
        print("M1 shape: ", M1.shape)
        M2 = M1 @ K
        M1.fill_diagonal_((1-flat_dim)/flat_dim)
        # [x] FIXCHANGE: ~~this could almost certainly be done more efficiently - see if a matrix factorization might apply~~
            # UPDATE: went from 8n^3 - n^2 flops to 4n^3 - n^2 + n flops with this refactor
        # K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
        K_centered = (M2 - K) @ M1
        return K_centered


    def update_accumulated_kernel_matrix(self, K_new):
        # If this is the first batch, initialize K_accumulated
        if self.K_accumulated is None:
            self.K_accumulated = K_new
        else:
            print("K_accumulated shape: ", self.K_accumulated.shape)
            # Combine the new kernel matrix with accumulated matrix for an average
            # FIXCHANGE: faulty averaging - should be a running average
            self.K_accumulated = (self.K_accumulated + K_new) / 2

    def fit(self, X):
        # Fit the kernel PCA model on a single batch, initializing the kernel matrix
        X = X.to(self.device)
        K = self.compute_kernel_matrix(X)
        print("K shape: ", K.shape)
        K_centered = self.center_kernel_matrix(K)
        # ? NOTE: needs to be modified for multiple batches of a large dataset
        # eigenvalue decomposition - switch to streaming eigendecomposition for large datasets
        eigvals, eigvecs = torch.linalg.eigh(K_centered)
        # flip to descending order to get largest principal components first
        eigvals, eigvecs = eigvals.flip(-1), eigvecs.flip(-1)
        # store the top eigenvalues and eigenvectors
        self.alphas = eigvecs[:, :self.n_components]
        self.lambdas = eigvals[:self.n_components]
        self.X_fit_ = X
        self.fitted = True


    def partial_fit(self, X):
        # Enable incremental fitting for large datasets by updating the kernel matrix incrementally
        X = X.to(self.device)
        print("X shape: ", X.shape)
        K_new = self.compute_kernel_matrix(X, self.X_fit_) if self.X_fit_ is not None else self.compute_kernel_matrix(X)
        print("K_new shape: ", K_new.shape)
        K_new_centered = self.center_kernel_matrix(K_new)
        print("K_new_centered shape: ", K_new_centered.shape)
        if self.K_accumulated is None:
            self.update_accumulated_kernel_matrix(K_new)
        # Perform eigenvalue decomposition on the accumulated kernel matrix
        eigvals, eigvecs = torch.linalg.eigh(self.K_accumulated)
        eigvals, eigvecs = eigvals.flip(-1), eigvecs.flip(-1)
        # Update the top components based on incremental eigen decomposition
        self.alphas = eigvecs[:, :self.n_components]
        self.lambdas = eigvals[:self.n_components]
        self.X_fit_ = X if self.X_fit_ is None else torch.cat([self.X_fit_, X], dim=0)
        print("X_fit_ shape: ", self.X_fit_.shape)
        self.fitted = True


    def transform(self, X):
        # Transform new data based on the principal components
        if not self.fitted:
            raise RuntimeError("The model has not been fitted. Please call `fit` or `partial_fit` first.")
        K = self.compute_kernel_matrix(X, self.X_fit_)
        return (K @ self.alphas) / torch.sqrt(self.lambdas)

    def fit_transform(self, X):
        # Convenience method to fit the model and immediately transform the input data
        self.fit(X)
        return self.transform(X)



#------------------ misc functions for dimensionality reduction that were mostly unused ------------------#

def welfords_variance(x, n, mu, m2):
    """Welford's variance calculation, numerically stable and designed to prevent inf values."""
    n += 1
    delta = x - mu
    mu += delta / n
    delta2 = x - mu
    m2 += delta * delta2
    return n, mu, m2

def rbf_kernel_torch(X, Y, gamma):
    pairwise_sq_dists = torch.cdist(X, Y) ** 2
    K = torch.exp(-gamma * pairwise_sq_dists)
    return K

def compute_gamma(data_loader):
    """Compute gamma based on Welford's variance for the entire dataset."""
    count, mean, M2 = 0, 0.0, 0.0
    for batch in data_loader:
        batch = batch['img']  # Assuming each batch contains an "img" key
        batch = batch.view(batch.size(0), -1) #.cpu().numpy()  # Flatten spatial dims
        # for x in batch:
        count, mean, M2 = welfords_variance(batch, count, mean, M2)  # Update count, mean, and M2
    # Final variance and gamma computation
    variance = M2 / (count - 1)
    gamma = 1 / (2 * variance)
    return gamma


def compute_gamma_torch(X):
    variance = torch.var(X, unbiased=False)  # compute variance directly on the full dataset
    gamma_value = 1 / (2 * variance.sqrt().mean().item())  # Take the mean across features
    return gamma_value


def select_components(eigenvalues, threshold=0.90):
    """ Select the number of components that capture a given percentage of eigenvalue sum.
        Parameters:
            eigenvalues (np.ndarray): Array of eigenvalues sorted in descending order.
            threshold (float): Cumulative sum threshold (e.g., 0.90 for 90%).
        Returns:
            int: Optimal number of components.
    """
    total_sum = np.sum(eigenvalues)
    cumulative_sum = np.cumsum(eigenvalues) / total_sum
    num_components = np.searchsorted(cumulative_sum, threshold) + 1
    return num_components