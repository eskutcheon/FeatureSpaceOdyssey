from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, Literal
import os
import json
from math import sqrt, floor
import heapq
import torch
import re
import torchvision.transforms.v2 as TT
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN



# might end up being useless but it's fairly straightforward and wraps a really common operation
class EnsureBatched(torch.nn.Module):
    """ Composable wrapper to ensure that the input tensor is batched before passing it to any other nn.Module instances """
    def __init__(self, expected_ndims: int):
        super().__init__()
        self.expected_ndims: int = expected_ndims

    def forward(self, x):
        assert x.dim() <= self.expected_ndims, f"Input tensor has too many dimensions. Expected {self.expected_ndims} but got {x.dim()}."
        while x.dim() < self.expected_ndims:
            x = x.unsqueeze(0)
        return x


#& small TSNE functions ported from the old eval_utils.py - needs a lot of updates, but they're not coupled with loaders at least


def compute_tsne_features(train_features, test_features, perplexity=30, n_components=2, learning_rate=200):
    """ Compute t-SNE embeddings for training and test features.
        Args:
            train_features: Numpy array of training dataset features.
            test_features: Numpy array of test dataset features.
            perplexity: t-SNE perplexity parameter.
            n_components: Number of t-SNE components (usually 2 or 3).
            learning_rate: t-SNE learning rate.
    """
    # Combine training and test features for t-SNE
    all_features = np.concatenate([train_features, test_features], axis=0)
    # Fit t-SNE
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        early_exaggeration=36,
        learning_rate=learning_rate,
        max_iter=1000,
        metric="mahalanobis"
    )
    tsne_results = tsne.fit_transform(all_features)
    # Split back into training and test t-SNE embeddings
    tsne_train = tsne_results[:train_features.shape[0], :]
    tsne_test = tsne_results[train_features.shape[0]:, :]
    #? NOTE: previously combined with the plotting functionality - split into plot_utils.plot_tsne (function expected a downsampling string label)
    return tsne_train, tsne_test





# TODO: determine whether to keep this or not - used in the old project for plotting t-SNE with DBSCAN clustering
def plot_tsne_with_dbscan(train_features, test_features, downsampling, n_components=2, perplexity=30,
                          learning_rate=200, eps=0.5, min_samples=5, max_clusters=20
):
    """ Compute and plot t-SNE with DBSCAN clustering. """
    all_features = np.concatenate([train_features, test_features], axis=0)
    tsne = TSNE(n_components=n_components, early_exaggeration=6, perplexity=perplexity, learning_rate=learning_rate, max_iter=1200, metric="cosine")
    tsne_results = tsne.fit_transform(all_features)
    tsne_train, tsne_test = tsne_results[:train_features.shape[0]], tsne_results[train_features.shape[0]:]
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    train_cluster_labels = dbscan.fit_predict(tsne_train)
    #plot_util.plot_tsne_clusters(tsne_train, tsne_test, train_cluster_labels, max_clusters)





