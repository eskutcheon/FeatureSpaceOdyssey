from typing import Union, Tuple, Dict, List, Literal, Any
import heapq
import torch
import numpy as np


#& contents of this file is mostly from `eval_utils.py` in the old project

class RunningMedian:
    def __init__(self):
        # Two heaps: one for the lower half (max heap) and one for the upper half (min heap)
        self.lower_half = []  # Max-heap (invert values to simulate a max-heap)
        self.upper_half = []  # Min-heap

    def add(self, value):
        # Add a new value to the running median calculation
        if not self.lower_half or value <= -self.lower_half[0]:
            heapq.heappush(self.lower_half, -value)  # Add to max-heap (invert sign)
        else:
            heapq.heappush(self.upper_half, value)  # Add to min-heap
        # Balance the two heaps
        if len(self.lower_half) > len(self.upper_half) + 1:
            heapq.heappush(self.upper_half, -heapq.heappop(self.lower_half))
        elif len(self.upper_half) > len(self.lower_half):
            heapq.heappush(self.lower_half, -heapq.heappop(self.upper_half))

    def get_median(self):
        # Return the median value
        if len(self.lower_half) == len(self.upper_half):
            # If the two heaps are of equal size, return the average of the two middle values
            return (-self.lower_half[0] + self.upper_half[0]) / 2.0
        else:
            # If the heaps are not equal, the larger heap contains the median
            return -self.lower_half[0]


def get_shannon_entropy(features: torch.Tensor, TOL = 1e-6):
    probs = torch.softmax(features.flatten(2), dim=2)
    log_probs = torch.log(probs + TOL)  # Add small epsilon to avoid log(0)
    return -torch.sum(probs * log_probs, dim=2) #.mean(dim=2)  # Shannon entropy


def spatial_stat_summary(features: torch.Tensor) -> torch.Tensor:
    """ Compute the mean and standard deviation over the spatial dimensions (H, W) of extracted features
        - should work for plain images or intermediate features, where in the former case N=3
        Args:
            features: Tensor of shape (B, N, H, W)
        Returns:
            Feature summaries of shape (B, 5*N) containing first 4 central moments and the entropy for each feature channel
    """
    TOL = 1e-6  # Small value to avoid division by zero
    mean = features.mean(dim=(2, 3))
    std = features.std(dim=(2, 3))
    skewness = (features ** 3).mean(dim=(2, 3))/(std**3 + TOL)
    kurtosis = (features ** 4).mean(dim=(2, 3))/(std**4 + TOL) - 3
    entropy = get_shannon_entropy(features, TOL)
    # concatenate statistics along the feature dimension and return the summary tensor
    return torch.cat([mean, std, skewness, kurtosis, entropy], dim=1)


def filter_outliers_by_distance(features: np.ndarray, outlier_percent: float):
    """ Filter a percentage of extreme outliers from the feature set based on Euclidean distance from the mean.
        Args:
            features: NumPy array of features to filter (shape: [n_samples, n_features]).
            outlier_percent: Percentage of samples to drop (default: 5%).
        Returns:
            Filtered feature set with fewer outliers.
    """
    # Compute the Euclidean distance from each point to the mean of the feature set
    mean_vector = np.mean(features, axis=0)
    distances = np.linalg.norm(features - mean_vector, axis=1)
    # Determine the cutoff distance to filter out the top 'outlier_percent' of points
    cutoff = np.percentile(distances, 100 - outlier_percent)
    # Filter and retain only the points below the cutoff distance
    filtered_features = features[distances <= cutoff]
    return filtered_features

#& possibly unnecessary - combine with `filter_outliers_by_distance`
    # main difference is that its passed scores rather than raw features
def filter_low_percentile_scores(scores_dict: Dict[str, List[float]], percentile=0.01):
    """ Filter out scores below a certain percentile from each model's score list.
        Args:
            scores_dict: Dictionary where keys are model names and values are lists of scores.
            percentile: Percentile threshold to filter out low scores (default: 0.01).
        Returns:
            Dictionary with filtered scores for each model.
    """
    # Concatenate all scores from all models into a single array
    all_scores = np.concatenate(list(scores_dict.values()))
    # Determine the bottom percentile threshold
    bottom_percentile = np.percentile(all_scores, percentile)
    # Filter out scores below the bottom percentile for each model
    for model_name, scores in scores_dict.items():
        scores = [score for score in scores if score > bottom_percentile]
        scores_dict[model_name] = scores
    return scores_dict


#& eval_utils.py functions never imported:
#& `set_name_from_alias`, `update_histograms`, `extract_feature_patches`, `compute_kl_divergence_histograms`
#& `compute_feature_embeddings`, `compute_and_plot_metrics`, `standardize_cluster_key`

#& NOTE: `compute_kl_divergence_of_lle` moved to `metrics/ood_scores.py`