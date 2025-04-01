from typing import Dict, List, Optional, Tuple, Union, Literal
import torch
import torch.nn.functional as F
#& in new project, rethink whether to keep this since overall, they're not very complex implementations
#from pytorch_ood.detector import EnergyBased, Entropy #NegativeEnergy


#! might rename this whole file to prob_scores or something - they're all probability or divergence measures


def compute_energy(logits: torch.Tensor, temperature: float = 1.0):
    return -temperature * torch.logsumexp(logits / temperature, dim=1)

def compute_entropy(logits: torch.Tensor):
    #? NOTE: second term using log_softmax is more numerically stable than
        # p = logits.softmax(dim=1) followed by -(p * p.log()).sum(dim=1)
    return -(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1)).sum(dim=1)


#& updated from the older version to take 2 general probability distribution tensors rather than single logits
def compute_kl_divergence(P: torch.Tensor, Q: torch.Tensor, reduction: Literal["none", "mean", "batchmean", "sum"] = "none"):
    """ Compute KL Divergence between two distributions P and Q. """
    # Convert logits to log probabilities
    log_P = F.log_softmax(P, dim=1) # F.kl_div expects log-probability inputs
    log_Q = F.log_softmax(Q, dim=1)  # Use softmax followed by log for Q
    # Compute KL divergence with specified reduction
    kl_div = F.kl_div(log_P, log_Q, reduction=reduction, log_target=True)
    return kl_div


#& updated from the older version to make clear that it compares logits to a uniform distribution only
def compute_kl_divergence_from_uniform(logits: torch.Tensor, reduction: Literal["none", "mean", "batchmean", "sum"] = "none"):
    # predicted log probabilities (logits transformed with log_softmax)
    log_probs = F.log_softmax(logits, dim=1) # logits to log-probabilities - same as log(softmax(logits))
    # define a uniform target distribution to get the divergence between uniform probabilities and predicted probabilities
    num_classes = logits.size(1)
    uniform_dist = torch.full_like(log_probs, 1.0 / num_classes) # no softmax since they're already probabilities
    # compute KL divergence with specified reduction, e.g. "batchmean" for averaging over batch or "sum" for summing over batch
    kl_div = F.kl_div(log_probs, uniform_dist, reduction=reduction, log_target=False)
    return kl_div


#& previously was `get_kde_log_likelihoods` - now 
def get_kde_log_likelihoods(test_feats: torch.Tensor, kde_centroids: torch.Tensor, bandwidths: torch.Tensor) -> torch.Tensor:
    """ computes the log-likelihood for each test sample under a kernel-based distribution (centroids plus bandwidth),
        giving us how well each sample fits the local kernel approximations' distributions.
    """
    TOL = 1e-8  # small value to avoid division by zero in bandwidth calculations
    likelihoods = []
    # helper function for faster list comprehension
    def _compute_lle(x, mu, bw):
        """ Compute log-likelihood for a single test sample x given a centroid mu and bandwidth bw. """
        diff = (x - mu) / (bw + TOL)
        # log likelihood contribution of diff found by summing the negative half of the squared distance
            # and subtracting a term that accounts for the bandwidth’s effect in the log scale
        return -0.5 * torch.sum(diff ** 2) - torch.log(bw + TOL) * mu.numel()
    # iterate over each test sample and compute its log-likelihood under the KDE model
    for x in test_feats:
        logs = [_compute_lle(x, mu, bw) for mu, bw in zip(kde_centroids, bandwidths)]
        likelihoods.append(torch.mean(torch.stack(logs)).item())
    return likelihoods




# TODO: salvage some of the old OOD testing functions that used dataloaders and try to use some of the Running Average code from the old utils
#& old functions never imported include:
#& `evaluate_ood_on_test`, `get_energy_scores_from_source`, `compute_running_median_over_energies`, `compute_confidence_interval_over_energies`, `get_energy_median_from_source`
#& OOD evaluation logic (e.g., running tests over multiple checkpoints) should live in `evaluation/ood_evaluators.py`



#& previously located in eval_utils.py in the old project - updated to accept torch.Tensor directly
#TODO: replace with call to compute_kl_divergence - but ensure they're logically equivalent in a regression test first
# def compute_kl_divergence_of_lle(source_likelihoods: np.ndarray, test_likelihoods: np.ndarray) -> float:
#     """ Compute KL Divergence between source and test log likelihood distributions. """
#     from scipy.stats import entropy
#     # Convert likelihoods to probabilities using softmax to ensure non-negativity
#     source_probs = F.softmax(torch.tensor(source_likelihoods), dim=0)
#     test_probs = F.softmax(torch.tensor(test_likelihoods), dim=0)
#     # Convert to numpy arrays for scipy entropy function
#     source_probs = source_probs.detach().cpu().numpy()
#     test_probs = test_probs.detach().cpu().numpy()
#     # Compute KL divergence (test distribution relative to source distribution)
#     """
#         If only probabilities pk are given, the Shannon entropy is calculated as H = -sum(pk * log(pk)).
#         If qk is not None, then compute the relative entropy D = sum(pk * log(pk / qk)).
#             This quantity is also known as the Kullback-Leibler divergence.
#     """
#     kl_div = entropy(test_probs, source_probs)
#     return kl_div