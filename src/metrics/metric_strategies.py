from abc import ABC, abstractmethod
import torch

##############################################################################
# Strategy interface for metrics
##############################################################################
class Metric(ABC):
    @abstractmethod
    def compute_metric(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Compute a scalar metric from predictions and targets.
        Could also apply to 'source_features' vs. 'test_features', etc.
        """
        pass

##############################################################################
# Concrete metric strategies
##############################################################################

class IoUMetric(Metric):
    def compute_metric(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        # minimal placeholder
        return 0.0


class MMDMetric(Metric):
    def compute_metric(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        # interpret preds and targets as two sets of features
        # compute MMD
        return 0.0


class MahalanobisMetric(Metric):
    def compute_metric(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        # interpret preds and targets as two sets of features
        # compute Mahalanobis distance
        return 0.0
