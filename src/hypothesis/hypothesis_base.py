from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Union, Literal, Any
from dataclasses import dataclass
import numpy as np
import torch
from scipy.stats import ks_2samp, wilcoxon



#& currently the exact same as the hypothesis_base.py in the old project

# -----------------------------
# Unified result container
# -----------------------------
@dataclass
class HypothesisTestResult:
    stat: float
    p_value: float
    result: Literal["Reject H0", "Fail to reject H0"]
    test_name: str
    model_pair: Optional[Tuple[str, str]] = None
    additional: Optional[Dict[str, Union[str, float]]] = None


# -----------------------------
# Base Hypothesis Test
# -----------------------------
class HypothesisTest(ABC):
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    @abstractmethod
    def compare(self, group1: np.ndarray, group2: np.ndarray, model_pair: Optional[Tuple[str, str]] = None) -> HypothesisTestResult:
        pass


# -----------------------------
# KS Test Strategy
# -----------------------------
class KSTest(HypothesisTest):
    def __init__(self, alpha: float = 0.05, method: str = "exact"):
        super().__init__(alpha)
        self.method = method

    def compare(self, group1: np.ndarray, group2: np.ndarray, model_pair=None) -> HypothesisTestResult:
        """ Perform KS test between group1 and group2 scores
            Kolmogorov-Smirnov test: https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
        """
        stat, pval = ks_2samp(group1, group2, method=self.method)
        result = "Reject H0" if pval < self.alpha else "Fail to reject H0"
        return HypothesisTestResult(stat, pval, result, "KS Test", model_pair)


# -----------------------------
# Wilcoxon Test Strategy
# -----------------------------
class WilcoxonTest(HypothesisTest):
    def __init__(self, alpha: float = 0.05, alternative: str = "greater"):
        super().__init__(alpha)
        self.alternative = alternative

    def compare(self, group1: np.ndarray, group2: np.ndarray, model_pair=None) -> HypothesisTestResult:
        stat, pval = wilcoxon(group1, group2, alternative=self.alternative)
        result = "Reject H0" if pval < self.alpha else "Fail to reject H0"
        return HypothesisTestResult(stat, pval, result, "Wilcoxon Signed-Rank", model_pair)


# -----------------------------
# Generalized Bootstrap Wrapper
# -----------------------------
class BootstrappedTest(HypothesisTest):
    def __init__(
        self,
        base_test: HypothesisTest,
        n_bootstraps: int = 100,
        sample_size: int = 36,
        alpha: float = 0.05,
        seed: Optional[int] = None
    ):
        super().__init__(alpha)
        self.base_test = base_test
        self.n_bootstraps = n_bootstraps
        self.sample_size = sample_size
        self.seed = seed

    # TODO: may need a kwargs argument for the base_test to pass additional parameters
    def compare(self, group1: np.ndarray, group2: np.ndarray, model_pair=None) -> HypothesisTestResult:
        rng = np.random.default_rng(self.seed)
        stats, pvals = [], []
        for _ in range(self.n_bootstraps):
            subset = rng.choice(group1, self.sample_size, replace=False)
            result = self.base_test.compare(subset, group2)
            stats.append(result.stat)
            pvals.append(result.p_value)
        avg_stat = float(np.mean(stats))
        avg_pval = float(np.mean(pvals))
        result = "Reject H0" if avg_pval < self.alpha else "Fail to reject H0"
        return HypothesisTestResult(avg_stat, avg_pval, result, f"Bootstrapped {self.base_test.test_name}", model_pair, {"bootstraps": self.n_bootstraps})



##############################################################################
# Factory for creating tests by name
##############################################################################
class HypothesisTestFactory:
    @staticmethod
    def create_test(test_name: str, **kwargs) -> HypothesisTest:
        if test_name.lower() == "ks":
            return KSTest(**kwargs)
        elif test_name.lower() == "wilcoxon":
            return WilcoxonTest(**kwargs)
        else:
            raise ValueError(f"Unknown test_name '{test_name}'")


HYPOTHESIS_TESTS = {
    "ks": KSTest,
    "wilcoxon": WilcoxonTest,
    #"mmd": MMDTest,
    # ...
}