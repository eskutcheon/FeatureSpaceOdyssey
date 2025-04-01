from abc import ABC, abstractmethod
from typing import Any, Dict
import torch
# local imports
from src.models.modeler_base import ModelerBase
from src.hypothesis.hypothesis_base import HypothesisTest

""" TO CHANGE:
- [x] Keep `EvaluatorBase`, adapt to `Template Method` style (e.g., `load_data_hook`, `extract_features_hook`, `compute_results_hook`).
- [ ] Move each old evaluator into its own subclass:
    - `EnergyEvaluator`, `KDEEvaluator`, `EntropyEvaluator`, etc.
- Remove legacy `ModelEvaluator` (from old code)
"""


# class BaseEvaluator(ABC):
#     """ Template Method pattern: define 'evaluate()' with a fixed skeleton, calling abstract hooks so subclasses can override details """
#     def evaluate(self, data: Any) -> dict:
#         # 1. Preprocess or load data
#         loaded = self.load_data_hook(data)
#         # 2. Extract features
#         features = self.extract_features_hook(loaded)
#         # 3. Compute metrics or do hypothesis test
#         results = self.compute_results_hook(features)
#         # Possibly more steps...
#         return results

#     @abstractmethod
#     def load_data_hook(self, data: Any):
#         pass

#     @abstractmethod
#     def extract_features_hook(self, loaded_data: Any):
#         pass

#     @abstractmethod
#     def compute_results_hook(self, features: Any) -> dict:
#         pass

class EvaluatorBase(ABC):
    """ Template method base class for evaluating a modeler and test combination. """
    def evaluate(self, source_data: Any, test_data: Any) -> Dict[str, Any]:
        source_features = self.load_data(source_data)
        test_features = self.load_data(test_data)
        self.modeler.fit(source_features)
        scores_source = self.modeler.score_samples(source_features)
        scores_test = self.modeler.score_samples(test_features)
        results = self.test.run_test(scores_source, scores_test)
        return results

    @abstractmethod
    def load_data(self, data: Any):
        pass


#& previously used BaseModelEvaluator, KDEEvaluator, and HypothesisTestRunner classes in hypothesis_testing.py

class KDEEvaluator(EvaluatorBase):
    def __init__(self, modeler: ModelerBase, test: HypothesisTest):
        self.modeler = modeler
        self.test = test

    def load_data(self, data: Any):
        if isinstance(data, torch.utils.data.DataLoader):
            all_data = [batch for batch in data]
            return torch.cat(all_data, dim=0)
        return torch.as_tensor(data)  # fallback for direct tensors
