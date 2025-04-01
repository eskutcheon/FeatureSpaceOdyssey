from abc import ABC, abstractmethod
import torch
from typing import Any, Dict, Optional, Union



class ModelerBase(ABC):
    """ Abstract base class for all feature modeling classes which provides a uniform interface
        for models like KDE, Diffusion Maps, Normalizing Flows, GMMs, etc.
        Functionality:
            - fitting to a feature set
            - scoring new samples
            - saving modeled representations
            - loading from saved representations
            - optionally exposing latent variables or cluster assignments
    """

    @abstractmethod
    def fit(self, features: torch.Tensor) -> None:
        """ Fit the model to the given features (e.g., [N, D]) """
        pass

    # @abstractmethod
    # def transform(self, data: Union[torch.Tensor, DataLoader]) -> torch.Tensor:
    #     pass

    @abstractmethod
    def score(self, samples: torch.Tensor) -> torch.Tensor:
        """ Return a score for each sample (e.g., log-prob, distance, etc.) """
        pass

    # def save(self, path: str) -> None:
    #     """ Save the model to disk. Default assumes torch serialization; override for other libraries """
    #     torch.save(self, path)

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    # @classmethod
    # def load_from_path(cls, path: str) -> 'ModelerBase':
    #     """ Load a previously saved model. Override for different backends (e.g., sklearn joblib) """
    #     # TODO: add error handling for file not found, etc
    #     # TODO: add other arguments device mapping
    #     return torch.load(path, weights_only=True)

    @staticmethod
    @abstractmethod
    def load(path: str) -> 'ModelerBase':
        pass


    def get_metadata(self) -> Optional[Dict[str, Any]]:
        """ Return optional metadata, e.g., cluster assignments or hyperparameters """
        return None

    @abstractmethod
    def to(self, device: Union[str, torch.device]) -> 'ModelerBase':
        """ Optional: move model tensors to a specific device """
        pass


# -------------------------------------------------------------------------
# ModelContainer wraps a FeatureModel instance for saving/loading/usage
# -------------------------------------------------------------------------

class ModelContainer:
    """ Container for managing a feature model instance, including:
        - fitting the model
        - saving modeled features
        - restoring from disk
        - keeping track of source paths / metadata / parameters
    """

    def __init__(self, model: ModelerBase, model_name: str):
        self.model = model
        self.name = model_name
        self.fitted = False
        self.metadata: Dict[str, Any] = {}

    def fit_and_save(self, features: torch.Tensor, save_path: str) -> None:
        """ Fit the model and save it to disk """
        self.model.fit(features)
        self.model.save(save_path)
        self.fitted = True

    def load_model(self, path: str) -> None:
        """ Load model from disk """
        self.model = self.model.__class__.load_from_path(path)
        self.fitted = True

    def score_samples(self, samples: torch.Tensor) -> torch.Tensor:
        """ Score input samples using the model """
        assert self.fitted, "Model must be fit or loaded before scoring."
        return self.model.score(samples)

    def get_model_metadata(self) -> Optional[Dict[str, Any]]:
        return self.model.get_metadata()