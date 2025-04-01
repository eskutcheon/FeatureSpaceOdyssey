
from .modeler_base import ModelerBase
from .modelers.kde import MixtureOfKDEsModeler
from .modelers.spectral import DiffusionMapModeler
#from src.models.manifold import DiffusionMapModel  # hypothetical future model
# Add imports for additional models here

MODEL_REGISTRY = {
    "mokde": MixtureOfKDEsModeler,
    "diffusion": DiffusionMapModeler,
    # "flow": NormalizingFlowModel,
    # "gmm": GaussianMixtureModel,
}

def create_model(name: str, **kwargs) -> ModelerBase:
    """ Return  instance of a feature model given its name.
        Also allows config-driven workflows to dynamically create modelers.
    """
    if name.lower() in MODEL_REGISTRY:
        return MODEL_REGISTRY[name.lower()](**kwargs)
    raise ValueError("Unknown modeler name.")
