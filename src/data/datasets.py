
import os
import h5py
from typing import List, Dict, Optional, Union, Tuple, Callable
import torch
import numpy as np
from torch.utils.data import Dataset #, DataLoader
#from PIL import Image  # or from torchvision import transforms if you want to transform



#!##########################################################################################################
#! REPLACE WITH LogitsDataset - no longer dealing with images directly
    #! - though we may still need to pass the masks to compare results with ground truth - not sure yet
#!##########################################################################################################

# class ImageDataset(Dataset):
#     """ Basic dataset for loading images (and optionally masks) from directories. """


#!##########################################################################################################


class FeatureDataset(Dataset):
    """
        A generic dataset for precomputed features, logits, or metadata.
        - Supports .npy, .pt, or HDF5
        #! below suggestion won't work for generators or literally any open data streams because dataloaders use pickling
        - Future support: streaming via callable / generator
    """
    def __init__(
        self,
        data: Union[str, torch.Tensor, np.ndarray, Callable],
        key: str = "features",
        transform: Optional[Callable] = None
    ):
        super().__init__()
        self.key = key
        self.transform = transform
        self.data = self._load(data)

    def _load(self, data):
        if isinstance(data, (torch.Tensor, np.ndarray)):
            return torch.as_tensor(data)
        elif callable(data):
            #!! FIXME: needs to be revisited to actually consider streaming properly - reference the way I did it in MCAPST for video frames
            self.streaming = True
            return data  # Generator function or data streamer
        # TODO: review these calls to ensure they're loading properly, allow for setting the map_location, etc
        elif isinstance(data, str):
            if data.endswith(".pt"):
                return torch.load(data, weights_only=True)
            elif data.endswith(".npy"):
                return torch.from_numpy(np.load(data))
            elif data.endswith(".h5") or data.endswith(".hdf5"):
                with h5py.File(data, "r") as f:
                    return torch.from_numpy(f[self.key][...])
        raise ValueError(f"Unsupported data type or path: {type(data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx] if not callable(self.data) else self.data(idx)
        if self.transform:
            sample = self.transform(sample)
        return sample



class KDEModelDataset(Dataset):
    """ Example dataset for “KDE model” files that store cluster means/bandwidth or entire embeddings.
        Suppose we have .pt or .h5 files that store a 'kde_clusters' and 'bandwidths' or similar.
        Each item: { 'kde_clusters': <Tensor>, 'bandwidths': <Tensor>, 'path': <str> }
    """
    def __init__(self, kde_files: List[str], file_format: str = "pt"):
        self.kde_files = kde_files
        self.file_format = file_format

    def __len__(self):
        return len(self.kde_files)

    def __getitem__(self, idx):
        file_path = self.kde_files[idx]
        kde_data = self._load_kde_model(file_path)
        return {
            "kde_clusters": kde_data["kde_clusters"],
            "bandwidths": kde_data["bandwidths"],
            "path": file_path
        }

    def _load_kde_model(self, path: str) -> Dict[str, torch.Tensor]:
        """ Minimal logic for loading a file that has 2 keys: { 'kde_clusters', 'bandwidths' }. Adapt for your actual data structure. """
        if self.file_format == "pt":
            data = torch.load(path, map_location="cpu")  # e.g. data = { "kde_clusters":..., "bandwidths":... }
            # ensure these are Tensors
            clusters = data.get("kde_clusters", torch.tensor([]))
            bws = data.get("bandwidths", torch.tensor([]))
            return {"kde_clusters": clusters, "bandwidths": bws}
        # elif self.file_format == "h5":
        #   # read HDF5 and parse
        # fallback
        raise ValueError(f"Unsupported KDE file format: {self.file_format}")
