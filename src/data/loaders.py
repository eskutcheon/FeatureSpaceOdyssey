import os
import torch
from typing import Optional, List, Callable
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Optional, Tuple, Union
# local imports
from .datasets import FeatureDataset, KDEModelDataset


#! FIXME: will need to be updated for streaming capabilities
def get_loader(
    data_source: Union[str, torch.Tensor, np.ndarray, Callable],
    batch_size: int = 16,
    shuffle: bool = False,
    num_workers: int = 0,
    transform: Optional[Callable] = None
) -> DataLoader:
    """ Create a DataLoader from a source path, tensor, or generator """
    dataset = FeatureDataset(data_source, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)



# for both logits and feature dataloaders, I may actually want some way to stream data from model outputs in the main project

#!##########################################################################################################
#! REPLACE WITH create_logits_dataloader - no longer dealing with images directly
    #! - though we may still need to pass the masks to compare results with ground truth - not sure yet
#!##########################################################################################################

def create_image_dataloader(
    img_dir: str,
    batch_size: int,
    out_size: Union[int, Tuple[int,int]] = (512,512),
    mask_dir: Optional[str] = None,
    shuffle: bool = False,
    num_workers: int = 4
) -> DataLoader:
    """ Create a DataLoader for images + optional masks from disk """
    img_paths = sorted([os.path.join(img_dir, p) for p in os.listdir(img_dir) if p.endswith(".png")])
    mask_paths = None
    if mask_dir:
        mask_paths = sorted([os.path.join(mask_dir, p) for p in os.listdir(mask_dir) if p.endswith(".png")])
    #! use logits datasets
    dataset = ImageDataset(
        img_paths=img_paths,
        mask_paths=mask_paths,
        transform=None,  # or pass a custom transform
        out_size=out_size
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

#!##########################################################################################################


def create_feature_dataloader(
    feature_files: List[str],
    file_format: str = "pt",
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0
) -> DataLoader:
    """ Create a DataLoader for precomputed feature files. Typically batch_size=1 is enough if each file is a big array of shape (N,D). """
    dataset = FeatureDataset(feature_files, file_format=file_format)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader


def create_kde_dataloader(
    kde_files: List[str],
    file_format: str = "pt",
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0
) -> DataLoader:
    """ Create a DataLoader for saved KDE model data. Usually we just want each file as a single sample, so default is batch_size=1 """
    dataset = KDEModelDataset(kde_files, file_format=file_format)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader




# version of image dataloader that still uses the representative subset approach:
# def create_image_dataloader(
#     base_dir: str,
#     batch_size: int,
#     out_size: Union[int, Tuple[int,int]],
#     use_representative: bool = False,
#     original_only: bool = False,
#     original_size: Optional[int] = None,
#     # ...
# ):
#     img_dir = os.path.join(base_dir, "rgbImages")
#     mask_dir = os.path.join(base_dir, "gtLabels")
#     all_imgs = sorted([p for p in os.listdir(img_dir) if p.endswith(".png")])
#     all_masks = sorted([p for p in os.listdir(mask_dir) if p.endswith(".png")])
#     if use_representative:
#         # Insert your logic to pick only a subset, e.g. read metadata, filter sequences, etc.
#         pass
#     if original_only and original_size is not None:
#         all_imgs = all_imgs[:original_size]
#         all_masks = all_masks[:original_size]
#     # prepend full path
#     img_paths = [os.path.join(img_dir, f) for f in all_imgs]
#     mask_paths = [os.path.join(mask_dir, f) for f in all_masks]
#     dataset = ImageDataset(
#         img_paths=img_paths,
#         mask_paths=mask_paths,
#         out_size=out_size
#     )
#     return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
