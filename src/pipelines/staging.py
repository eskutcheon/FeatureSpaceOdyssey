from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple
import torch
from src.data.loaders import create_image_dataloader




class PipelineStage(ABC):
    """ Each pipeline stage implements a 'transform' method that takes some input and returns transformed output """
    @abstractmethod
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        pass


class LoadFeaturesStage(PipelineStage):
    def __init__(self, path: str):
        self.path = path

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        # ignoring input data, just load from self.path
        # or read from disk
        loaded = torch.load(self.path)
        return loaded


class NormalizeStage(PipelineStage):
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        # example: normalize by mean & std
        return (data - data.mean()) / (data.std() + 1e-6)


class SomeClusteringStage(PipelineStage):
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        # do clustering
        # return e.g. cluster labels or a centroid-based representation
        return data  # placeholder


# universal pipeline driver:
class Pipeline:
    def __init__(self, stages):
        self.stages = stages

    def run(self, init_data: torch.Tensor = None) -> torch.Tensor:
        out = init_data
        for stage in self.stages:
            out = stage.transform(out)
        return out

########### usage example: ############
# stages = [
#     LoadFeaturesStage("my_features.pt"),
#     NormalizeStage(),
#     SomeClusteringStage()
# ]

# pipeline = Pipeline(stages)
# final_output = pipeline.run(None)



# class ImageLoaderStage(PipelineStage):
#     """ A pipeline stage that loads images from disk using the data loaders. `transform()` returns a PyTorch DataLoader (or possibly the dataset) """
#     def __init__(
#         self,
#         base_dir: str,
#         batch_size: int = 8,
#         out_size: Union[int, Tuple[int,int]] = (512, 512),
#         use_representative: bool = False,
#         original_only: bool = False,
#         original_size: Optional[int] = None,
#         device: str = "cpu"
#     ):
#         self.base_dir = base_dir
#         self.batch_size = batch_size
#         self.out_size = out_size
#         self.use_representative = use_representative
#         self.original_only = original_only
#         self.original_size = original_size
#         self.device = device

#     def transform(self, data) -> torch.utils.data.DataLoader:
#         """ pipeline stage usually ignores `data` if it's the initial stage, but we could combine `data` with self.base_dir logic if needed """
#         dataloader = create_image_dataloader(
#             base_dir=self.base_dir,
#             batch_size=self.batch_size,
#             out_size=self.out_size,
#             use_representative=self.use_representative,
#             original_only=self.original_only,
#             original_size=self.original_size,
#             device=self.device,
#         )
#         return dataloader
