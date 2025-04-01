from typing import Union, Tuple, Dict, List, Literal, Any
import os
import json
import pandas as pd
import h5py
import torch
import torchvision.io as IO

######^ moved from evaluate.py

def save_tensor_to_hdf5_no_metadata(tensor, filepath):
    """ Saves a single PyTorch tensor to an HDF5 file without any metadata.
        Parameters:
        - filepath (str): The path where the tensor should be saved.
        - tensor (torch.Tensor): The tensor to save.
    """
    # Ensure the tensor is on CPU
    tensor = tensor.cpu()
    # Save the tensor to HDF5
    with h5py.File(filepath, 'w') as h5_file:
        h5_file.create_dataset('tensor', data=tensor.numpy(), compression='gzip')
    print(f"Tensor saved to {filepath} without metadata.")



# ~also might just want to make this a custom context manager to used instead of open or h5py.File
def save_tensor_chunks_to_hdf5(tensor_chunk, metadata, dest_dir, save_individually=False, json_fallback=False):
    """Factory function that calls either save_tensor_as_individual_files or save_tensor_in_single_file."""
    # Validate tensor and metadata
    if not isinstance(tensor_chunk, torch.Tensor):
        raise ValueError("The input must be a PyTorch tensor.")
    if len(tensor_chunk) != len(metadata):
        raise ValueError("The length of tensor_chunk must match the length of metadata.")
    # Ensure the tensor is on CPU
    tensor_chunk = tensor_chunk.cpu()
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    if save_individually:
        save_tensor_as_individual_files(tensor_chunk, metadata, dest_dir, json_fallback)
    else:
        save_tensor_in_single_file(tensor_chunk, metadata, dest_dir, json_fallback)


def save_tensor_as_individual_files(tensor_chunk, metadata, dest_dir, json_fallback):
    """Saves each tensor along the batch dimension separately as individual HDF5 files."""
    for idx, (tensor, meta) in enumerate(zip(tensor_chunk, metadata)):
        filename = meta["filename"]  # Get filename from metadata
        filepath = os.path.join(dest_dir, filename)
        try:
            save_tensor_to_hdf5(filepath, tensor, meta)
            print(f"Tensor saved to HDF5: {filepath}")
        except Exception as e:
            handle_save_exception(filepath, tensor, meta, json_fallback, e)

def save_tensor_in_single_file(tensor_chunk, metadata, dest_dir, json_fallback):
    """Saves the entire tensor as a single HDF5 file, with metadata mapped to batch indices."""
    dir_size = len(os.listdir(dest_dir))
    filename = "batch0_tensor.h5" if dir_size == 0 else f"batch{dir_size}_tensor.h5"
    filepath = os.path.join(dest_dir, filename)
    try:
        with h5py.File(filepath, 'w') as h5_file:
            h5_file.create_dataset('tensor', data=tensor_chunk.numpy(), compression='gzip')
            metadata_group = h5_file.create_group('metadata')
            for idx, meta in enumerate(metadata):
                filename = meta["filename"]
                metadata_group.attrs[f'index_{idx}_filename'] = filename
                for key, value in meta.items():
                    if key != 'filename':  # Exclude filename from HDF5 metadata
                        metadata_group.attrs[f'index_{idx}_{key}'] = json.dumps(value) if isinstance(value, (list, dict)) else value
        #print(f"Batched tensor saved to HDF5: {filepath}")
    except Exception as e:
        handle_save_exception(filepath, tensor_chunk, metadata, json_fallback, e)


def save_tensor_to_hdf5(filepath, tensor, meta):
    """Helper function to save a single tensor with its metadata to an HDF5 file."""
    with h5py.File(filepath, 'w') as h5_file:
        h5_file.create_dataset('tensor', data=tensor.numpy(), compression='gzip')
        metadata_group = h5_file.create_group('metadata')
        for key, value in meta.items():
            if key != 'filename':  # Don't save the filename in the HDF5 metadata
                metadata_group.attrs[key] = json.dumps(value) if isinstance(value, (list, dict)) else value


def load_tensor_chunks_from_hdf5(src_dir, as_individual_files=True):
    """ Loads tensor chunks and metadata from HDF5 files stored in a directory.
        Args:
            src_dir: The directory containing the HDF5 files
            as_individual_files: if True, load each tensor separately
        Returns:
            tensors: A list of PyTorch tensors
            metadata: A list of metadata dictionaries
    """
    tensors = []
    metadata = []
    if as_individual_files:
        for filename in sorted(os.listdir(src_dir)):
            if filename.endswith(".h5"):
                filepath = os.path.join(src_dir, filename)
                with h5py.File(filepath, 'r') as h5_file:
                    tensor_data = torch.tensor(h5_file['tensor'][:])
                    meta = {key: json.loads(val) if isinstance(val, str) else val 
                            for key, val in h5_file['metadata'].attrs.items()}
                tensors.append(tensor_data)
                metadata.append(meta)
    else:
        # Load the entire batch of tensors from a single HDF5 file
        batch_file = os.path.join(src_dir, "batched_tensor.h5")
        with h5py.File(batch_file, 'r') as h5_file:
            tensor_data = torch.tensor(h5_file['tensor'][:])
            for key, val in h5_file['metadata'].attrs.items():
                index, meta_key = key.split('_', 1)
                idx = int(index.split('_')[1])
                if idx >= len(metadata):
                    metadata.append({})
                metadata[idx][meta_key] = json.loads(val) if isinstance(val, str) else val
        tensors = [tensor_data]
    return tensors, metadata


def load_single_tensor_from_hdf5(filepath):
    """ Loads a single batched tensor and its metadata from a specified HDF5 file. """
    with h5py.File(filepath, 'r') as h5_file:
        tensor = torch.tensor(h5_file['tensor'][:])
        # metadata = {key: json.loads(val) if isinstance(val, str) else val
        #             for key, val in h5_file['metadata'].attrs.items()}
    return tensor #, metadata


def get_features_from_h5(feature_dir, num_samples, feature_dim=256, spatial_dims=(16,16)):
    feature_tensor_files = [f for f in os.listdir(feature_dir) if f.endswith(".h5")]
    global_features = torch.empty(num_samples, feature_dim, *spatial_dims, dtype=torch.float32)
    curr_batch_idx = 0
    for h5file in feature_tensor_files:
        tensor_path = os.path.join(feature_dir, h5file)
        feat_tensor = load_single_tensor_from_hdf5(tensor_path)
        curr_batch_size = feat_tensor.shape[0]
        global_features[curr_batch_idx:(curr_batch_idx+curr_batch_size)] = feat_tensor
    return global_features



def handle_save_exception(filepath, tensor, meta, json_fallback, exception):
    """Handles saving failure and provides a fallback to JSON."""
    print(f"Failed to save to HDF5: {exception}")
    if json_fallback:
        json_filepath = filepath.replace('.h5', '_metadata.json')
        metadata_dict = {
            'tensor_shape': tensor.shape,
            'tensor_dtype': str(tensor.dtype),
            'metadata': meta
        }
        with open(json_filepath, 'w') as json_file:
            json.dump(metadata_dict, json_file, indent=4)
        print(f"Metadata saved to JSON: {json_filepath}")
    else:
        raise exception


def save_tensor_chunks_to_png(tensor_chunk, metadata, dest_dir):
    if not isinstance(tensor_chunk, torch.Tensor):
        raise ValueError("The input must be a PyTorch tensor.")
    if len(tensor_chunk) != len(metadata):
        raise ValueError("The length of tensor_chunk must match the length of metadata.")
    # Ensure tensor is on CPU before saving to avoid GPU-HDF5 transfer issues
    tensor_chunk = tensor_chunk.cpu()
    # Create the destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    # Iterate over the batch dimension (B)
    for idx, (tensor, meta) in enumerate(zip(tensor_chunk, metadata)):
        filename = meta["filename"] #meta.get('filename', f'tensor_{idx}.h5')
        filepath = os.path.join(dest_dir, filename)
        try:
            # Save the tensor slice to png
            IO.write_png(tensor, filepath, compression_level=3)
        except Exception as e:
            print(f"ERROR: Failed to save to png for {filename}: {e}")


def load_metrics_from_json(json_paths, model_names: dict):
    data = []
    for json_file in json_paths:
        model_name = model_names[os.path.basename(os.path.dirname(json_file))]
        with open(json_file, 'r') as f:
            metrics = json.load(f)
            for image_name, metric_values in metrics.items():
                metric_values['image'] = image_name
                metric_values['model'] = model_name
                # Convert all metric values to numeric
                for key in ["iou", "accuracy", "precision", "recall", "f1", "confusion", "phi"]:
                    metric_values[key] = pd.to_numeric(metric_values.get(key, None), errors='coerce')
                data.append(metric_values)
    return pd.DataFrame(data)

######^