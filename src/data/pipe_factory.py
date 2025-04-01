import inspect
#from contextlib import contextmanager
from typing import Iterable, Dict, Union, Any, Callable
from collections.abc import Mapping, Sequence
import torch
from torch.utils.data import Dataset, IterableDataset
import torch.multiprocessing as mp


BatchType = Union[Dict[str, Union[torch.Tensor, str]], torch.Tensor, Iterable[torch.Tensor]]

# util functions will be moved to another file eventually
# UPDATED since types.LambdaType always evaluates to true for all functions
def is_picklable_fn(fn: Any) -> bool:
    """ Returns True if 'fn' is a valid, top-level function that is non-lambda, non-generator, non-async,
        and suitable for pickling in a multiprocessing context.
    """
    if not inspect.isfunction(fn):
        print("Provided fn is not a function object.")
        return False
    if fn.__name__ == "<lambda>": # (both lambdas and regular functions are 'FunctionType', so check the name)
        print("Lambda functions are not picklable.")
        return False
    if inspect.iscoroutinefunction(fn):
        print("Async functions are not allowed.")
        return False
    if inspect.isgeneratorfunction(fn):
        print("Generator functions are not picklable.")
        return False
    # Must be defined at the top level of a module (so it doesn’t capture state that prevents pickling)
    if fn.__module__ is None or fn.__qualname__ != fn.__name__:
        print("Function is not defined at the top-level of a module.")
        return False
    return True


def assert_model_device_is_safe(model: torch.nn.Module):
    """ Check if the model is on a device that is safe for multiprocessing. """
    def warning_msg(attr, attr_name, device_type):
        return f"Model {attr} '{attr_name}' is on device '{device_type}', which is not safe for multiprocessing. " + \
                "Move your model to CPU before passing it to the piped dataset factory \n\t Example: model.to('cpu')"
    #? NOTE: would've done this by just checking if it wasn't on CPU, but I assumed some of these would be fine to use
    bad_devices = {"cuda", "mps", "xpu", "xla", "meta"}
    # Check all parameters
    for name, param in model.named_parameters():
        device_type = param.device.type
        if device_type in bad_devices:
            raise ValueError(warning_msg("parameter", name, device_type))
    # Optional: check buffers too
    for name, buf in model.named_buffers():
        device_type = buf.device.type
        if device_type in bad_devices:
            raise ValueError(warning_msg("buffer", name, device_type))



def apply_to_tensors(data, func):
    if isinstance(data, torch.Tensor):
        return func(data)
    elif isinstance(data, Mapping):
        return {k: apply_to_tensors(v, func) for k, v in data.items()}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return type(data)(*(apply_to_tensors(d, func) for d in data))
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return type(data)(apply_to_tensors(d, func) for d in data)
    else:
        return data

def move_to_device(data, device):
    return apply_to_tensors(data, lambda x: x.to(device))

def ensure_batch_dim(data):
    def _ensure_batch_dim(tensor):
        return tensor if tensor.dim() > 3 else tensor.unsqueeze(0)
    return apply_to_tensors(data, _ensure_batch_dim)



# just assumes that given model forward method is compatible with `inputs` - top-level definition ensures picklability
def default_model_output_fn(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """ Default function to get model output. Assumes model is a torch.nn.Module or has a defined __call__ method. """
    return model(inputs)


# --- Indexed Dataset Implementation --- #

class IndexedPipeDataset(Dataset):
    def __init__(self, conn, length):
        self.conn = conn
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self.conn.send(('get', idx))
        data = self.conn.recv()
        if data is None:
            raise IndexError("Index out of bounds")
        return data


class IndexedPipeDatasetWrapper:
    def __init__(self, conn, worker_process, length):
        self.conn = conn
        self.worker_process = worker_process
        self.dataset = IndexedPipeDataset(conn, length)

    def __getattr__(self, attr):
        return getattr(self.dataset, attr)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def __del__(self):
        if self.worker_process.is_alive():
            try:
                self.conn.send('close')
                self.worker_process.join()
            except:
                pass


# Worker for Indexed Dataset
def indexed_feature_generation_worker(conn, dataset, model, device, output_fn):
    model.to(device)
    model.eval()
    with torch.no_grad():
        while True:
            msg = conn.recv()
            if isinstance(msg, tuple) and msg[0] == 'get':
                idx = msg[1]
                try:
                    sample = dataset[idx]
                    sample_gpu = ensure_batch_dim(move_to_device(sample, device))
                    output = apply_to_tensors(output_fn(model, sample_gpu), lambda x: x.to(device="cpu"))
                    conn.send((output, sample))
                except IndexError:
                    conn.send(None)
            elif msg == 'close':
                break
    conn.close()


# Factory for Indexed Dataset
def create_indexed_streaming_feature_dataset(dataset, model, device='cuda:0', output_fn=default_model_output_fn):
    assert_model_device_is_safe(model)
    if not is_picklable_fn(output_fn):
        raise ValueError(
            f"The provided output_fn '{output_fn.__name__}' is not picklable. "
            "Ensure it is defined in the global scope (at the top-level of a module)."
        )
    parent_conn, child_conn = mp.Pipe()
    worker_process = mp.Process(
        target=indexed_feature_generation_worker,
        args=(child_conn, dataset, model, device, output_fn)
    )
    worker_process.start()
    return IndexedPipeDatasetWrapper(parent_conn, worker_process, len(dataset))



# --- Iterable Dataset Implementation --- #

class IterablePipeDataset(IterableDataset):
    def __init__(self, conn):
        super(IterablePipeDataset).__init__()
        self.conn = conn

    def _generator(self):
        while True:
            self.conn.send('next')
            data = self.conn.recv()
            if data is None:
                break
            yield data

    def __iter__(self):
        return iter(self._generator())

    # def __next__(self):
    #     return next(self._generator())

#? NOTE: need this to subclass IterableDataset because an internal check by the dataloader depends on it to address it correctly
class IterablePipeDatasetWrapper(IterableDataset):
    def __init__(self, conn, worker_process):
        super().__init__()
        self.conn = conn
        self.worker_process = worker_process
        self.dataset = IterablePipeDataset(conn)

    def __getattr__(self, attr):
        return getattr(self.dataset, attr)

    def __iter__(self):
        return iter(self.dataset)

    def _join_worker(self):
        if self.worker_process.is_alive():
            try:
                self.conn.send('close')
                self.worker_process.join()
            except:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._join_worker()

    def __del__(self):
        self._join_worker()


# Worker for Iterable Dataset
def iterable_feature_generation_worker(conn, dataset, model, device, output_fn):
    model.to(device)
    model.eval()
    data_iter = iter(dataset)
    with torch.no_grad():
        while True:
            msg = conn.recv()
            if msg == 'next':
                try:
                    sample = next(data_iter)
                    sample_gpu = ensure_batch_dim(move_to_device(sample, device))
                    output = apply_to_tensors(output_fn(model, sample_gpu), lambda x: x.to(device="cpu"))
                    conn.send((output, sample))
                except StopIteration:
                    conn.send(None)
            elif msg == 'close':
                break
    conn.close()


# TODO: may want to make separate factory methods that return loaders directly
    # the loaders could be subclassed to be essentially regular DataLoaders that requires num_workers = 0
    # and which have additional restrictions in the constructor based on the type of dataset (iterable or indexable) being passed

# Factory for Iterable Dataset
def create_iterable_streaming_feature_dataset(dataset, model, device='cuda:0', output_fn=default_model_output_fn):
    assert_model_device_is_safe(model)
    if not is_picklable_fn(output_fn):
        raise ValueError(
            f"The provided output_fn '{output_fn.__name__}' is not picklable. "
            "Ensure it is defined in the global scope (at the top-level of a module)."
        )
    parent_conn, child_conn = mp.Pipe()
    worker_process = mp.Process(
        target=iterable_feature_generation_worker,
        args=(child_conn, dataset, model, device, output_fn)
    )
    worker_process.start()
    return IterablePipeDatasetWrapper(parent_conn, worker_process)


#########################################################################################################
# testing functions - not part of final imports to new project
#########################################################################################################

# testing this with a local function to see what happens then trying it globally
def model_logit_output_fn(model: torch.nn.Module, inputs: BatchType) -> torch.Tensor:
    """ Custom function to get model output. Assumes model is a torch.nn.Module or has a defined __call__ method. """
    #print("img shape, type:", inputs["img"].shape, inputs["img"].dtype)  # Debugging line to check the input shape
    return model(inputs["img"]).squeeze(0).logits


def model_feature_output_fn(model: torch.nn.Module, inputs: BatchType) -> torch.Tensor:
    """ for testing feature capture of the PartitionedSegformer model """
    # TODO: planning to make a convenience function for this later
    return model.partial_decoding(model.encode(inputs["img"])[0]).squeeze(0)



""" Short-term Extension:
PROBLEM: Users might inadvertently pass lambdas or locally defined functions as output_fn causing pickling errors.
TODO:
- [ ] Clearly document that output_fn must be top-level defined to avoid pickling issues.
- [x] Provide an explicit error message or guard (using inspect) when non-picklable objects are detected as output_fn.

PROBLEM: Currently, the model is explicitly duplicated in subprocess memory. So if users attempt parallel or nested multiprocessing later
    (e.g., multiple workers each with separate pipes), GPU memory limitations will quickly become apparent.
TODO:
- [ ] Clearly document that the current design should be used with a single subprocess per GPU model.
- [ ] Look into TensorRT or TorchScript serialization if scaling model serving across many parallel workers in the future.
    - may just try to serialize the model with TorchScript and throw a warning if it fails

PROBLEM: Currently, __getitem__ in `PipeDatasetWrapper` ignores the provided index, as the pipe always returns the next available data from the workers.
    Thus users might incorrectly assume random indexing or shuffling is supported directly by the pipe dataset.
TODO:
- [N/A] Explicitly document that your current streaming pipe dataset only supports sequential iteration and does not support random access or shuffling directly.
- [ ] If shuffling or sampling is required, explicitly state it must be done upstream in the wrapped source dataset.

PROBLEM: Users might provide datasets with unexpected tensor shapes (not exactly 3 or 4 dimensions).
TODO:
- [ ] Provide generalized and well-documented shape-checking logic, possibly with user-configurable dimension checks.
- [ ] remove more restrictive dimensionality requirements; more generalized logic might be beneficial depending on future datasets.
"""