from numpy import ndarray
import torch
from deeplite.profiler.utils import Device
from deeplite.profiler.data_loader import DataLoader, TensorSampler, ForwardPass


class TorchDataLoader(DataLoader):
    # TODO expose something for the end-user to allow stateful dataset?
    def dump_native_info(self):
        return

    def load_native_info(self, native_state):
        return self

    def __len__(self):
        # This won't work if the dataset is of IterableDataset type
        return len(self.native_dl)

    def _create_iter(self):
        return iter(self.native_dl)

    @property
    def batch_size(self):
        return self.native_dl.batch_size


class TorchForwardPass(ForwardPass):
    @property
    def _tensor_sampler_cls(self):
        return TorchTensorSampler

    # provides a default implementation but nothing prevents the user from overriding it
    def model_call(self, model, x, device):
        if not self.expecting_common_inputs:
            raise TypeError(
                "If not using the TorchForwad pass common inputs default implementation, 'model_call' should"
                " be overridden")

        if device == Device.CPU:
            model.cpu()
        else:
            model.cuda()

        x = self._tensor_sampler.to_device(x, device, standardize=False)
        return model(*x)


class TorchTensorSampler(TensorSampler):
    def _standardize_tensor(self, x):
        if isinstance(x, ndarray):
            x = torch.from_numpy(x)
        elif not isinstance(x, torch.Tensor):
            raise ValueError()
        return x

    def _create_random_tensor(self, x_info, batch_size):
        if x_info.dtype in (torch.complex64, torch.complex128, torch.complex32,):
            raise RuntimeError("Complex number not supported")
        return torch.rand(batch_size, *x_info.shp, dtype=torch.float).cpu().type(x_info.dtype)

    def _get_info(self, x):
        # dont forget to strip that batch axis!
        return x.shape[1:], x.dtype

    # unfortunately cannot be made abstract as device calls are too framework specific
    def to_device(self, tensors_tuple, device, standardize=True):
        if device == Device.CPU:
            f = lambda x: x.cpu()
        else:
            f = lambda x: x.cuda()
        if standardize:
            tensors_tuple = self.standardize_tensors(tensors_tuple)
        rval = self._loop_over_tensors_tuple(tensors_tuple, f)
        return rval
