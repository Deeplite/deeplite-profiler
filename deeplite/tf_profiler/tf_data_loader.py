import time

import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
try:
    tf.compat.v1.enable_eager_execution()
except:
    pass
import numpy as np
import time
from deeplite.profiler.utils import Device
from deeplite.profiler.data_loader import DataLoader, TensorSampler, ForwardPass


class TFDataLoader(DataLoader):
    # TODO expose something for the end-user to allow stateful dataset?
    def dump_native_info(self):
        return

    def load_native_info(self, native_state):
        return self

    def __len__(self):
        # This won't work if the dataset is of IterableDataset type
        return len(list(self.native_dl))

    def _create_iter(self):
        return iter(self.native_dl)

    @property
    def batch_size(self):
        # self.native_dl.batch_size
        for cb in self.native_dl.take(1):
            batch_size = cb[0].shape[0]
        return batch_size


class TFForwardPass(ForwardPass):
    @property
    def _tensor_sampler_cls(self):
        return TFTensorSampler

    def set_device(self, device):
        if device == Device.GPU:
            device_name = "/gpu:0"
        else:
            device_name = "/cpu:0"
        return device_name

    # provides a default implementation but nothing prevents the user from overriding it
    def model_call(self, model, x, device, training=False):
        if not self.expecting_common_inputs:
            raise TypeError(
                "If not using the TF Forwad pass common inputs default implementation, 'model_call' should"
                " be overridden")

        device_name = self.set_device(device)
        with tf.device(device_name):
            x = self._tensor_sampler.to_device(x, device, standardize=True)
            start_time = time.time()
            res = model(*x, training=False)
            return res, time.time() - start_time


class TFTensorSampler(TensorSampler):
    def _standardize_tensor(self, x):

        if tf.is_tensor(x):
            x = x.numpy()
        elif not isinstance(x, np.ndarray):
            raise ValueError()
        return x

    def _create_random_tensor(self, x_info, batch_size):
        if x_info.dtype in (np.complex64, np.complex128,):
            raise RuntimeError("Complex number not supported")
        return np.float32(np.random.rand(batch_size, *(x_info.shp)))

    def _get_info(self, x):
        # dont forget to strip that batch axis!
        if len(x.shape) == 4:
            return x.shape[1:], x.dtype
        return x.shape, x.dtype

    def to_device(self, tensors_tuple, device, standardize=True):
        f = lambda x: x
        if standardize:
            tensors_tuple = self.standardize_tensors(tensors_tuple)
        rval = self._loop_over_tensors_tuple(tensors_tuple, f)
        return rval
