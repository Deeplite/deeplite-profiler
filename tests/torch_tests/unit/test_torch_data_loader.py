import pytest
import numpy as np
from tests.torch_tests.unit import BaseUnitTest, TORCH_AVAILABLE, fp

class TestTorchDataLoader(BaseUnitTest):
    def test_pass(self, fp):
        from deeplite.profiler import Device
        fp.expecting_common_inputs = False
        with pytest.raises(TypeError):
            fp.model_call(None, 1, Device.CPU)

    def test_sampler(self):
        from deeplite.profiler import Device
        from deeplite.torch_profiler.torch_data_loader import TorchTensorSampler
        x = np.array([[1], [2]])
        sampler = TorchTensorSampler((x,))
        y = sampler.to_device((x,), Device.CPU, standardize=True)
        assert len(y) == 1
        assert np.all(y[0].numpy() == x)

        with pytest.raises(ValueError):
            TorchTensorSampler((None,))



