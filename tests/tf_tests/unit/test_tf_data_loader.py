import pytest
from tests.tf_tests.unit import BaseUnitTest, TENSORFLOW_SUPPORTED, TENSORFLOW_AVAILABLE, fp
import numpy as np

class TestTFDataLoader(BaseUnitTest):
    def test_pass(self, fp):
        from deeplite.profiler.utils import Device
        fp.expecting_common_inputs = False
        with pytest.raises(TypeError):
            fp.model_call(None, 1, Device.CPU)

    def test_sampler(self):
        from deeplite.profiler.utils import Device
        from deeplite.tf_profiler.tf_data_loader import TFTensorSampler
        x = np.array([[1], [2]])
        sampler = TFTensorSampler((x,))
        y = sampler.to_device((x,), Device.CPU, standardize=True)
        assert len(y) == 1
        assert np.all(y[0] == x)

        with pytest.raises(ValueError):
            TFTensorSampler((None,))

