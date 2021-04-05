import pytest, sys
from unittest import mock

try:
    import torch
    TORCH_AVAILABLE = True
except (ImportError, NameError, AttributeError, OSError):
    TORCH_AVAILABLE = False

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not used as a backend")
class BaseUnitTest:
	def setup_method(self, method):
		""" setup any state tied to the execution of the given method in a
		class.  setup_method is invoked for every test method of a class.
		"""
		print("setup ", method.__name__)
		pass


	def teardown_method(self, method):
		""" teardown any state that was previously setup with a setup_method
		call.
		"""
		print("teardown ", method.__name__)
		pass

class FakeData:
    def __init__(self):
        self.x = [torch.randn(1, 1, 2, 2)]
        self.y = [torch.randn(1, 1)]
        self.batch_size = 1

    def __iter__(self):
        for x, y in zip(self.x, self.y):
            yield torch.FloatTensor(x), torch.FloatTensor(y)

    def __len__(self):
        return 1

def get_profiler():
    data = {"train": FakeData(), "test": FakeData()}
    data = TorchProfiler.enable_forward_pass_data_splits(data)
    profiler = TorchProfiler(mock.MagicMock(), data)
    profiler.register_profiler_function(ComputeComplexity())
    profiler.register_profiler_function(ComputeExecutionTime())
    return profiler

@pytest.fixture(scope='module')
def dataloader():
    return TorchDataLoader(FakeData(), TorchForwardPass(model_input_pattern=(0, '_')))

@pytest.fixture(scope='module')
def fp():
    return TorchForwardPass(model_input_pattern=(0, '_'))

if TORCH_AVAILABLE:
    import torch
    from deeplite.profiler.utils import Device
    from deeplite.torch_profiler.torch_profiler import *
    from deeplite.torch_profiler.torch_data_loader import TorchDataLoader, TorchForwardPass

