import pytest, sys

try:
    import torch
    TORCH_AVAILABLE = True
except (ImportError, NameError, AttributeError, OSError):
    TORCH_AVAILABLE = False

MODEL, DATA = None, None
get_profiler = None

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not used as a backend")
class BaseFunctionalTest:
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
        self.x = [torch.randn(1, 3, 32, 32)]
        self.y = [torch.randn(1, 8192)]
        self.batch_size = 1

    def __iter__(self):
        for x, y in zip(self.x, self.y):
            yield torch.FloatTensor(x), torch.FloatTensor(y)

    def __len__(self):
        return 1

def get_profiler():
    data = TorchProfiler.enable_forward_pass_data_splits(DATA)
    profiler = TorchProfiler(MODEL, data)
    profiler.register_profiler_function(ComputeComplexity())
    profiler.register_profiler_function(ComputeExecutionTime())
    profiler.register_profiler_function(ComputeEvalMetric(get_missclass, 'missclass', unit_name='%'))
    return profiler

def get_custom_profiler():
    data = TorchProfiler.enable_forward_pass_data_splits(DATA)

    profiler = TorchProfiler(CUSTOM_MODEL, data)
    profiler.register_profiler_function(ComputeComplexity())
    profiler.register_profiler_function(ComputeExecutionTime())
    profiler.register_profiler_function(ComputeEvalMetric(get_missclass, 'missclass', unit_name='%'))
    return profiler

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    from deeplite.profiler import ComputeEvalMetric
    from deeplite.torch_profiler.torch_profiler import *
    from deeplite.torch_profiler.torch_inference import get_missclass

    MODEL = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
    )

    class CustomConv(nn.Conv2d):
        def compute_module_complexity(self, inputs, outputs):
            return dict(flops=0,
                    param_size=2 / 8,
                    activation_size=4 / 8)

    CUSTOM_MODEL = nn.Sequential(
        CustomConv(3, 32, kernel_size=5, stride=1, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
    )

    DATA = {"train": FakeData(), "test": FakeData()}
