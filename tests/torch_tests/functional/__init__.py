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

def get_profiler(model_name=None, export=False):
    data = TorchProfiler.enable_forward_pass_data_splits(DATA)
    if model_name == 'custom':
        profiler = TorchProfiler(CUSTOM_MODEL, data)
    elif model_name == 'coverage':
        profiler = TorchProfiler(COVERAGE_MODEL, data)
    else:
        profiler = TorchProfiler(MODEL, data)

    profiler.register_profiler_function(ComputeComplexity(export=export))
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
                    param_size={'weight': 2 / 8},
                    activation_size=4 / 8)

    CUSTOM_MODEL = nn.Sequential(
        CustomConv(3, 32, kernel_size=5, stride=1, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
    )

    class CoverageModule(nn.Module):
        def forward(self, x):
            a = torch.ones((10))
            b = torch.ones((10,10))
            c = torch.ones((10,10,10))

            y = x * torch.ones((1, 10))
            q = torch.matmul(torch.matmul(b,b), y.transpose(0,1))
            r = (torch.matmul(a, b) + torch.matmul(b, a) + torch.matmul(a,a)).unsqueeze(1)
            s = (torch.matmul(c,b) + torch.matmul(a,c) + torch.matmul(c,a)).sum((0,1)).unsqueeze(1)
            return q + r + s

    # linaer, matmul, matvec, multiplication, add, avg pool, upsample
    COVERAGE_MODEL = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(32),
        nn.ReLU(),

        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.LayerNorm((8192)),
        nn.Linear(8192, 10),
        CoverageModule()
    )

    DATA = {"train": FakeData(), "test": FakeData()}
