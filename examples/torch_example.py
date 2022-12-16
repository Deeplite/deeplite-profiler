import logging
import sys
from copy import deepcopy

from deeplite.profiler import ComputeEvalMetric, Device
from deeplite.torch_profiler.torch_inference import get_accuracy
from deeplite.torch_profiler.torch_profiler import (ComputeComplexity,
                                                    ComputeExecutionTime,
                                                    TorchProfiler)
from deeplite_torch_zoo import get_data_splits_by_name, get_model_by_name

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# Step 1: Define native pytorch dataloaders and model
# 1a. data_splits = {"train": train_dataloder, "test": test_dataloader}
data_splits = get_data_splits_by_name(
    data_root='',
    dataset_name='cifar100',
    model_name='resnet18',
    batch_size=128,
    num_workers=4,
)

# 1b. Load the native Pytorch model
model = get_model_by_name(
    model_name='resnet18',
    dataset_name='cifar100',
    pretrained=True,
    progress=True
)


# Step 2: Create Profiler class and register the profiling functions
data_loader = TorchProfiler.enable_forward_pass_data_splits(data_splits)
profiler = TorchProfiler(model, data_loader, name="Original Model")
profiler.register_profiler_function(ComputeComplexity())
profiler.register_profiler_function(ComputeExecutionTime())
profiler.register_profiler_function(
    ComputeEvalMetric(get_accuracy, 'accuracy', unit_name='%')
)

# Step 3: Compute the registered profiler metrics for the PyTorch Model
profiler.compute_network_status(
    batch_size=1,
    device=Device.GPU,
    short_print=False,
    include_weights=True,
    print_mode='info',
)

# Step 4: Clone and Compare
profiler2 = profiler.clone(model=deepcopy(model))
profiler2.name = "Clone Model"
profiler2.compare(
    profiler, short_print=False, batch_size=1, device=Device.GPU, print_mode='info'
)
