# Deeplite Profiler

The `deeplite-profiler` package is a collection of metrics to profile a single deep learning or compare two different deep learning models. Supports `PyTorch` and `Tensorflow Keras` models, as backend.


# Installation

## Install using pip

Use following command to install the package from our internal PyPI repository. 

```
    $ pip install --upgrade pip
    $ pip install deeplite-profiler[`backend`]
```

> **_NOTE:_**  Currently, we support Tensorflow 1.14 and 1.15 versions, for Python 3.6 and 3.7. We _do not_ support Python 3.8+.



# How to Use

## For PyTorch Model

```
# Step 1: Define native pytorch dataloaders and model
# 1a. data_splits = {"train": train_dataloder, "test": test_dataloader}
data_splits = /* ... load iterable data loaders ... */
model = /* ... load native deep learning model ... */

# Step 2: Create Profiler class and register the profiling functions
data_loader = TorchProfiler.enable_forward_pass_data_splits(data_splits)
profiler = TorchProfiler(model, data_splits, name="Original Model")
profiler.register_profiler_function(ComputeComplexity())
profiler.register_profiler_function(ComputeExecutionTime())
profiler.register_profiler_function(ComputeEvalMetric(get_accuracy, 'accuracy', unit_name='%'))

# Step 3: Compute the registered profiler metrics for the PyTorch Model
profiler.compute_network_status(batch_size=1, device=Device.CPU, short_print=False,
                                                 include_weights=True, print_mode='debug')

# Step 4: Compare two different models or profilers.
profiler2 = profiler.clone(model=deepcopy(model)) # Creating a dummy clone of the current profiler
profiler2.name = "Clone Model"
profiler.compare(profiler2, short_print=False, batch_size=1, device=Device.CPU, print_mode='debug')
```

## For Tensorflow Model

```
# Step 1: Define native tensorflow dataloaders and model
# 1a. data_splits = {"train": train_dataloder, "test": test_dataloader}
data_splits = /* ... load iterable data loaders ... */
model = /* ... load native deep learning model ... */

# Step 2: Create Profiler class and register the profiling functions
data_loader = TFProfiler.enable_forward_pass_data_splits(data_splits)
profiler = TFProfiler(model, data_splits, name="Original Model")
profiler.register_profiler_function(ComputeFlops())
profiler.register_profiler_function(ComputeSize())
profiler.register_profiler_function(ComputeParams())
profiler.register_profiler_function(ComputeLayerwiseSummary())
profiler.register_profiler_function(ComputeExecutionTime())
profiler.register_profiler_function(ComputeEvalMetric(get_accuracy, 'accuracy', unit_name='%'))

# Step 3: Compute the registered profiler metrics for the Tensorflow Keras Model
profiler.compute_network_status(batch_size=1, device=Device.CPU, short_print=False,
                                                 include_weights=True, print_mode='debug')

# Step 4: Compare two different models or profilers.
profiler2 = profiler.clone(model=model) # Creating a dummy clone of the current profiler
profiler2.name = "Clone Model"
profiler.compare(profiler2, short_print=False, batch_size=1, device=Device.CPU, print_mode='debug')
```

## Output Display

An example output of the ``deeplite-profiler`` for ``resnet18`` model using the standard ``CIFAR100`` dataset using ``PyTorch`` backend, looks as follows

```
    +---------------------------------------------------------------+
    |                    deeplite Model Profiler                    |
    +-----------------------------------------+---------------------+
    |             Param Name (Original Model) |                Value|
    |                   Backend: TorchBackend |                     |
    +-----------------------------------------+---------------------+
    |                   Evaluation Metric (%) |              76.8295|
    |                         Model Size (MB) |              42.8014|
    |     Computational Complexity (GigaMACs) |               0.5567|
    |             Total Parameters (Millions) |              11.2201|
    |                   Memory Footprint (MB) |              48.4389|
    |                     Execution Time (ms) |               2.6537|
    +-----------------------------------------+---------------------+
```

- **Evaluation Metric:** Computed performance of the model on the given data
- **Model Size:** Memory consumed by the parameters (weights and biases) of the model
- **Computational Complexity:** Summation of Multiply-Add Cumulations (MACs) per single image (batch_size=1)
- **#Total Parameters:** Total number of parameters (trainable and non-trainable) in the model
- **Memory Footprint:** Total memory consumed by parameters and activations per single image (batch_size=1)
- **Execution Time:** On `NVIDIA TITAN V <https://www.nvidia.com/en-us/titan/titan-v/>`_ GPU, time required for the forward pass per single image (batch_size=1)
