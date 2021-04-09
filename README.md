<p align="center">
  <img src="https://docs.deeplite.ai/neutrino/_static/content/deeplite-logo-color.png" />
</p>

[![Build Status](https://travis-ci.com/Deeplite/deeplite-profiler.svg?token=KrazyWqBHDFfVzGZSU9X&branch=master)](https://travis-ci.com/Deeplite/deeplite-profiler) [![codecov](https://codecov.io/gh/Deeplite/deeplite-profiler/branch/master/graph/badge.svg?token=D1RMWA1TDC)](https://codecov.io/gh/Deeplite/deeplite-profiler)

# Deeplite Profiler

To be able to use a deep learning model in research and production, it is essential to understand different performance metrics of the model beyond just the model's accuracy.  ``deeplite-profiler`` helps to easily and effectively measure the different performance metrics of a deep learning model. In addition to the existing metrics in the ``deeplite-profiler``, users could seamlessly contribute any custom metric to measure using the profiler. ``deeplite-profiler`` could also be used to compare the performance between two different deep learning models, for example, a teacher and a student model. ``deeplite-profiler`` currently supports PyTorch and TensorFlow Keras (v1) as two different backend frameworks.

<p align="center">
  <img src="https://docs.deeplite.ai/neutrino/_images/profiler.png" />
</p>

* [Installation](#Installation)
    * [Install using pip](#Install-using-pip)
    * [Install from source](#Install-from-source)
    * [Install in Dev mode](#Install-in-dev-mode)
* [How to Use](#How-to-Use)
    * [For PyTorch Model](#For-pytorch-Model)
    * [For Tensorflow Model](#For-Tensorflow-Model)
    * [Output Display](#FOutput-Display)
* [Examples](#Examples)
* [Contribute a Custom Metric](#Contribute-a-Custom-Metric)


# Installation

## Install using pip

Use following command to install the package from our internal PyPI repository. 

```console
$ pip install --upgrade pip
$ pip install deeplite-profiler[`backend`]
```

## Install from source

```console
$ git clone https://github.com/Deeplite/deeplite-profiler.git
$ pip install .[`backend`]
```

One can install specific ``backend`` modules, depending on the required framework and compute support. ``backend`` could be one of the following values
- ``torch``: to install a ``torch`` specific profiler
- ``tf``: to install a ``TensorFlow`` specific profiler (this supports only CPU compute)
- ``tf-gpu``: to install a ``TensorFlow-gpu`` specific profiler (this supports both CPU and GPU compute)
- ``all``: to install both ``torch`` and ``TensorFlow`` specific profiler (this supports only CPU compute for TensorFlow)
- ``all-gpu``: to install both ``torch`` and ``TensorFlow-gpu`` specific profiler (for GPU environment) (this supports both CPU and GPU compute for TensorFlow)


## Install in Dev mode

```console
$ git clone https://github.com/Deeplite/deeplite-profiler.git
$ pip install -e .[`backend`]
$ pip install -r requirements-test.txt
```

To test the installation, one can run the basic tests using `pytest` command in the root folder.

> **_NOTE:_**  Currently, we support Tensorflow 1.14 and 1.15 versions, for Python 3.6 and 3.7. We _do not_ support Python 3.8+.


# How to Use

## For PyTorch Model

```python
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

```python
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

```console
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

# Examples

A list of different examples to use ``deeplite-profiler`` to profiler different PyTorch and TensorFlow models can be found [here](./examples) 

# Contribute a Custom Metric

> **_NOTE:_**  If you looking for an SDK documentation, please head over [here](https://deeplite.github.io/deeplite-profiler/).

We always welcome community contributions to expand the scope of `deeplite-profiler` and also to have additional new metrics. Please refer to the [documentation](https://docs.deeplite.ai/neutrino/profiler.html) for the detailed steps on how to design a new metrics. In general, we follow the `fork-and-pull` Git workflow.

1. **Fork** the repo on GitHub
2. **Clone** the project to your own machine
3. **Commit** changes to your own branch
4. **Push** your work back up to your fork
5. Submit a **Pull request** so that we can review your changes

NOTE: Be sure to merge the latest from "upstream" before making a pull request!