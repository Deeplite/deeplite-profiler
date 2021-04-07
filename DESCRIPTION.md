# Deeplite Profiler

The `deeplite-profiler` package is a collection of metrics to profile a single deep learning or compare two different deep learning models. Supports `PyTorch` and `Tensorflow Keras` models, as backend.


# Installation

## Install using pip

Use following command to install the package from our internal PyPI repository. 
```console
>>> pip install --upgrade pip
>>> pip install deeplite-profiler[`backend`]
```

One can install specific ``backend`` modules, depending on the required framework and compute support. ``backend`` could be one of the following values
    - ``torch``: to install a ``torch`` specific profiler
    - ``tf``: to install a ``TensorFlow`` specific profiler (this supports only CPU compute)
    - ``tf-gpu``: to install a ``TensorFlow-gpu`` specific profiler (this supports both CPU and GPU compute)
    - ``all``: to install both ``torch`` and ``TensorFlow`` specific profiler (this supports only CPU compute for TensorFlow)
    - ``all-gpu``: to install both ``torch`` and ``TensorFlow-gpu`` specific profiler (for GPU environment) (this supports both CPU and GPU compute for TensorFlow)

> **_NOTE:_**  Currently, we support Tensorflow 1.14 and 1.15 versions, for Python 3.6 and 3.7. We do not support Python 3.8+.


# How to Use

## For PyTorch Model
