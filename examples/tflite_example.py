import logging
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.lite.python import lite
import tflite

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
try:
    tf.compat.v1.enable_eager_execution()
except Exception:
    pass

from deeplite.profiler import ComputeEvalMetric, Device
from deeplite.tflite_profiler.tflite_inference import get_accuracy
from deeplite.tflite_profiler.tflite_profiler import *

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# Step 1: Define native Tensorflow dataloaders and model (tf.data.Dataset)
# 1a. data_splits = {"train": train_dataloder, "test": test_dataloader}
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = np.eye(100)[y_train.reshape(-1)]
y_test = np.eye(100)[y_test.reshape(-1)]
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
        .shuffle(buffer_size=x_train.shape[0]) \
        .batch(1)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)) \
        .batch(1)
data_splits = {'train': train_dataset, 'test': test_dataset}


# Define the input and output shapes
input_shape = (1, 32, 32, 3)

# Create a TensorFlow graph
graph = tf.Graph()
with graph.as_default():
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input#tf.keras.applications.vgg19.preprocess_input
    # Create the input and output tensors
    input_tensor = tf.placeholder(tf.float32, shape=input_shape, name='input')

    output_tensor = tf.identity(input_tensor, name='output')

    # Convert the graph to a TensorFlow Lite model
    converter = tf.lite.TFLiteConverter.from_session(sess=tf.Session(graph=graph),
                                                     input_tensors=[input_tensor],
                                                     output_tensors=[output_tensor])
    tflite_model = converter.convert()

#with open("models/cifar10.tflite", "rb") as f:
#   tflite_model = f.read()
# Step 2: Create Profiler class and register the profiling functions
data_loader = TFLiteProfiler.enable_forward_pass_data_splits(data_splits)
profiler = TFLiteProfiler(tflite_model, data_loader, name="Original Model")
profiler.register_profiler_function(ComputeFlops())
profiler.register_profiler_function(ComputeSize())
profiler.register_profiler_function(ComputeParams())
profiler.register_profiler_function(ComputeEvalMetric(get_accuracy, 'accuracy', unit_name='%'))
#
## Step 3: Compute the registered profiler metrics for the tflite Model
profiler.compute_network_status(batch_size=1, device=Device.CPU, short_print=False,
                                                 include_weights=True, print_mode='info', k=1)
# k for top k accuracy
