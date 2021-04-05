import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
try:
    tf.compat.v1.enable_eager_execution()
except Exception:
    pass

from deeplite.tf_profiler.tf_profiler import TFProfiler
from deeplite.tf_profiler.tf_profiler import *
from deeplite.profiler import Device, ComputeEvalMetric
from deeplite.tf_profiler.tf_inference import get_accuracy, get_missclass

# Step 1: Define native Tensorflow dataloaders and model (tf.data.Dataset)
# 1a. data_splits = {"train": train_dataloder, "test": test_dataloader}
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255    
y_train = np.eye(100)[y_train.reshape(-1)]
y_test = np.eye(100)[y_test.reshape(-1)]
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
        .shuffle(buffer_size=x_train.shape[0]) \
        .batch(128) 
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)) \
        .batch(128) 
data_splits = {'train': train_dataset, 'test': test_dataset}

# 1b. Load the native Tensorflow Keras model: Transfer learning from pretrained model
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input#tf.keras.applications.vgg19.preprocess_input
base_model = tf.keras.applications.VGG19(input_shape=(32, 32, 3),
                                               include_top=False,
                                               weights='imagenet')
inputs = tf.keras.Input(shape=(32, 32, 3))
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(100)(x)
native_teacher = tf.keras.Model(inputs, outputs)
native_teacher.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.SGD(), metrics=['accuracy'])

# Step 2: Create Profiler class and register the profiling functions
data_loader = TFProfiler.enable_forward_pass_data_splits(data_splits)
profiler = TFProfiler(native_teacher, data_loader, name="Original Model")
profiler.register_profiler_function(ComputeFlops())
profiler.register_profiler_function(ComputeSize())
profiler.register_profiler_function(ComputeParams())
profiler.register_profiler_function(ComputeLayerwiseSummary())
profiler.register_profiler_function(ComputeExecutionTime())
profiler.register_profiler_function(ComputeEvalMetric(get_accuracy, 'accuracy', unit_name='%'))

# Step 3: Compute the registered profiler metrics for the Tensorflow Keras Model
profiler.compute_network_status(batch_size=1, device=Device.GPU, short_print=False,
                                                 include_weights=True, print_mode='debug')


