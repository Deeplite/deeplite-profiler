import pytest, sys

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except (ImportError, NameError, AttributeError, OSError):
    TENSORFLOW_AVAILABLE = False

TENSORFLOW_SUPPORTED = False
if sys.version_info < (3,8):
    TENSORFLOW_SUPPORTED = True

MODEL, DATA = None, None
get_profiler, make_model = None, None

@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="Tensorflow is not used as a backend")
@pytest.mark.skipif(not TENSORFLOW_SUPPORTED, reason="Tensorflow not supported for Python 3.8+")
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

def make_model():
    # Create a TensorFlow graph
    graph = tf.Graph()
    with graph.as_default():
        input_shape = (None, 32, 32, 3)

        # Create the input tensor
        input_tensor = tf.placeholder(tf.float32, shape=input_shape, name='input')

        # Add Conv2D layer
        conv_output = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)

        # Add DEPTHWISE_CONV_2D layer
        depthwise_conv_output = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), activation='relu', padding='same')(conv_output)

        # Flatten the output of Conv2D layer
        flatten = tf.keras.layers.Flatten()(depthwise_conv_output)

        # Create the fully connected layer with one unit
        fc_weights = tf.Variable(tf.random_normal([flatten.shape[1], 1]), name='fc_weights')
        fc_biases = tf.Variable(tf.random_normal([1]), name='fc_biases')

        # Initialize the variables
        init_op = tf.global_variables_initializer()

        # Create the session
        with tf.Session() as sess:
            # Run the initialization operation
            sess.run(init_op)

            fc_output = tf.add(tf.matmul(flatten, fc_weights), fc_biases, name='fc_output')

            # Create the output tensor
            output_tensor = tf.identity(fc_output, name='output')

            # Convert the model to TFLite
            converter = tf.lite.TFLiteConverter.from_session(
                sess=sess,
                input_tensors=[input_tensor],
                output_tensors=[output_tensor]
            )
            tflite_model = converter.convert()

    return tflite_model

class TFLITE_FakeData:
    def __init__(self):       
        self.x = np.float32(np.random.rand(1, 32, 32, 3))
        self.y = np.int32(np.random.rand(1, 1))

    def __iter__(self):
        for x, y in zip(self.x, self.y):            
            yield x, np.eye(10)[y.reshape(-1)]

    def __len__(self):
        return self.x.shape[0]

def get_profiler():
    data = TFLiteProfiler.enable_forward_pass_data_splits(DATA)
    profiler = TFLiteProfiler(MODEL, data)
    profiler.register_profiler_function(ComputeFlops())
    profiler.register_profiler_function(ComputeSize())
    profiler.register_profiler_function(ComputeParams())
    profiler.register_profiler_function(ComputeEvalMetric(get_accuracy, 'accuracy', unit_name='%'))
    return profiler


if TENSORFLOW_AVAILABLE and TENSORFLOW_SUPPORTED:
    import numpy as np
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    try:
        tf.compat.v1.enable_eager_execution()
    except Exception:
        pass
    
    from deeplite.profiler import ComputeEvalMetric
    from deeplite.tflite_profiler.tflite_profiler import *
    from deeplite.tflite_profiler.flops.compute_flops import ComputeFlops

    from deeplite.tflite_profiler.tflite_inference import get_accuracy

    MODEL = make_model()
    DATA = {"train": list(TFLITE_FakeData()), "test": list(TFLITE_FakeData())}
