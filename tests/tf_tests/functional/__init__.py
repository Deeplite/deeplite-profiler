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
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (5, 5), padding='same', data_format="channels_last", input_shape=(224, 224, 3)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2,2), data_format="channels_last"))
    model.add(tf.keras.layers.Flatten(data_format="channels_last"))    
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    model.compile(optimizer=optimizer, loss=loss)
    return model

class TF_FakeData:
    def __init__(self):       
        self.x = np.float32(np.random.rand(10, 224, 224, 3)) #tf.Variable(tf.random.normal([10, 224, 224, 3]))
        self.y = np.int32(np.random.rand(10, 1)) #tf.Variable(tf.random.normal([10, 1]))

    def __iter__(self):
        for x, y in zip(self.x, self.y):            
            yield x, np.eye(10)[y.reshape(-1)]

    def __len__(self):
        return self.x.shape[0]

def get_profiler():
    data = TFProfiler.enable_forward_pass_data_splits(DATA)
    profiler = TFProfiler(MODEL, data)
    profiler.register_profiler_function(ComputeFlops())
    profiler.register_profiler_function(ComputeSize())
    profiler.register_profiler_function(ComputeParams())
    profiler.register_profiler_function(ComputeLayerwiseSummary())
    profiler.register_profiler_function(ComputeExecutionTime())
    profiler.register_profiler_function(ComputeEvalMetric(get_missclass, 'missclass', unit_name='%'))
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
    from deeplite.tf_profiler.tf_profiler import *
    from deeplite.tf_profiler.tf_inference import get_missclass

    MODEL = make_model()
    DATA = {"train": list(TF_FakeData()), "test": list(TF_FakeData())}