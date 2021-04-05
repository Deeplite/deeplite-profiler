import pytest, sys
from unittest import mock

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except (ImportError, NameError, AttributeError, OSError):
    TENSORFLOW_AVAILABLE = False

TENSORFLOW_SUPPORTED = False
if sys.version_info < (3,8):
    TENSORFLOW_SUPPORTED = True

fp, get_profiler = None, None

@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="Tensorflow is not used as a backend")
@pytest.mark.skipif(not TENSORFLOW_SUPPORTED, reason="Tensorflow not supported for Python 3.8+")
class BaseUnitTest:
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


@pytest.fixture(scope="module")
def dataloader():
    return {"train": FakeData(), "test": FakeData()} #TFDataLoader(FakeData(), TFForwardPass(model_input_pattern=(0, '_')))

class FakeData:
    def __init__(self):        
        self.x = tf.Variable(tf.random.normal([1, 224, 224, 3]))
        self.y = tf.Variable(tf.random.normal([1, 1]))

    def __iter__(self):
        for x, y in zip(tf.unstack(self.x), tf.unstack(self.y)):            
            yield x, y

    def take(self, n):
        return tf.unstack(self.x, n), tf.unstack(self.y, n)

    def __len__(self):
        return self.x.shape[0]

@pytest.fixture(scope='module')
def fp():
    return TFForwardPass(model_input_pattern=(0, '_'))

def get_profiler():
    data = {"train": FakeData(), "test": FakeData()}
    data = TFProfiler.enable_forward_pass_data_splits(data)
    profiler = TFProfiler(mock.MagicMock(), data)
    profiler.register_profiler_function(ComputeFlops())
    profiler.register_profiler_function(ComputeSize())
    profiler.register_profiler_function(ComputeParams())
    profiler.register_profiler_function(ComputeLayerwiseSummary())
    profiler.register_profiler_function(ComputeExecutionTime())
    return profiler

if TENSORFLOW_AVAILABLE and TENSORFLOW_SUPPORTED:
    import numpy as np
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    try:
        tf.compat.v1.enable_eager_execution()
    except Exception:
        pass

    from deeplite.tf_profiler.tf_profiler import *
    from deeplite.tf_profiler.tf_data_loader import TFDataLoader, TFForwardPass, TFTensorSampler
    