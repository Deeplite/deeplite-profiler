import pytest
from pytest_mock import mocker
from unittest import mock

from deeplite.profiler.evaluate import EvaluationFunction
from deeplite.profiler import ComputeEvalMetric
from deeplite.profiler.metrics import Comparative
from tests.tflite_tests.functional import BaseFunctionalTest, TENSORFLOW_SUPPORTED, TENSORFLOW_AVAILABLE, get_profiler, make_model


@pytest.mark.skipif(not TENSORFLOW_SUPPORTED, reason="Tensorflow not supported for Python 3.8+")
class TestTFLITEProfiler(BaseFunctionalTest):
    def test_empty_display_status(self, *args):
        profiler = get_profiler()
        profiler.display_status()

    def test_compute_network_status(self, *args):
        from deeplite.profiler import Device
        device = Device.CPU
        batch_size = 1
        profiler = get_profiler()
        status = profiler.compute_network_status(batch_size=batch_size, device=device, short_print=False,
                                                 include_weights=True, print_mode='debug')
        assert(status['flops'] == 6.4e-8)
        assert(status['total_params'] == 0.006146)
        #assert(status['model_size'] == 0.012917)
        assert(status['memory_footprint'] == 0.024584)
        assert(status['eval_metric'] == 100)
        assert 'inference_time' in status
