import pytest
from pytest_mock import mocker
from unittest import mock

from deeplite.profiler.evaluate import EvaluationFunction
from deeplite.profiler import ComputeEvalMetric
from deeplite.profiler.metrics import Comparative
from tests.tflite_tests.functional import BaseFunctionalTest, get_profiler, make_model


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
        assert(status['flops'] == 0.00235936)
        assert(status['total_params'] == 0.102562)
        assert(status['model_size'] == 0.137165)
        assert(status['memory_footprint'] == 0.410248)
        assert(status['eval_metric'] == 100)
        assert 'inference_time' in status
