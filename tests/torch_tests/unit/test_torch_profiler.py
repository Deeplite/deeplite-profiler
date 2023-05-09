
import pytest
from tests.torch_tests.unit import BaseUnitTest, TORCH_AVAILABLE, get_profiler
from unittest import mock

@mock.patch("deeplite.torch_profiler.torch_profiler.ComputeComplexity._compute_complexity",
            return_value=(1, 2, 3, 4, "summary_str", 'layer_data'))
@mock.patch("deeplite.profiler.utils.AverageAggregator.get", return_value=(1))
class TestTorchProfiler(BaseUnitTest):
    def test_compute_flops(self, *args):
        profiler = get_profiler()
        profiler.compute_status('flops')

    def test_compute_execution_time(self, *args):
        from deeplite.profiler import Device
        profiler = get_profiler()
        batch_size = 1
        device = Device.CPU
        profiler.compute_status('execution_time', device=device, batch_size=batch_size)

    def test_compute_size(self, *args):
        from deeplite.profiler import Device
        profiler = get_profiler()
        batch_size = 1
        device = Device.CPU
        profiler.compute_status('model_size', device=device, batch_size=batch_size)

    def test_compute_memory_footprint(self, *args):
        from deeplite.profiler import Device
        profiler = get_profiler()
        batch_size = 1
        device = Device.CPU
        profiler.compute_status('memory_footprint', device=device, batch_size=batch_size)

