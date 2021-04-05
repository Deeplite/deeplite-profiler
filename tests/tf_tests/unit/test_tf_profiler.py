import pytest
from tests.tf_tests.unit import BaseUnitTest, TENSORFLOW_SUPPORTED, TENSORFLOW_AVAILABLE, get_profiler
from pytest_mock import mocker
from unittest import mock


@mock.patch("deeplite.tf_profiler.tf_profiler.ComputeFlops._compute_flops",
            return_value=(1))
@mock.patch("deeplite.tf_profiler.tf_profiler.ComputeSize._compute_size",
            return_value=(2, 3))
@mock.patch("deeplite.tf_profiler.tf_profiler.get_temp_model", return_value=mock.MagicMock())
@mock.patch("deeplite.profiler.utils.AverageAggregator.get", return_value=(1))
class TestTFProfiler(BaseUnitTest):
    def test_compute_flops(self, *args):
        profiler = get_profiler()
        profiler.compute_status('flops')

    def test_compute_execution_time(self, *args):
        from deeplite.profiler.utils import Device
        profiler = get_profiler()
        batch_size = 1
        device = Device.CPU
        profiler.compute_status('execution_time', device=device, batch_size=batch_size)

    def test_compute_size(self, *args):
        from deeplite.profiler.utils import Device
        profiler = get_profiler()
        batch_size = 1
        device = Device.CPU
        profiler.compute_status('model_size', device=device, batch_size=batch_size)

    def test_compute_memory_footprint(self, *args):
        from deeplite.profiler.utils import Device
        profiler = get_profiler()
        batch_size = 1
        device = Device.CPU
        profiler.compute_status('memory_footprint', device=device, batch_size=batch_size)


