import pytest
from tests.profiler_tests.unit import BaseUnitTest
from unittest import mock

from deeplite.profiler.formatter import getLogger, setLogger
from deeplite.profiler.profiler import Profiler, ProfilerFunction, ExternalProfilerFunction
from deeplite.profiler.metrics import *


class TestProfiler(BaseUnitTest):
    def test_compute_status(self):
        profiler = get_profiler()
        flops_func = DummyFlopsProfilerFunction()
        profiler.register_profiler_function(flops_func)
        flops = profiler.compute_status('flops', dummy_arg=2)
        assert flops == 2

        with pytest.raises(ValueError):
            profiler.compute_status('coverage')
        assert not profiler.status_contains('coverage')

    def test_compute_network_status(self):
        profiler = get_profiler()
        flops_func = DummyFlopsProfilerFunction()
        profiler.register_profiler_function(flops_func)
        profiler.compute_status('flops', dummy_arg='a')

        rval = profiler.compute_network_status(recompute=False)
        assert rval['flops'] == 'a'
        rval = profiler.compute_network_status(recompute=True)
        assert rval['flops'] == 1

    def test_register_profiler_function(self):
        profiler = get_profiler()
        flops_func = DummyFlopsProfilerFunction()
        pfunc = DummyProfilerFunction()

        profiler.register_profiler_function(flops_func)

        # this should trigger the warning
        profiler.register_profiler_function(pfunc)

        profiler.register_profiler_function(flops_func, override=True)
        assert profiler._profiling_functions_register['flops'].function is flops_func
        assert profiler._profiling_functions_register['flops'].overriding is True

        with pytest.raises(RuntimeError):
            # conflicts over 'flops' with two override=True
            profiler.register_profiler_function(pfunc, override=True)

    def test_fail_register_profiler_function(self):
        profiler = get_profiler()
        with pytest.raises(TypeError):
            profiler.register_profiler_function(BadDefinedProfilerFunction())
        with pytest.raises(TypeError):
            profiler.register_profiler_function(BadDefinedExternalProfilerFunction())

    def test_profiler_logger(self):
        logger_holder = getLogger()
        logger = logger_holder.logger
        setLogger(logger)
        assert logger_holder.logger is logger

        with pytest.raises(TypeError):
            type(logger_holder)()


class DummyFlopsProfilerFunction(ProfilerFunction):
    def get_bounded_status_keys(self):
        return Flops()

    def __call__(self, model, data_splits, dummy_arg=1):
        return dummy_arg


class DummyProfilerFunction(ProfilerFunction):
    def get_bounded_status_keys(self):
        return Flops()

    def __call__(self, model, data_splits):
        return 'x'


class BadDefinedProfilerFunction(ProfilerFunction):
    def __call__(self, *args, **kwargs):
        pass

class BadDefinedExternalProfilerFunction(ExternalProfilerFunction):
    def __init__(self):
        def bad_func(x, y, **kwargs):
            pass
        super().__init__(bad_func)
    def get_bounded_status_keys(self):
        return Flops()


class DummyProfiler(Profiler):
    @property
    def dl_cls(self):
        return mock.MagicMock()

    @property
    def fp_cls(self):
        return mock.MagicMock()


def get_profiler():
    model = mock.MagicMock()
    data_splits = {'train': mock.MagicMock()}
    return DummyProfiler(model, data_splits)
