import pytest
from pytest_mock import mocker
from unittest import mock

from deeplite.profiler.evaluate import EvaluationFunction
from deeplite.profiler import ComputeEvalMetric
from deeplite.profiler.metrics import Comparative
from tests.tf_tests.functional import BaseFunctionalTest, TENSORFLOW_SUPPORTED, TENSORFLOW_AVAILABLE, get_profiler, make_model


class TestTFProfiler(BaseFunctionalTest):
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
        assert(status['flops'] == 0.122030433)
        assert(status['total_params'] == 0.002432)
        assert(status['execution_time'] > 0)
        assert(status['model_size'] == 0.00927734375)
        assert(status['memory_footprint'] == 15.89599609375)
        assert(status['eval_metric'] == 100)
        assert(status['layerwise_summary'])
        assert 'inference_time' in status

    @mock.patch('deeplite.profiler.metrics.Flops.get_comparative', return_value='coverage')
    def test_compare_profiles(self, *args):
        from deeplite.profiler import Device
        device = Device.CPU
        batch_size = 1
        profiler = get_profiler()
        profiler2 = profiler.clone(model=make_model())
        profiler.compare(profiler2, short_print=False, batch_size=batch_size, device=device, print_mode='debug')
        assert profiler.status_get('layerwise_summary') is not None
        assert profiler2.status_get('layerwise_summary') is not None
        assert all(v1 == v2 for (k1, v1), (k2, v2) in zip(profiler.status_items(), profiler2.status_items())
                   if k1 not in ('layerwise_summary', 'inference_time', 'execution_time'))

    def test_secondary_eval_override(self, *args):
        profiler = get_profiler()
        rval = {'acc': 1, 'b': 2, 'c': 4}
        class DictReturnEval(EvaluationFunction):
            def __call__(self, model, loader):
                return rval

        dummy_eval = DictReturnEval()
        eval_profiler = ComputeEvalMetric(dummy_eval, 'acc', unit_name='%')
        eval_profiler.add_secondary_metric('b')
        eval_profiler.add_secondary_metric('c', 'ms', 'milliseconds', Comparative.DIV)
        profiler.register_profiler_function(eval_profiler, override=True)
        profiler.compute_status('eval_metric')
        profiler.display_status()
        assert profiler.status_get('eval_metric') == rval['acc']
        assert profiler.status_get('b') == rval['b']
        assert profiler.status_get('c') == rval['c']