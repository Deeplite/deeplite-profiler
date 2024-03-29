import pytest
import math
from copy import deepcopy

from deeplite.profiler.evaluate import EvaluationFunction
from deeplite.profiler import ComputeEvalMetric
from deeplite.profiler.metrics import Comparative
from tests.torch_tests.functional import BaseFunctionalTest, MODEL, get_profiler
from unittest import mock

class TestTorchProfiler(BaseFunctionalTest):
    def test_empty_display_status(self, *args):
        profiler = get_profiler()
        profiler.display_status()

    @pytest.mark.skip(reason="no dict support yet with tracing implementation")
    def test_dict_profiling(self, *args):
        import torch
        import torch.nn as nn
        from deeplite.torch_profiler.torch_profiler import TorchProfiler, ComputeComplexity
        from deeplite.torch_profiler.torch_data_loader import TorchDataLoader, TorchForwardPass

        class DictModule(nn.Module):
            def forward(self, x=None, y=None):
                return {'add': x + y, 'sub': x - y}

        class DataLoader:
            def __len__(self):
                return 10

            def __iter__(self):
                for i in range(10):
                    yield {'x': torch.FloatTensor(10), 'y': torch.FloatTensor(i)}

        class DictForwardPass(TorchForwardPass):
            def extract_model_inputs(self, batch):
                return batch

            def model_call(self, model, batch, device):
                return model(**batch)

            def get_model_input_shapes(self):
                return (1,), (1,)

            def create_random_model_inputs(self, batch_size):
                return {'x': torch.rand(10), 'y': torch.rand(batch_size)}

        ds = {'train': TorchDataLoader(DataLoader(), fp=DictForwardPass(model_input_pattern=None, expecting_common_inputs=False)),
              'test': TorchDataLoader(DataLoader())}

        # this should not crash and work
        model = DictModule()
        profiler = TorchProfiler(model, ds)
        profiler.register_profiler_function(ComputeComplexity())
        profiler.compute_network_status()

        class DictModule(nn.Module):
            def forward(self, x=None, y=None):
                return {'add': x + y, 'sub': 3}
        # this should not crash and warn
        model = DictModule()
        profiler = TorchProfiler(model, ds)
        profiler.register_profiler_function(ComputeComplexity())
        profiler.compute_network_status()

    def test_custom_conv(self, *args):
        profiler = get_profiler()
        old_size = profiler.compute_status('model_size')
        old_flops = profiler.compute_status('flops')

        custom_profiler = get_profiler('custom')
        custom_profiler.compute_network_status()
        assert custom_profiler.status_get('model_size') == old_size - ((96*25) * (4 - 2/8) / (2**20))
        math.isclose(custom_profiler.status_get('flops'), old_flops - 2490368e-9)  # removed conv flops

    def test_handlers(self, *args):
        profiler = get_profiler('coverage', export=True)
        profiler.compute_network_status()

    @mock.patch('deeplite.profiler.utils.AverageAggregator.get', return_value=2)
    def test_compute_network_status(self, *args):
        from deeplite.profiler import Device

        # test invalid device
        with pytest.raises(ValueError):
            Device('unknown device')

        device = Device('CPU')
        batch_size = 1
        profiler = get_profiler()
        status = profiler.compute_network_status(batch_size=batch_size, device=device, short_print=False,
                                                 print_mode='debug')
        # print(status['layerwise_summary'])
        assert(status['flops'] == 0.002555904)
        assert(status['total_params'] == 0.002432)
        assert(status['execution_time'] == 2000)
        assert(status['model_size'] == 0.00927734375)
        assert(status['memory_footprint'] == 1.0)
        assert(status['eval_metric'] == 100)
        assert(status['layerwise_summary'])
        assert 'inference_time' in status

        status2 = profiler.compute_network_status(batch_size=batch_size*2, device=device, short_print=False,
                                                  print_mode='debug')
        # assert independent of batch size
        assert(status['flops'] == status2['flops'])
        assert(status['total_params'] == status2['total_params'])
        assert(status['model_size'] == status2['model_size'])
        assert(status['memory_footprint'] == status2['memory_footprint'])

    @mock.patch('deeplite.profiler.metrics.Flops.get_comparative', return_value='coverage')
    def test_compare_profiles(self, *args):
        from deeplite.profiler import Device
        device = Device.CPU
        batch_size = 1
        profiler = get_profiler()
        profiler2 = profiler.clone(model=deepcopy(MODEL))
        profiler.compare(profiler2, short_print=False, batch_size=batch_size, device=device, print_mode='debug')
        assert profiler.status_get('layerwise_summary') is not None
        assert profiler2.status_get('layerwise_summary') is not None
        assert all(v1 == v2 for (k1, v1), (k2, v2) in zip(profiler.status_items(), profiler2.status_items())
                   if k1 not in ('layerwise_summary', 'inference_time', 'execution_time', 'layerwise_data'))

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
        profiler.display_status(short_print=False)
        assert profiler.status_get('eval_metric') == rval['acc']
        assert profiler.status_get('b') == rval['b']
        assert profiler.status_get('c') == rval['c']
