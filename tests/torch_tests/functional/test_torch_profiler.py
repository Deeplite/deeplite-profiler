import pytest
from copy import deepcopy
from tests.torch_tests.functional import BaseFunctionalTest, TORCH_AVAILABLE, MODEL, get_profiler
from unittest import mock

class TestTorchProfiler(BaseFunctionalTest):
    def test_empty_display_status(self, *args):
        profiler = get_profiler()
        profiler.display_status()

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


    @mock.patch('deeplite.profiler.utils.AverageAggregator.get', return_value=2)
    def test_compute_network_status(self, *args):
        from deeplite.profiler import Device
        device = Device.CPU
        batch_size = 1
        profiler = get_profiler()
        status = profiler.compute_network_status(batch_size=batch_size, device=device, short_print=False,
                                                 include_weights=True, print_mode='debug')
        assert(status['flops'] == 0.002555904)
        assert(status['total_params'] == 0.002432)
        assert(status['execution_time'] == 2000)
        assert(status['model_size'] == 0.00927734375)
        assert(status['memory_footprint'] == 0.33349609375)
        assert(status['eval_metric'] == 100)
        assert(status['layerwise_summary'])
        assert 'inference_time' in status

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
                   if k1 not in ('layerwise_summary', 'inference_time', 'execution_time'))




