from copy import deepcopy
import time
import sys

import numpy as np
from thirdparty import ptflops as flops_counter
import torch

from deeplite.profiler import Profiler, ProfilerFunction
from deeplite.profiler.metrics import *
from deeplite.profiler.utils import AverageAggregator, Device
from deeplite.profiler.formatter import getLogger

from .torch_data_loader import TorchDataLoader, TorchForwardPass

logger = getLogger()

__all__ = ['TorchProfiler', 'ComputeComplexity', 'ComputeExecutionTime']


class TorchProfiler(Profiler):
    def __init__(self, model, data_splits, **kwargs):
        super().__init__(model, data_splits, **kwargs)
        self.backend = 'TorchBackend'

    @classmethod
    def dl_cls(cls):
        return TorchDataLoader

    @classmethod
    def fp_cls(cls):
        return TorchForwardPass

    @staticmethod
    def model_to_device(m, device):
        m = m.cpu() if device == Device.CPU else m.cuda()
        return m


class ComputeComplexity(ProfilerFunction):
    @classmethod
    def _get_bounded_status_keys_cls(cls):
        return Flops, TotalParams, ModelSize, MemoryFootprint, LayerwiseSummary

    def get_bounded_status_keys(self):
        sk_cls = self._get_bounded_status_keys_cls()
        rval = tuple(cls() for cls in sk_cls)
        return rval

    def __call__(self, model, data_splits, batch_size=1, device=Device.CPU, include_weights=True):
        sk_cls = self._get_bounded_status_keys_cls()
        rval = self._compute_complexity(model, data_splits['train'], batch_size=batch_size, device=device,
                                        include_weights=include_weights)
        assert len(sk_cls) == len(rval)
        return {x.NAME: y for x, y in zip(sk_cls, rval)}

    # This is adapted from ptflops
    # HAS TO RETURN A TUPLE IN THE SAME ORDER OF STATUSKEYS
    def _compute_complexity(self, model, dataloader, batch_size=1, device=Device.CPU, include_weights=True):
        temp_model = deepcopy(model)
        forward_pass = dataloader.forward_pass

        with torch.no_grad():
            # synchronize gpu time and measure fp
            temp_model = TorchProfiler.model_to_device(temp_model, device)
            flops_model = flops_counter.add_flops_counting_methods(temp_model)
            flops_model.eval()

            # DRY RUNS
            for _ in range(5):
                if device == Device.GPU:
                    torch.cuda.synchronize()
                    forward_pass.random_perform(flops_model, batch_size=batch_size, device=device)

            # Add hooks and start the run
            flops_model.start_flops_count(ost=sys.stdout, verbose=False, ignore_list=[])
            if device == Device.GPU:
                torch.cuda.synchronize()

            t0 = time.time()
            forward_pass.random_perform(flops_model, batch_size=batch_size, device=device)
            flops_count, params_count, model_size, activation_size, summary_str = \
                flops_model.compute_average_flops_cost("", t0)

            flops_model.stop_flops_count()

        flops = flops_count / 1e9  # Giga Flops
        params = params_count / 1e6  # Million Flops
        model_size = abs(model_size / (1024 ** 2.))  # Convert bytes to MB
        shapes_tuple = forward_pass.get_model_input_shapes()
        total_input_size = abs(
            np.prod(sum(shapes_tuple, ()))) * batch_size * 4  # Multiply with batch_size and bit precision
        memory_footprint = abs((total_input_size + activation_size) / (1024 ** 2.))
        total_memory_footprint = model_size + memory_footprint if include_weights else memory_footprint

        return flops, params, model_size, total_memory_footprint, summary_str


class ComputeExecutionTime(ProfilerFunction):
    def get_bounded_status_keys(self):
        return ExecutionTime()

    def __call__(self, model, data_splits, split='train', batch_size=1, device=Device.CPU):
        def timer(f, aggr):
            if device == Device.CPU:
                def call_timing_decorator(*args, **kwargs):
                    start_time = time.perf_counter()
                    rval = f(*args, **kwargs)
                    end_time = time.perf_counter()
                    aggr.update(end_time - start_time)
                    return rval
            else:
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

                def call_timing_decorator(*args, **kwargs):
                    starter.record()
                    rval = f(*args, **kwargs)
                    ender.record()
                    torch.cuda.synchronize()
                    aggr.update(starter.elapsed_time(ender))
                    return rval
            return call_timing_decorator

        dataloader = data_splits[split]
        temp_model = deepcopy(model)
        temp_model.eval()
        aggregator = AverageAggregator()
        og_call = type(temp_model).__call__
        type(temp_model).__call__ = timer(type(temp_model).__call__, aggregator)

        with torch.no_grad():
            # synchronize gpu time and measure fp
            temp_model = TorchProfiler.model_to_device(temp_model, device)

            # DRY RUNS
            for _ in range(5):
                if device == Device.GPU:
                    torch.cuda.synchronize()
                _ = dataloader.timed_random_forward(temp_model, batch_size=batch_size, device=device)
            # resets the aggregator and makes sure it was updated in the decorator
            assert aggregator.get() != 0

            # START BENCHMARKING
            steps = 10
            fp_time = 0.
            for _ in range(steps):
                if device == Device.GPU:
                    torch.cuda.synchronize()
                fp_time += dataloader.timed_random_forward(temp_model, batch_size=batch_size, device=device)
            fp_time = fp_time / steps / batch_size

        type(temp_model).__call__ = og_call

        # execution_time = fp_time * 1000
        execution_time = aggregator.get() / batch_size
        if device == Device.CPU:
            execution_time *= 1000
        return execution_time
