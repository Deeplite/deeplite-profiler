from copy import deepcopy
import time
import sys
import torch
import warnings

from deeplite.profiler import Profiler, ProfilerFunction
from deeplite.profiler.metrics import *
from deeplite.profiler.utils import AverageAggregator, Device
from deeplite.profiler.formatter import getLogger

from deeplite.torch_profiler.torch_trace import trace
from deeplite.profiler.ir import Layer, Tensor
from deeplite.profiler.memory_allocation.placer import Placer
from deeplite.profiler.report import Report
from deeplite.torch_profiler.torch_handlers import torch_handlers

from .torch_data_loader import TorchDataLoader, TorchForwardPass

logger = getLogger(__name__)

__all__ = ['TorchProfiler', 'ComputeComplexity', 'ComputeExecutionTime']


def get_total_params(model):
    total = 0
    for p in model.parameters():
        if p.requires_grad:
            total += p.numel()
    return total


def get_params(model, node_complexity_map, num_bytes):  # for each module check for hook? They can be registered like handles
    total = 0
    counted_params = set()
    for name, module in model.named_modules():
        if name in node_complexity_map:
            param_size_map = node_complexity_map[name].get('param_size', None)
            if param_size_map is not None:
                assert isinstance(param_size_map, dict)
                for p_name, param_size in param_size_map.items():
                    p = getattr(module, p_name)
                    counted_params.add(p)
                    total += p.numel() * param_size  # param_size = bytes per param
    for p in model.parameters():
        if p.requires_grad and p not in counted_params:
            total += p.numel() * num_bytes
    return total


def get_macs(graph, node_complexity_map, reduction=sum):
    results = dict()
    for node in graph.nodes:
        for operators, func in torch_handlers:
            if isinstance(operators, str):
                operators = [operators]
            if node.operator in operators:
                if node.scope in node_complexity_map:
                    flops = node_complexity_map[node.scope].get('flops', None)
                    if flops is None:
                        raise RuntimeError('Invalid compute_module_complexity() output')
                    results[node] = flops
                elif func is not None:
                    results[node] = func(node)
                break
        # else:
        #     warnings.warn('No handlers found: "{}". Skipped.'.format(
        #         node.operator))

    if reduction is not None:
        return reduction(results.values())
    else:
        return results

def get_nodes(graph):
    nodes = []
    for i, node in enumerate(graph.nodes):
        if 'aten' in node.operator: # aten ops
            inputs = []
            outputs = []
            weights = []
            bias = []
            if 'conv' in node.operator:
                weights = node.inputs[1].shape
                if node.inputs[2].shape is not None:
                    bias = node.inputs[2].shape
                inputs.append(Tensor(name=node.inputs[0].name,
                        dtype=node.inputs[0].dtype, shape=node.inputs[0].shape, scope=node.scope))
            elif 'mm' in node.operator:
                weights = node.inputs[2].shape
                if node.inputs[0].shape is not None:
                    bias = node.inputs[0].shape
                inputs.append(Tensor(name=node.inputs[1].name,
                        dtype=node.inputs[1].dtype, shape=node.inputs[1].shape, scope=node.scope))
            elif 'matmul' in node.operator:
                weights = node.inputs[1].shape
                inputs.append(Tensor(name=node.inputs[0].name,
                        dtype=node.inputs[0].dtype, shape=node.inputs[0].shape, scope=node.scope))
            elif node.operator in ['aten::batch_norm', 'aten::instance_norm']:
                if node.inputs[1].shape is not None:
                    weights = node.inputs[1].shape # to double-chek
                    bias = node.inputs[2].shape #
                inputs.append(Tensor(name=node.inputs[0].name,
                        dtype=node.inputs[0].dtype, shape=node.inputs[0].shape, scope=node.scope))
            elif node.operator in ['aten::layer_norm', 'aten::group_norm']:
                if node.inputs[2].shape is not None:
                    weights = node.inputs[2].shape # to double-chek
                    bias = node.inputs[2].shape # ???
                inputs.append(Tensor(name=node.inputs[0].name,
                        dtype=node.inputs[0].dtype, shape=node.inputs[0].shape, scope=node.scope))
            elif node.operator == 'aten::t':  # skip weight transpose, shouldn't be present
                continue
            else:
                for x in node.inputs:
                    if x.shape is not None:
                        if x.ndim > 1:
                            inputs.append(Tensor(name=x.name, dtype=x.dtype,
                                    shape=x.shape, scope=node.scope))
            for x in node.outputs:
                outputs.append(Tensor(name=x.name, dtype=x.dtype,
                    shape=x.shape, scope=node.scope))

            nodes.append(Layer(name="{}_{}".format(i, node.operator),
                inputs=inputs, outputs=outputs, weights=weights, bias=bias, scope=node.scope))

    return nodes


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
    def __init__(self, num_bytes=4, export=False):
        super().__init__()
        self.num_bytes = num_bytes
        self.export = export

    @classmethod
    def _get_bounded_status_keys_cls(cls):
        return Flops, TotalParams, ModelSize, MemoryFootprint, LayerwiseSummary, LayerwiseData

    def get_bounded_status_keys(self):
        sk_cls = self._get_bounded_status_keys_cls()
        rval = tuple(cls() for cls in sk_cls)
        return rval

    def __call__(self, model, data_splits, batch_size=1, device=Device.CPU):
        sk_cls = self._get_bounded_status_keys_cls()
        rval = self._compute_complexity(model, data_splits['train'],
                batch_size=batch_size, device=device)
        assert len(sk_cls) == len(rval)
        return {x.NAME: y for x, y in zip(sk_cls, rval)}


    def _compute_complexity(self, model, dataloader, batch_size=1,
            device=Device.CPU):
        model.eval()
        inputs = dataloader.forward_pass.create_random_model_inputs(batch_size)
        assert isinstance(inputs, tuple)
        node_complexity_map = {}
        graph = trace(model.cpu(), inputs, node_complexity_map)
        aten_nodes = get_nodes(graph)
        placer = Placer(aten_nodes)
        aten_nodes = placer.place(num_bytes=self.num_bytes)
        report = Report(aten_nodes, export=self.export, filename='outmodel') # filename
        df = report.get_stats(verbose=self.export)

        params_size = get_params(model.cpu(), node_complexity_map, num_bytes=self.num_bytes)
        macs = get_macs(graph, node_complexity_map)
        macs /= 1e9
        model_size = (params_size) / (2**20)
        total_params = get_total_params(model) / 1e6
        peak_memory = df.ram.max() * self.num_bytes / (2**20) / batch_size
        keys = ['weight', 'input_shape', 'output_shape', 'scope', 'ram']
        header = ['Weight', 'Input Shape','Output Shape', 'Scope', 'Memory']
        df_str = df[keys].to_string(header=header, col_space=10, justify='right', max_colwidth=40)
        ncols = df_str.find('\n') + 1
        df_str = '-'*ncols + '\n' + df_str[:ncols] + '='*ncols + '\n' + \
                df_str[ncols:] + '\n' + '-'*ncols + '\n'
        return macs, total_params, model_size, peak_memory, df_str, df


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
