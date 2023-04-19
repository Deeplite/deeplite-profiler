import warnings
import re
from collections import deque

import torch
import torch.jit
import torch.nn as nn

from deeplite.profiler.ir import Graph, Node, Variable

__all__ = ['trace']


def filter_torch_scope(node):
    """
    given torch graph scope, returns module name in format matching mod.named_modules()
    """
    raw = node.scopeName()
    raw_split = raw.split('/')
    raw_name = raw_split[-1]
    mod_name = raw_name.replace('__module.', '')
    mod_name = mod_name.strip('.')
    return mod_name


class custom_complexity_hook(object):
    def __init__(self, node_complexity_map):
        self.backup = None
        self.node_complexity_map = node_complexity_map

    def __enter__(self):
        def _slow_forward(self_, *input, **kwargs):  # self_ is module
            result = self.backup(self_, *input, **kwargs)
            if hasattr(self_, 'compute_module_complexity'):
                rval = self_.compute_module_complexity(input, result, **kwargs)
                # what to give it as a key?
                recording_scopes = torch.jit._trace._trace_module_map is not None
                if recording_scopes:
                    key = torch.jit._trace._trace_module_map[self_]
                    self.node_complexity_map[key] = rval

            return result

        self.backup = torch.nn.Module._slow_forward
        setattr(torch.nn.Module, '_slow_forward', _slow_forward)

    def __exit__(self, type, value, tb):
        setattr(torch.nn.Module, '_slow_forward', self.backup)



def trace(model, args=(), node_complexity_map=None):
    trace_module_map = {}
    def register_submods(mod, prefix):
        for name, child in mod.named_children():
            submod_qualname = prefix + "." + name
            trace_module_map[child] = submod_qualname
            register_submods(child, submod_qualname)

    trace_module_map["__module"] = model
    torch.jit._trace._trace_module_map = trace_module_map
    register_submods(model, "__module")

    with custom_complexity_hook(node_complexity_map) as work:
        graph, _ = torch.jit._get_trace_graph(Flatten(model), args, kwargs=None)

    variables = dict()
    for x in graph.nodes():
        for v in list(x.inputs()) + list(x.outputs()):
            if 'tensor' in v.type().kind().lower():
                variables[v] = Variable(
                    name=v.debugName(),
                    dtype=v.type().scalarType(),
                    shape=v.type().sizes(),
                )
            else:
                variables[v] = Variable(
                    name=v.debugName(),
                    dtype=str(v.type()),
                )

    nodes = []
    for x in graph.nodes():
        raw_scope = x.scopeName()
        scope = filter_torch_scope(x)
        if raw_scope in node_complexity_map:
            if scope not in node_complexity_map:
                node_complexity_map[scope] = node_complexity_map[raw_scope]
                assert raw_scope != scope
                node_complexity_map.pop(raw_scope)
            else:
                raise RuntimeError("Repeated module complexity functions")
        node = Node(
            operator=x.kind(),
            attributes={
                s: getattr(x, x.kindOf(s))(s)
                for s in x.attributeNames()
            },
            inputs=[variables[v] for v in x.inputs() if v in variables],
            outputs=[variables[v] for v in x.outputs() if v in variables],
            scope=scope
        )
        nodes.append(node)

    graph = Graph(
        name=model.__class__.__module__ + '.' + model.__class__.__name__,
        variables=[v for v in variables.values()],
        inputs=[variables[v] for v in graph.inputs() if v in variables],
        outputs=[variables[v] for v in graph.outputs() if v in variables],
        nodes=nodes,
    )
    return graph


def flatten(inputs):
    queue = deque([inputs])
    outputs = []
    while queue:
        x = queue.popleft()
        if isinstance(x, (list, tuple)):
            queue.extend(x)
        elif isinstance(x, dict):
            queue.extend(x.values())
        elif isinstance(x, torch.Tensor):
            outputs.append(x)
    return outputs


class Flatten(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        return flatten(outputs)
