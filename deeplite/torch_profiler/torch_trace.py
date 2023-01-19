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
    mod_name = raw_name.replace('__module.model', '')
    mod_name = mod_name.strip('.')
    return mod_name


def trace(model, args=(), kwargs=None):
    assert kwargs is None, 'Keyword arguments are not supported for now. ' \
                           'Please use positional arguments instead!'
    # get module scope names
    trace_module_map = {}
    def register_submods(mod, prefix):
        for name, child in mod.named_children():
            submod_qualname = prefix + "." + name
            trace_module_map[child] = submod_qualname
            register_submods(child, submod_qualname)

    trace_module_map["__module"] = model
    torch.jit._trace._trace_module_map = trace_module_map
    register_submods(model, "__module")

    graph, _ = torch.jit._get_trace_graph(Flatten(model), args, kwargs)

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
        scope = filter_torch_scope(x)
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
