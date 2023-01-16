import warnings
import re
from collections import deque

import torch
import torch.jit
import torch.nn as nn

from deeplite.profiler.ir import Graph, Node, Variable

__all__ = ['trace']


class scope_name_workaround(object):
    """
    This class makes sure that torch module scope names are saved in the jit trace
    """
    def __init__(self):
        self.backup = None

    def __enter__(self):
        def _tracing_name(self_, tracing_state):
            if not tracing_state._traced_module_stack:
                return None
            module = tracing_state._traced_module_stack[-1]
            for name, child in module.named_children():
                if child is self_:
                    return name
            return None

        def _slow_forward(self_, *input, **kwargs):
            tracing_state = torch._C._get_tracing_state()
            if not tracing_state or isinstance(self_.forward, torch._C.ScriptMethod):
                return self_.forward(*input, **kwargs)
            if not hasattr(tracing_state, '_traced_module_stack'):
                tracing_state._traced_module_stack = []
            name = _tracing_name(self_, tracing_state)
            if name:
                tracing_state.push_scope('%s[%s]' % (self_._get_name(), name))
            else:
                tracing_state.push_scope(self_._get_name())
            tracing_state._traced_module_stack.append(self_)
            try:
                result = self_.forward(*input, **kwargs)
            finally:
                tracing_state.pop_scope()
                tracing_state._traced_module_stack.pop()
            return result

        self.backup = torch.nn.Module._slow_forward
        setattr(torch.nn.Module, '_slow_forward', _slow_forward)

    def __exit__(self, type, value, tb):
        setattr(torch.nn.Module, '_slow_forward', self.backup)


def filter_torch_scope(node):
    """
    given torch graph scope, returns module name in format matching mod.named_modules()
    """
    scope = node.scopeName() \
                .replace('Flatten/', '', 1) \
                .replace('Flatten', '', 1)
    scope_list = re.findall("\[.*?\]", scope)
    mod_name = ''
    if len(scope_list) >= 2:
        for tok in scope_list[1:]:  # pop model name from path
            mod_name += tok.strip('[').strip(']') + '.'
        mod_name = mod_name.strip('.')
    else:
        # return empty string for model level nodes (inputs, outputs)
        mod_name = ''
    return mod_name


def trace(model, args=(), kwargs=None):
    assert kwargs is None, 'Keyword arguments are not supported for now. ' \
                           'Please use positional arguments instead!'

    with warnings.catch_warnings(record=True), scope_name_workaround():
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
            scope=scope)
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
