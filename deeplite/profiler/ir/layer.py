__all__ = ['Layer']


class Layer:
    def __init__(self, name, inputs, outputs, weights=None, bias=None, scope=None):
        self._name = name
        self._inputs = inputs
        self._outputs = outputs
        self._weights = weights
        self._bias = bias
        self._scope = scope

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name.lower()

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, inputs):
        self._inputs = inputs

    @property
    def outputs(self):
        return self._outputs

    @outputs.setter
    def outputs(self, outputs):
        self._outputs = outputs

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, bias):
        self._bias = bias

    @property
    def scope(self):
        return self._scope

    @scope.setter
    def scope(self, scope):
        self._scope = scope.lower()

    def __repr__(self):
        text = "Node (name: {}, inputs: {}, outputs: {}, w: {}, b: {})".format(
                self.name, len(self.inputs), len(self.outputs), self.weights,
                self.bias)
        return text

