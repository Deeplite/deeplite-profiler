from abc import ABC, abstractmethod
from enum import Enum

__all__ = ["Comparative", "LayerwiseSummary", "Flops", "ModelSize", "ExecutionTime", "TotalParams",
           "MemoryFootprint", "EvalMetric", "InferenceTime"]


class Comparative(Enum):
    DIFF = 'diff'  # ref - value
    DIV = 'div'  # ref / value
    RECIPROCAL = 'reciprocal'  # 1 / (ref / value) => value / ref
    NONE = 'none'


def compare_status_values(comp, x, y):
    if comp == Comparative.DIFF:
        return x - y
    if comp in (Comparative.DIV, Comparative.RECIPROCAL):
        rval = x / y
        if comp == Comparative.RECIPROCAL:
            rval = 1 / rval
        return rval
    if comp == Comparative.NONE:
        return None
    raise ValueError("Unknown Comparative '{}'".format(comp))


class StatusKey(ABC):
    """
    Fundamental building block of the :class:`Profiler`'s mechanics. Every :class:`ProfilerFunction` are tied
    to one or more :class:`~StatusKey`. The bare bone class only expects an attribute `value` and a
    class attribute `NAME`. This one is very important and represents the `str` key by which to fetch
    this `StatusKey`.

    NOTE: When making new `StatusKey`, care should be taken to not have `NAME` clash among them. This is not
    checked at runtime, but a `Profiler` should have a unique set of key `NAME`.
    """
    NAME = NotImplemented

    def __init__(self):
        self.value = None


class LayerwiseSummary(StatusKey):
    NAME = 'layerwise_summary'


class Metric(StatusKey):
    """
    Metric is the most used interface of :class:`~StatusKey`. A :class:`~Metric` usually has a description,
    a human readable name, units (seconds, meters...), ways to compare against another `Metric` value of
    the same class and optionally value formatting.

    Most of the methods are used to properly format `Metric` when displaying them and comparing them.
    """

    @staticmethod
    @abstractmethod
    def description():
        """
        String description of what this metric is supposed to have computed
        """

    @staticmethod
    @abstractmethod
    def friendly_name():
        """
        String friendly human recognizable name for this metric
        """

    @abstractmethod
    def get_units(self):
        """
        String units for this metric.
        **NOTE: return '' if there is none
        """

    @abstractmethod
    def get_comparative(self):
        """
        How does a value of this metric compare with another value of the same metric. Refer to
        :method:`~compare_status_values` function.

        Should return a `Comparative` Enum type or a `str` parsable into a `Comparative`
        """

    def get_value(self):
        """
        Optionally rescale the value. Mostly used for formatting when sending to pretty format on stdout
        """
        return self.value


class Flops(Metric):
    NAME = 'flops'

    @staticmethod
    def description():
        return "Summation of Multiply-Add Cumulations (MACs) per single image (batch_size=1)"

    @staticmethod
    def friendly_name():
        return 'Computational Complexity'

    def get_comparative(self):
        return Comparative.RECIPROCAL

    def get_value(self):
        if self.value > 1e-3:
            return self.value
        if self.value > 1e-6:
            return self.value * 1024
        return self.value * (1024 ** 2)

    def get_units(self):
        if self.value > 1e-3:
            return 'GigaMACs'
        if self.value > 1e-6:
            return 'MegaMACs'
        return 'KiloMACs'


class TotalParams(Metric):
    NAME = 'total_params'

    @staticmethod
    def description():
        return "Total number of parameters (trainable and non-trainable) in the model"

    @staticmethod
    def friendly_name():
        return "Total Parameters"

    def get_comparative(self):
        return Comparative.RECIPROCAL

    # TODO this should be dynamic
    def get_units(self):
        return 'Millions'


class ExecutionTime(Metric):
    NAME = 'execution_time'

    @staticmethod
    def description():
        return "On current device, time required for the forward pass per single image"

    @staticmethod
    def friendly_name():
        return "Execution Time"

    def get_comparative(self):
        return Comparative.RECIPROCAL

    # TODO this should be dynamic
    def get_units(self):
        return 'ms'


class ModelSize(Metric):
    NAME = 'model_size'

    @staticmethod
    def description():
        return "Memory consumed by the parameters (weights and biases) of the model"

    @staticmethod
    def friendly_name():
        return "Model Size"

    def get_comparative(self):
        return Comparative.RECIPROCAL

    # TODO this should be dynamic
    def get_units(self):
        return 'MB'


class MemoryFootprint(Metric):
    NAME = 'memory_footprint'

    @staticmethod
    def description():
        return "Total memory consumed by parameters and activations per single image (batch_size=1)"

    @staticmethod
    def friendly_name():
        return "Memory Footprint"

    def get_comparative(self):
        return Comparative.RECIPROCAL

    # TODO this should be dynamic
    def get_units(self):
        return 'MB'


class EvalMetric(Metric):
    NAME = 'eval_metric'

    def __init__(self, unit_name='', comparative=Comparative.DIFF):
        super().__init__()
        self.unit_name = unit_name
        self.comparative = comparative

    @staticmethod
    def description():
        return "Computed performance of the model on the given data"

    @staticmethod
    def friendly_name():
        return "Evaluation Metric"

    def get_units(self):
        return self.unit_name

    def get_comparative(self):
        return self.comparative


class InferenceTime(Metric):
    NAME = 'inference_time'

    @staticmethod
    def description():
        return "On current device, time required to run the evaluation function"

    @staticmethod
    def friendly_name():
        return "Inference Time"

    def get_comparative(self):
        return Comparative.RECIPROCAL

    def get_units(self):
        return 's'
