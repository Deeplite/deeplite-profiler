from abc import ABC, abstractmethod
from collections import namedtuple
from copy import copy, deepcopy
from inspect import signature, Parameter
import time

from .evaluate import EvaluationFunction
from .formatter import getLogger, make_one_model_summary_str, make_two_models_summary_str, \
    default_display_filter_function
from .metrics import EvalMetric, InferenceTime, Comparative
from .utils import cast_tuple

logger = getLogger()

_ProfilerFunctionRegister = namedtuple("_ProfilerFunctionRegister", ('function', 'overriding', 'status'),
                                       module=__name__)


class Profiler(ABC):
    """
    Abstract class for holding profiler functions. It is the entrypoint for profiling a model over some data.
    The user simply needs to register :class:`~ProfilerFunction` to the :class:`~Profiler` instance through
    :method:`~register_profiler_function`. Once this is done, the user can query the `Profiler` for some
    :class:`StatusKey`. `Profiler` will take care to call the relevant `ProfilerFunction` that can compute
    this `StatusKey`.

    Each new framework should subclass this class. See :class:`TorchProfiler` for example using Pytorch.

    NOTE: Status keys are held in a dictionary like fashion. However, the user should never
    access and write directly on this dictionary and should go though the various status methods defined
    on this class. This is to keep some sort of consistency between the `ProfilerFunction` and the values
    they compute. (for ex.: the `recompute` flag of :method:`~compute_status`)

    :param model: An object known by `ProfilerFunction` of that concrete framework (Pytorch, Tensorflow, ...)
    :type model: <native object type>
    :param data_splits: Splits of the data used to for the `ProfilerFunction`.
    :type data_splits: `dict` of `str` to <native data object>
    :param name: A user-friendly name for the model, defaults to "<UnNamedModel>"
    :type name: `str`, optional
    :param display_status_filter_func: A method to display the computed metrics in a user-friendly readable format,
        defaults to deeplite.profiler.formatter.default_display_filter_function
    :type display_status_filter_func: `callable`, optional
    """

    def __init__(self, model, data_splits, name="<UnNamedModel>",
                 display_status_filter_func=default_display_filter_function):
        self._profiling_functions_register = {}
        self.model = model
        self.name = name
        self.data_splits = data_splits
        self.backend = '<None>'
        self.display_status_filter_func = display_status_filter_func

    def register_profiler_function(self, profiler_func, override=False):
        """
        Register a :class:`~ProfilerFunction` with the :class:`~Profiler`. It will become `callable` that the
        `Profiler` recognizes and can fetch through a :method:`~compute_status` call. Since each
        `ProfilerFunction` is associated to different :class:`StatusKey`, it is possible that two
        `ProfilerFunction` compete for the same `StatusKey`. For this reason, the :param:`override` can be
        provided to override a currently registered function with the incoming :param:`profiler_func`.

        :param profiler_func: A callable that computes some `StatusKey`
        :type profiler_func: :class:`~ProfilerFunction`
        :param override: Overrides any `StatusKey` computed with already registered `ProfilerFunction` to new
            `profiler_func`, defaults to False
        :type override: `bool`, optional

        :raises TypeError: Cannot pipe the provided `ProfilerFunction`
        :raises RuntimeError: Two `ProfilerFunction`'s with the same `StatusKey` and both with `override=True`

        :return: `profiler_func` itself
        :rtype: `ProfilerFunction`
        """
        if not profiler_func.can_pipe():
            raise TypeError("Piping keywords check failed. Profiler cannot register profiler function '{}'. "
                            "In normal circumstances, it means this function contains *args or **kwargs in "
                            "the concrete implementation of __call__ signature.")
        status_keys = cast_tuple(profiler_func.get_bounded_status_keys())

        for sk in status_keys:
            sk_name = sk.NAME
            if sk_name in self._profiling_functions_register.keys():
                pf = self._profiling_functions_register[sk_name]
                if pf.overriding and override:
                    raise RuntimeError(
                        "Two profiler functions, '{}' and '{}', both with 'override=True' ".format(
                            profiler_func, pf.function) + "are competing over status key '{}'".format(
                            sk_name))
                if pf.overriding is False and override is False:
                    logger.warning(
                        "Registered profiler function '{}' for status key '{}' is being overridden by '{}'".format(
                        pf.function, status_keys, profiler_func))
                elif pf.overriding is True and override is False:
                    continue
            self._profiling_functions_register[sk_name] = _ProfilerFunctionRegister(
                profiler_func, override, sk)
        return profiler_func

    @classmethod
    @abstractmethod
    def dl_cls(cls):
        """
        Returns the concrete framework :class:`DataLoader`
        """

    @classmethod
    @abstractmethod
    def fp_cls(cls):
        """
        Returns the concrete framework :class:`ForwardPass`
        """

    @classmethod
    def enable_forward_pass_data_splits(cls, data_splits, forward_pass=None):
        """
        Creating deeplite dataloader from the provided data splits dictionary.

        :param native_ds: Splits of the train-test input data (ex: {'train': train_loader,
            'test': test_loader}). Internally each dataloader can one of a PyTorch or Tensorflow dataloader
        :type native_ds: <native data loader object> (ex.: torch.utils.data.DataLoader or tf.data)
        :param forward_pass: A forward pass utility to send the data through the model, defaults to None
        :type forward_pass: :class:`ForwardPass`, optional
        :return: deeplite dataloader
        :rtype: :class:`deeplite.profiler.data_loader.DataLoader`
        """
        if forward_pass is None:
            logger.debug("Defaulting to (x,y) ForwardPass for training dataset")
            forward_pass = cls.fp_cls()(model_input_pattern=(0, '_'), expecting_common_inputs=True)
        if type(forward_pass) is not dict:
            # this assumes it is only enabled for the training set
            forward_pass = {k: forward_pass if k == 'train' else None for k in data_splits.keys()}
        return {k: cls.dl_cls()(v, forward_pass[k]) for k, v in data_splits.items()}

    def compute_status(self, status, recompute=False, **kwargs):
        """
        Computing a :class:`StatusKey`. The method will fetch its associated :class:`~ProfilerFunction`
        and call it with its model, data_splits and `kwargs`.

        :param status: Unique string identifier of wanted `StatusKey`
        :type status: `str`
        :param recompute: If the metric values need to be recomputed, defaults to False
        :type recompute: `bool`, optional

        :raises ValueError: If an unrecognized status key is found
        :raises TypeError: If the profiling function and its status key are matched incorrectly

        :return: Returns what the `ProfilerFunction` returned for this :param:`status`
        """
        if not self.status_contains(status):
            raise ValueError("Requested status '{}' not recognized (valid := {})".format
                             (status, tuple(self.status_keys())))

        if not recompute and self.status_get(status) is not None:
            return self.status_get(status)

        pfr = self._profiling_functions_register[status]
        pf_func = pfr.function
        rval = pf_func.pipe_kwargs_to_call(self.model, self.data_splits, kwargs)

        if isinstance(rval, dict):
            if status not in rval:
                raise TypeError(
                    "Profiler function '{}' returns a dict but not with a key of its associating status '{}'".format(
                        pf_func, status))
            # important!! If the ProfilerFunction returned a dict, it means it is associated with more than
            # one status keys. We can therefore set all the other status keys related to this profiler func
            for k, v in rval.items():
                if k in self._profiling_functions_register:
                    self._profiling_functions_register[k].status.value = v
            rval = rval[status]
        else:
            self._profiling_functions_register[status].status.value = rval

        return rval

    def compute_network_status(self, print_mode=None, recompute=False, short_print=True, **kwargs):
        """
        Calls all registered :class:`~ProfilerFunction` and populates its whole status dictionary.

        :param print_mode: A default printing mode of the logger, defaults to None (no logging)
        :type print_mode: str, optional
        :param recompute: If all the :class:`StatusKey` need to be recomputed, defaults to False
        :type recompute: bool, optional
        :param short_print: Display a short version of the profiled output or a long detailed version,
            defaults to True
        :type short_print: bool, optional

        :return: A dictionary of updated key-value pairs of the metric status key and the metric value
        :rtype: dictionary
        """
        if print_mode:
            getattr(logger, print_mode.lower())("Computing network status...")

        if recompute:
            self.reset_status()

        for sk in self.status_keys():
            if self.status_get(sk) is not None:
                continue
            self.compute_status(sk, recompute=recompute, **kwargs)

        if print_mode:
            self.display_status(print_mode=print_mode, short_print=short_print)

        return dict(self.status_items())

    def compare(self, other, print_mode='info', recompute=False, short_print=True, **kwargs):
        """
        Compare two different :class:`~Profiler`s. The two different profilers could belong to the same model
        or different models. Compare *is* for stdout / logging comparison.

        :param other: Another profiler object
        :type other: :class:`~Profiler`
        :param print_mode: Logger print mode, defaults to 'info'
        :type print_mode: str, optional
        :param recompute: If all the :class:`StatusKey` need to be recomputed, defaults to False
        :type recompute: bool, optional
        :param short_print: If to display a short version of the profiled output or a long detailed version,
            defaults to True
        :type short_print: bool, optional

        :raises ValueError: Can only compare with another profiler instance
        :raises ValueError: Given other profiler does not have the same set of status keys as this profiler
        """
        if not isinstance(other, Profiler):
            raise ValueError("Can only compare with another profiler instance")
        if set(self.status_keys()) != set(other.status_keys()):
            raise ValueError(
                "Given other profiler does not have the same set of status keys as this profiler")

        getattr(logger, print_mode.lower())("Comparing networks status...")

        self.compute_network_status(print_mode=None, recompute=recompute, **kwargs)
        other.compute_network_status(print_mode=None, recompute=recompute, **kwargs)

        self.display_status(other=other, print_mode=print_mode, short_print=short_print)

    def display_status(self, other=None, print_mode='debug', short_print=True):
        """
        Beautiful user-friendly display of either one or two dictionaries of profiler status

        :param other: An optional second status message to compare, defaults to None
        :type other: dictionary, optional
        :param print_mode: Logger print mode, defaults to 'debug'
        :type print_mode: str, optional
        :param short_print: If to display a short version of the profiled output or a long detailed version,
            defaults to True
        :type short_print: bool, optional
        """
        # NOTE: force pop layerwise summary and display in debug mode as it is quite long regardless
        # of the display filter function.
        assert isinstance(print_mode, str) and hasattr(logger, print_mode.lower()), \
            "'print_mode' needs to be a logger level"

        status_dict = self.status_to_dict(to_value=False)
        layerwise_summary_1 = status_dict.pop('layerwise_summary', None)
        status_dict = self.display_status_filter_func(status_dict)
        status_dict['backend'] = self.backend
        status_dict['name'] = self.name

        if other is not None:
            other_status_dict = other.status_to_dict(to_value=False)
            layerwise_summary_2 = other_status_dict.pop('layerwise_summary', None)
            other_status_dict = self.display_status_filter_func(other_status_dict)
            other_status_dict['backend'] = other.backend
            other_status_dict['name'] = other.name
            summary_str = make_two_models_summary_str(status_dict, other_status_dict, short_print=short_print)
        else:
            layerwise_summary_2 = None
            summary_str = make_one_model_summary_str(status_dict, short_print=short_print)

        if not short_print:
            if layerwise_summary_1:
                logger.debug(layerwise_summary_1.value)
            if layerwise_summary_2:
                logger.debug(layerwise_summary_2.value)
        getattr(logger, print_mode.lower())(summary_str)

    def clone(self, model=None, data_splits=None):
        # create a new instance instead of deepcopying stuff
        model = self.model if not model else model
        data_splits = self.data_splits if not data_splits else data_splits
        new = type(self)(model, data_splits)
        new._profiling_functions_register = deepcopy(self._profiling_functions_register)
        if new.model is not self.model:
            new.reset_status()
        new.backend = self.backend
        return new

    # Below are methods to give dict-like access to the internal status storage.
    # Everything is returned as new iterable in order to make sure nothing external
    # corrupts the internal stored values.
    def status_get(self, sk):
        return self._profiling_functions_register[sk].status.value

    def status_contains(self, sk):
        return sk in self._profiling_functions_register

    def status_keys(self):
        # calling iter to be a bit more symmetric with .values() returning a (map)iterable
        return iter(self._profiling_functions_register.keys())

    def status_values(self, to_value=True):
        return map(self.__map_status(to_value), (sv for sv in self._profiling_functions_register.values()))

    def status_items(self, to_value=True):
        return zip(self.status_keys(), self.status_values(to_value))

    def status_to_dict(self, to_value=True):
        mapfunc = self.__map_status(to_value)
        status_dict = {k: mapfunc(pfr) for k, pfr in self._profiling_functions_register.items()}
        return status_dict

    def reset_status(self):
        for pfr in set(self._profiling_functions_register.values()):
            pfr.status.value = None

    @staticmethod
    def __map_status(to_value=True):
        def map_status(pfr):
            status = copy(pfr.status)
            if to_value:
                return status.value
            return status

        return map_status


class ProfilerFunction(ABC):
    """
    Abstract callable which computes any set of :class:`StatusKey` on given model and data.

    Every `ProfilerFunction` is bound to a set of `StatusKey` which they compute their values.
    The `Profiler` seeks this particular function upon a query for a `StatusKey` bounded to it.

    IMPORTANT: The common use case is to request for all profiler's functions to be computed. Since
    by nature its functions are dynamically registered, `Profiler` does not know in advance what
    arguments can be passed to them. It tries to magically pipe the user's `kwargs` to its
    functions. It creates a limitation on this class' __call__ signature when its instance is used
    directly (as opposed to mid-level class only defined as abstraction and never used as instance).
    Because of this magic, the signature of the __call__ cannot contain *args or **kwargs.
    """

    def __init__(self):
        self._call_sign = signature(self)

    def can_pipe(self):
        """
        Returns False if its __call__ signature contains *args or **kwargs else True. This is checked
        through python introspection module inspect.signature.
        """
        return not any(p.kind is Parameter.VAR_KEYWORD or p.kind is Parameter.VAR_POSITIONAL
                       for p in self._call_sign.parameters.values())

    def pipe_kwargs_to_call(self, model, data_splits, kwargs):
        """
        Calls itself using `model`, `data_splits` and an arbitrary `kwargs` dict. It uses the
        `bind` method of `inspect.Signature` object.
        """
        kwargs = {k: v for k, v in kwargs.items() if k in self._call_sign.parameters.keys()}
        bounded_args = self._call_sign.bind(model, data_splits, **kwargs)
        return self(**bounded_args.arguments)

    @abstractmethod
    def get_bounded_status_keys(self):
        """
        Returns a single or a tuple of :class:`StatusKey` that this `ProfilerFunction` computes.
        """

    @abstractmethod
    def __call__(self, model, data_splits, **kwargs):
        """
        Computes values for its :class:`StatusKey`.

        IMPORTANT: If the `ProfilerFunction` compute more than one `StatusKey` values, it should return them
        bundled in a `dict` with their corresponding keys.
        """


class ExternalProfilerFunction(ProfilerFunction):
    """
    :class:`~ProfilerFunction` in which the profiling computation comes from an external callable instead of
    being defined in this __call__ method.

    The function `func` still needs to respect the same limitation that `ProfilerFunction.__call__` has: It
    needs to accept (model, data, <any keywords BUT NOT **kwargs>). It is important to stress out the fact
    that it is not possible to support a **kwargs in the signature. See `ProfilerFunction` for details.
    """

    def __init__(self, func):
        self._call_sign = None
        self._func_sign = signature(func)
        self._func = func

    def can_pipe(self):
        return not any(p.kind is Parameter.VAR_KEYWORD or p.kind is Parameter.VAR_POSITIONAL
                       for p in self._func_sign.parameters.values())

    def pipe_kwargs_to_call(self, model, data_splits, kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k in self._func_sign.parameters.keys()}
        bounded_args = self._func_sign.bind(model, data_splits, **kwargs)
        return self(**bounded_args.arguments)

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)


class ComputeEvalMetric(ExternalProfilerFunction):
    """
    Compute :class:`EvalMetric` from an evaluation function. Examples of those functions can be found
    in deeplite.torch.profiler.torch_inference.

    This makes the additional assumption that an evaluation function is defined only on one split (
    or one data loader)
    """

    def __init__(self, func, key=None, default_split='test', unit_name='', comparative=Comparative.DIFF):
        super().__init__(func)
        self.default_split = default_split
        self.key = key
        self.unit_name = unit_name
        self.comparative = comparative

    def get_bounded_status_keys(self):
        return EvalMetric(unit_name=self.unit_name, comparative=self.comparative), InferenceTime()

    def pipe_kwargs_to_call(self, model, data_splits, kwargs):
        kwargs = kwargs.copy()
        split = kwargs.pop('split', None)
        split = split if split else self.default_split
        return super().pipe_kwargs_to_call(model, data_splits[split], kwargs)

    # TODO support multiple secondary evaluation metrics
    def __call__(self, *args, **kwargs):
        start = time.time()
        rval = self._func(*args, **kwargs)
        inf_time = abs(time.time() - start)

        key = 'akey' if self.key is None else self.key
        rval = EvaluationFunction.filter_call_rval(rval, return_dict=False, return_keys=key)
        return {'eval_metric': rval, 'inference_time': inf_time}
