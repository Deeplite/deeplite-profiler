from abc import ABC, abstractmethod

from .formatter import getLogger
from .utils import timer, Device

logger = getLogger()


class DataLoader(ABC):
    """
    The DataLoader serves as a thin wrapper around the original data loader object. It is mostly used to
    unify calls to the model through the mechanic of the :class:`~ForwardPass`.
    """

    def __init__(self, dl, fp=None):
        self.native_dl = dl
        self.dl_iter = self._create_iter()
        self._forward_pass = None
        self.forward_pass = fp

    @abstractmethod
    def dump_native_info(self):
        """ Dump 'native_dl' picklable state """

    @abstractmethod
    def load_native_info(self, native_state):
        """ Use dumped state from `dump_native_info` to reload itself """
        return self

    @property
    def forward_pass(self):
        return self._forward_pass

    @forward_pass.setter
    def forward_pass(self, fp):
        if fp is None:
            return
        self._forward_pass = fp
        self.forward_pass.infer_sampler(next(iter(self)))

    def __iter__(self):
        self.dl_iter = self._create_iter()
        return self

    def __next__(self):
        """
            Returns a tuple.
        """
        return next(self.dl_iter)

    @abstractmethod
    def __len__(self):
        """
            Returns the length of the data loader. This is typically the number of batches it yields.
        """

    @abstractmethod
    def _create_iter(self):
        """
            Returns an iterator over the native_dl.
        """

    @property
    @abstractmethod
    def batch_size(self):
        """
            Returns the batch size
        """

    @property
    def dataset_size(self):
        return len(self) * self.batch_size

    ######### ForwardPass enabled methods ###########
    def sample_forward(self, model, device=Device.CPU):
        """
        Sample a forward pass over the model on device. It takes the first next item out of the
        original DataLoader iterable as model input.
        """
        if self._forward_pass is None:
            raise TypeError("Cannot sample forward with a DataLoader that has no ForwardPass")
        return self.forward_pass.perform(model, next(iter(self)), device)

    def sample_random_forward(self, model, batch_size=None, device=Device.CPU):
        """
        Sample a random forward pass over the model on device. The model input is random.
        """
        if self._forward_pass is None:
            raise TypeError("Cannot sample random forward with a DataLoader that has no ForwardPass")
        batch_size = self.batch_size if batch_size is None else batch_size
        return self.forward_pass.random_perform(model, batch_size, device)

    @timer
    def timed_forward(self, model, device):
        return self.sample_forward(model, device)

    @timer
    def timed_random_forward(self, model, batch_size, device):
        return self.sample_random_forward(model, batch_size, device)


class ModelInputPattern(tuple):
    """
    Makes sure we can call unambiguously model(*x)

    Tries to make the sampling's life easier as most DataLoader are designed to return more than simply
    what the model takes as input for its forward pass. The most common example is the infamous ('x', 'y')
    input pattern:
        ex.: batch = next(dataloader)
             x, y = batch
             preds = model(x)
             loss(preds, y)

    In this example, model only has one input and it is the first element of the batch tuple while the second
    is for the loss function. The :class:`~ForwardPass` uses the information of the `ModelInputPattern` to
    correctly pipes the batch tuple to the model. In the example above, the `ModelInputPattern` is (0, '_')
    """

    def __new__(cls, args):
        if any(a != '_' and not (isinstance(a, int) and a >= 0) for a in args):
            raise TypeError(
                "ModelInputPattern should be a tuple of positive int or '_' indicating placeholder")
        largs = sorted(a for a in args if isinstance(a, int))
        if not all(x < y for x, y in zip(largs, largs[1:])):
            raise TypeError("Given integers should be strictly increasing")
        return tuple.__new__(cls, args)

    def rearrange_from_pattern(self, x):
        if len(self) == 0 or (len(self) == 1 and self[0] == '_'):
            return tuple()
        if len(self) == 1:
            return (x,)

        if len(x) != len(self):
            raise ValueError("ModelInputPattern (len=%d) does not match output (len=%d) of DataLoader" % (
            len(self), len(x)))
        sorted_int = sorted(a for a in self if isinstance(a, int))
        rval = tuple(x[self.index(i)] for i in sorted_int)

        return rval


class ForwardPass(ABC):
    """
    Unifies the output of a DataLoader and a model call. Its main entry points are :method:`~perform` and
    :method:`~random_perform`. It offers various mid-level automatic implementations of some functionalities
    if some assumptions are valid. There are only two assumptions that the user needs to inform,
    `model_input_pattern` and `expecting_common_inputs`.

    If a `model_input_pattern` is provided as a
    special tuple :class:`~ModelInputPattern`, the `ForwardPass` then knows how to filter the relevant
    element out of the batch tuple and perform a call on the model with them. Therefore, the user does not
    need to implement a custom :method:`~perform` method.

    If `expecting_common_inputs` is True, then the input is expected to be "standard". By that we mean that
    it follows the convention that we expect a tuple of tensors to be a tuple of
    common types i.e.: tuple, list, dict, numpy or framework tensor. The container
    types should have items of type numpy or framework tensors.
    Following this to be True, then `ForwardPass` can extract the shapes of the model's inputs and
    generate random tensors that will be compatible with the model's call signature. Therefore, the user
    does not need to implement a custom :method:`~random_perform`.

    In the case that `model_input_pattern` is provided and `expecting_common_inputs` is True, the the
    user has nothing to override! If the use case is too complex for these assumptions to work, the user
    is always free to override any of the `ForwardPass` method.

    The design goal of this object is that a :class:`ProfilerFunction` can be written more generally and have
    a unified way to pipe data to model using this object. All default functions written here assumes the
    `ForwardPass`'s logic.
    """

    def __new__(cls, model_input_pattern=None, expecting_common_inputs=True):
        # checking the __dict__ over hasattr to avoid True because of super()
        if model_input_pattern is None and 'extract_model_inputs' not in cls.__dict__:
            raise TypeError(
                "Cannot instantiate %s without a 'model_input_pattern' keyword " % cls.__name__ + \
                "and without a 'extract_model_inputs' method.")
        if not expecting_common_inputs and 'create_random_model_inputs' not in cls.__dict__:
            raise TypeError(
                "Cannot instantiate %s with 'expecting_common_inputs' at False " % cls.__name__ + \
                "and without a 'create_random_model_inputs' method.")
        return super().__new__(cls)

    def __getnewargs__(self):
        # bypyass the Type error checks when creating __new__ at pickling
        return 'dummy', True

    def __init__(self, model_input_pattern=None, expecting_common_inputs=True):
        if isinstance(model_input_pattern, tuple):
            model_input_pattern = ModelInputPattern(model_input_pattern)
        self.mip = model_input_pattern
        self.expecting_common_inputs = expecting_common_inputs
        self._tensor_sampler = None

    def perform(self, model, batch, device):
        """
        Performs a forward pass on the model with a batch coming out of the data loader. Returns what
        the model returns.
        """
        x = self.extract_model_inputs(batch)
        return self.model_call(model, x, device)

    def random_perform(self, model, batch_size, device):
        """
        Performs a forward pass with random input of corresponding batch size.
        """
        x = self.create_random_model_inputs(batch_size)
        return self.model_call(model, x, device)

    @abstractmethod
    def model_call(self, model, x, device):
        """
        Call the model with 'x' extracted from a loader's batch and on device 'device'.
        """
        raise NotImplementedError

    # @conditionnal_method('expecting_common_inputs')
    def create_random_model_inputs(self, batch_size):
        """
        Create a compatible random input of corresponding batch size. Compatible in the sense that
        `model_call` can run without crashing this return value.

        Default implementation is provided if the ForwardPass is instantiated expecting common inputs.
        """
        if self.expecting_common_inputs:
            return self._tensor_sampler.create_random_tensors(batch_size)
        raise NotImplementedError

    # @conditionnal_method('model_input_patterns')
    def extract_model_inputs(self, batch):
        """
        Extract a compatible input from a loader's batch. Compatible in the sense that
        `model_call` can run without crashing this return value.

        Default implementation is provided if the ForwardPass is instantiated with a pattern.
        """
        if self.mip is not None:
            return self.mip.rearrange_from_pattern(batch)
        raise NotImplementedError

    # @conditionnal_method('expecting_common_inputs')
    def get_model_input_shapes(self):
        """
        Returns a tuple of all input shapes that are fed to the model.

        Default implementation is provided if the ForwardPass is instantiated expecting common inputs.
        """
        if not self.expecting_common_inputs:
            raise NotImplementedError
        return self._tensor_sampler.get_flat_shapes_tuple()

    @property
    @abstractmethod
    def _tensor_sampler_cls(self):
        """
        Framework specific :class:`~TensorSampler` class
        """
        raise NotImplementedError

    def infer_sampler(self, batch):
        if not self.expecting_common_inputs:
            logger.debug("Cannot infer sampler for this forward pass")
            return
        x = self.extract_model_inputs(batch)
        self._tensor_sampler = self._tensor_sampler_cls(x)


class TensorSampler(ABC):
    """
    This should bridge the output of the data loader and enable model forward calls with the
    appropriate format / device (ex.: for a random input).
    It follows the convention that we expect a tuple of tensors to be a tuple of
    common types i.e.: tuple, list, dict, numpy or framework tensor. The container
    types should have items of type numpy or framework tensors.

    .. important:: We do NOT search deeper in the container types, standardize should raise
      UnrecognizedInputError and all other methods should beforehand standardize their inputs
    """

    def __init__(self, tensors_tuple_sample):
        sample = self.standardize_tensors(tensors_tuple_sample)
        # tensors_info is structured as the tensors_tuple_sample, it preserves the model input structure
        self.tensors_info = self.get_tensors_info(sample)

    @abstractmethod
    def _standardize_tensor(self, x):
        raise NotImplementedError

    @abstractmethod
    def _create_random_tensor(self, x_info, batch_size):
        raise NotImplementedError

    @abstractmethod
    def _get_info(self, x):
        raise NotImplementedError

    def _loop_over_tensors_tuple(self, tensors_tuple, single_tensor_function):
        rval = []
        for tp in tensors_tuple:
            if isinstance(tp, (tuple, list)):
                tp_ = type(tp)(map(single_tensor_function, tp))
            elif isinstance(tp, dict):
                tp_ = {k: single_tensor_function(v) for k, v in tp.items()}
            else:
                tp_ = single_tensor_function(tp)
            rval.append(tp_)
        return tuple(rval)

    def standardize_tensors(self, tensors_tuple):
        return self._loop_over_tensors_tuple(tensors_tuple, self._standardize_tensor)

    def create_random_tensors(self, batch_size):
        f = lambda x: self._create_random_tensor(x, batch_size)
        return self._loop_over_tensors_tuple(self.tensors_info, f)

    def get_tensors_info(self, tensors_tuple):
        f = lambda x: TensorInfo(*self._get_info(x))
        infos_tuple = self._loop_over_tensors_tuple(tensors_tuple, f)
        return infos_tuple

    def get_flat_shapes_tuple(self):
        rval = []
        for info in self.tensors_info:
            if isinstance(info, (tuple, list)):
                rval.extend(map(lambda s: s.shp, info))
            elif isinstance(info, dict):
                rval.extend(map(lambda s: s.shp, info.values()))
            else:
                rval.append(info.shp)
        return tuple(rval)


class TensorInfo:
    __slots__ = ('shp', 'dtype')

    def __init__(self, shp, dtype):
        self.shp = shp
        self.dtype = dtype
