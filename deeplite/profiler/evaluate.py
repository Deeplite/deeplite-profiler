from abc import ABC, abstractmethod
from .utils import Device, cast_tuple


class EvaluationFunction(ABC):
    @abstractmethod
    def __call__(self, mode, data_loader, device=Device.CPU, **kwargs):
        raise NotImplementedError

    @staticmethod
    def filter_call_rval(rval, return_dict=None, return_keys=None, key_for_non_dict=None):
        """
        Filter through what is returned by __call__
        TODO: Complete docstring
        """
        assert return_dict in (None, False, True), "'return_dict' should be None, False or True"

        # a simple fallthrough
        if return_dict is None:
            return rval

        # the caller does not wants a dict
        if not return_dict:
            if not isinstance(rval, dict):
                return rval
            if len(rval) == 1:
                return list(rval.values())[0]

            return_keys = cast_tuple(return_keys)
            if len(return_keys) != 1:
                raise ValueError(
                    "__call__ returned a dict but 'return_dict' is False, 'return_keys' has to be "
                    "convertible in a single key (ex.: a single element iterable with the hashable key, "
                    "the hashable key, etc)")
            return rval[return_keys[0]]

        # at this point, the caller wants a dict but rval is not a dict
        if not isinstance(rval, dict):
            if key_for_non_dict is not None:
                return {key_for_non_dict: rval}
            raise ValueError(
                "__call__ did not return a dict but 'return_dict' is True, 'key_for_non_dict' has "
                "to be provided")

        # at this point, the caller wants a dict and rval is a dict
        if return_keys is None:
            return_keys = '__all__'

        if return_keys == '__all__':
            return rval

        return_keys = cast_tuple(return_keys)
        return {k: rval[k] for k in return_keys}
