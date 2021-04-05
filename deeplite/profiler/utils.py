from abc import ABC, abstractmethod
import functools, time
from collections.abc import Iterable
from enum import Enum


class Device(Enum):
    CPU = 'cpu'
    GPU = 'gpu'


class Aggregator(ABC):
    __slots__ = tuple()

    @abstractmethod
    def update(self, value):
        """ Update the value"""

    @abstractmethod
    def get(self):
        """ Get the value and possibly resetting the state"""

class AverageAggregator(Aggregator):
    __slots__ = ('value', 'i')

    def __init__(self):
        self.value = 0
        self.i = 0

    def update(self, value):
        self.value += value
        self.i += 1

    def get(self):
        try:
            v = self.value / self.i
        except ZeroDivisionError:
            return 0
        self.__init__()
        return v

    def get_sum(self):
        v = self.value
        self.__init__()
        return v


def _cast_iterable(x, iter_type):
    if isinstance(x, str):
        return iter_type([x])
    if isinstance(x, iter_type):
        return x
    if isinstance(x, Iterable):
        return iter_type(x)
    return None

def cast_tuple(x):
    if x is None:
        return tuple()
    y = _cast_iterable(x, tuple)
    if y is None:
        y = (x,)
    return y

def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()   
        value = func(*args, **kwargs)
        end_time = time.perf_counter()     
        run_time = end_time - start_time   
        #print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return run_time
    return wrapper_timer