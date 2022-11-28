from abc import abstractmethod
from collections import OrderedDict
import logging

from .metrics import Comparative, compare_status_values, Metric, DynamicEvalMetric
from .utils import cast_tuple


class _LoggerHolder:
    """
    Holds the logger for the profiler module and redirect any calls made to its held logger. If none is held
    then defaults to logging.getLogger('deeplite_profiler'). This allows customization of the logging
    being used in the profiler by an external library by importing setLogger.
    """
    __singleton = False

    def __new__(cls, *args, **kwargs):
        if cls.__singleton:
            raise TypeError("This class is a singleton!")
        cls.__singleton = True
        return super().__new__(cls)

    def __init__(self):
        self._logger = None

    @property
    def logger(self):
        if self._logger is None:
            self._logger = logging.getLogger('deeplite_profiler')
            self._logger.debug("Defaulting to logging.getLogger('deeplite_profiler')'s logger")
        return self._logger

    @logger.setter
    def logger(self, new_logger):
        self._logger = new_logger
        self._logger.debug("Profiler's logger set to {}".format(new_logger))

    def __getattribute__(self, item):
        if item in ('logger', '_logger'):
            return object.__getattribute__(self, item)
        return getattr(self.logger, item)
_logger_holder = _LoggerHolder()


# silence the name for now
def getLogger(name=None):
    """
    Returns an instance of _LoggerHolder. Even though the name is misleading, any code using this return value
    should use it as if it was a logger. This enables to dynamically change the logger of the library after
    the modules were loaded and their logger declared in their headers.
    """
    return _logger_holder


def setLogger(logger):
    _logger_holder.logger = logger


class NotComputedValue:
    def __format__(self, format_spec):
        s = '<NotComputed>'
        if '>' in format_spec:
            format_spec = format_spec.split('>')[-1]
            if '.' in format_spec:
                format_spec = format_spec.split('.')[0]
            if format_spec.isdigit():
                s = ' ' * max(int(format_spec) - len(s), 0) + s
        return s

    def __repr__(self):
        return '<NotComputed>'

    def __str__(self):
        return '<NotComputed>'


def parse_metric(metric):
    text = metric.friendly_name()
    if metric.value is None:
        value = NotComputedValue()
        units = ''
        comparative = Comparative.NONE
    else:
        value = metric.get_value()
        units = metric.get_units()
        units = '' if units is None else units
        comparative = metric.get_comparative()
    description = metric.description()
    return text, value, units, comparative, description


class Display:
    def __init__(self, logger, order=None, include_leftovers=True, exclude=None):
        self.logger = logger
        self.order = order
        self.include_leftovers = include_leftovers
        self.exclude = exclude

    def display_status(self, profiler, other=None, print_mode='debug', short_print=True):
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
        assert isinstance(print_mode, str) and hasattr(self.logger, print_mode.lower()), \
            "'print_mode' needs to be a logger level"

        display_filter_func = self.make_display_filter_function(profiler, self.order, self.include_leftovers, self.exclude)
        status_dict = profiler.status_to_dict(to_value=False)
        layerwise_summary_1 = status_dict.pop('layerwise_summary', None)
        status_dict = display_filter_func(status_dict)
        status_dict['backend'] = profiler.backend
        status_dict['name'] = profiler.name

        if other is not None:
            other_status_dict = other.status_to_dict(to_value=False)
            layerwise_summary_2 = other_status_dict.pop('layerwise_summary', None)
            other_status_dict = display_filter_func(other_status_dict)
            other_status_dict['backend'] = other.backend
            other_status_dict['name'] = other.name
            summary_str = self.make_two_models_summary_str(status_dict, other_status_dict, short_print=short_print)
        else:
            layerwise_summary_2 = None
            summary_str = self.make_one_model_summary_str(status_dict, short_print=short_print)

        if not short_print:
            if layerwise_summary_1:
                self.logger.debug(layerwise_summary_1.value)
            if layerwise_summary_2:
                self.logger.debug(layerwise_summary_2.value)
        getattr(self.logger, print_mode.lower())(summary_str)

    @abstractmethod
    def make_display_filter_function(self, profiler, order=None, include_leftovers=True, exclude=None):
        """
        Create the display function. The default order displayed on the table is like this:
                - Evaluation Metric
                - Sub evaluation Metric (if any)
                - Model Size
                - Computational Complexity
                - Number of Parameters
                - Memory Footprint
                - Execution Time
                - Inference Time
                - Any other custom Metric

        :param order: Order of status keys to display, None is for the order defined in above docstring, defaults to None
        :type order: Optional[Iterable[str]]
        :param include_leftovers: Append all other status keys unspecified by the order at the end, defaults to True
        :type include_leftovers: Bool
        :param exclude: Sets of status keys to not display, defaults to None
        :type exclude: Optional[Set[str]]
        :return: filter function `callable`
        """

    @abstractmethod
    def make_one_model_summary_str(self, status_dict, short_print=True, description_str='', summary_str='\n'):
        """
        Return multiline string to display data from status_dict
        """

    @abstractmethod
    def make_two_models_summary_str(self, status_dict, status_dict_2, short_print=True, description_str='',
                                    summary_str='\n'):
        """
        Return multiline string to display data from two status_dicts
        """


class DefaultDisplay(Display):
    def make_display_filter_function(self, profiler, order=None, include_leftovers=True, exclude=None):
        eval_pfr = profiler.get_eval_pfr()
        secondary_metrics = ()
        if eval_pfr:
            secondary_metrics += tuple([m.NAME for m in eval_pfr.function.secondary_metrics])
        def order_iter():
            if order is None:
                _order = ('eval_metric',) + secondary_metrics
                _order += ('model_size', 'flops', 'total_params', 'memory_footprint', 'execution_time', 'inference_time')
            else:
                _order = order
            for o in _order:
                yield o
        exclude = cast_tuple(exclude)

        def display_filter_function(status_dict):
            new_status_dict = OrderedDict()

            for k in order_iter():
                if k not in exclude and k in status_dict:
                    new_status_dict[k] = status_dict[k]

            if include_leftovers:
                for k in set(status_dict) - (set(new_status_dict) | set(exclude)):
                    new_status_dict[k] = status_dict[k]

            return new_status_dict

        return display_filter_function

    def make_one_model_summary_str(self, status_dict, short_print=True, description_str='', summary_str='\n'):
        line_length, col1_length, col2_length = 63, 40, 20
        summary_str += "+" + "-" * line_length + "+" + "\n"
        line_new = "|{:^{line_length}}|".format("Deeplite Profiler", line_length=line_length)
        summary_str += line_new + "\n"
        summary_str += "+" + "-" * (col1_length + 1) + "+" + "-" * (col2_length + 1) + "+" + "\n"
        line_new = "|{:>{col1_length}} | {:>{col2_length}}|".format("Param Name (" + status_dict['name'] + ")",
                                                                    "Value", col1_length=col1_length,
                                                                    col2_length=col2_length)
        summary_str += line_new + "\n"
        line_new = "|{:>{col1_length}} | {:>{col2_length}}|".format("Backend: " + status_dict['backend'], "",
                                                                    col1_length=col1_length,
                                                                    col2_length=col2_length)
        summary_str += line_new + "\n"
        summary_str += "+" + "-" * (col1_length + 1) + "+" + "-" * (col2_length + 1) + "+" + "\n"

        for status in status_dict.values():
            if not isinstance(status, Metric):
                continue
            text, value, units, _, desc = parse_metric(status)
            print_str = text + ' (' + units + ')'
            line_new = "|{0:>{col1_length}} | {1:>{col2_length}.4f}|".format(print_str, value,
                                                                            col1_length=col1_length,
                                                                            col2_length=col2_length)
            summary_str += line_new + "\n"
            description_str += '* ' + text + ': ' + desc + "\n"
        summary_str += "+" + "-" * (col1_length + 1) + "+" + "-" * (col2_length + 1) + "+" + "\n"

        if not short_print:
            # Creating footnote
            summary_str += "Note: " + "\n"
            summary_str += description_str
            summary_str += "+" + "-" * line_length + "+"

        return summary_str

    def make_two_models_summary_str(self, status_dict, status_dict_2, short_print=True, description_str='',
                                    summary_str='\n'):
        line_length, col0_length, col1_length, col2_length, col3_length = 122, 40, 25, 25, 25
        summary_str += "+" + "-" * line_length + "+" + "\n"
        line_new = "|{:^{line_length}}|".format("Deeplite Profiler", line_length=line_length)
        summary_str += line_new + "\n"
        summary_str += "+" + "-" * (col0_length + 1) + "+" + "-" * (col1_length + 1) + "+" + "-" * (
                col2_length + 1) + "+" + "-" * (col2_length + 1) + "+" + "\n"
        line_new = "|{:>{col0_length}} | {:>{col1_length}}| {:>{col2_length}}| {:>{col3_length}}|".format(
            "Param Name", "Enhancement", "Value (" + status_dict['name'] + ")",
                                        "Value (" + status_dict_2['name'] + ")", col0_length=col0_length,
            col1_length=col1_length,
            col2_length=col2_length, col3_length=col3_length)
        summary_str += line_new + "\n"
        line_new = "|{:>{col0_length}} | {:>{col1_length}}| {:>{col2_length}}| {:>{col3_length}}|".format(
            "", "", "Backend: " + status_dict['backend'], "Backend: " + status_dict_2['backend'],
            col0_length=col0_length, col1_length=col1_length, col2_length=col2_length, col3_length=col3_length)
        summary_str += line_new + "\n"
        summary_str += "+" + "-" * (col0_length + 1) + "+" + "-" * (col1_length + 1) + "+" + "-" * (
                col2_length + 1) + "+" + "-" * (col2_length + 1) + "+" + "\n"

        for sk, sv in status_dict.items():
            if sk in status_dict_2 and isinstance(sv, Metric):
                sv2 = status_dict_2[sk]
                text, value_1, units_1, comp, desc = parse_metric(sv)
                _, value_2, units_2, _, _ = parse_metric(sv2)

                if comp is None:
                    comp = Comparative.NONE

                if units_1 != "" and units_2 != "" and units_1 != units_2:
                    # TODO fix when units are not the same
                    val = "<Unsupported units>"
                else:
                    try:
                        # this should raise a ValueError if the str is not correct
                        if isinstance(comp, str):
                            comp = Comparative(comp)
                        rval = compare_status_values(comp, value_1, value_2)
                    except ZeroDivisionError:
                        val = "INF"
                    except ValueError:
                        val = "<Error Comparing>"
                    else:
                        if rval is None:
                            val = '---'
                        else:
                            format_str = "{:>.2f}"
                            if comp is not Comparative.DIFF:
                                format_str += "x"
                            val = format_str.format(rval)

                print_str = text + ' (' + units_1 + ')'
                line_new = "|{:>{col0_length}} | {:>{col1_length}}| {:>{col2_length}.4f}| {:>{col3_length}.4f}|".format(
                    print_str, val, value_1, value_2,
                    col0_length=col0_length, col1_length=col1_length, col2_length=col2_length,
                    col3_length=col3_length)
                summary_str += line_new + "\n"
                description_str += '* ' + text + ': ' + desc + "\n"
        summary_str += "+" + "-" * (col0_length + 1) + "+" + "-" * (col1_length + 1) + "+" + "-" * (
                col2_length + 1) + "+" + "-" * (col2_length + 1) + "+" + "\n"

        # Creating footnote
        if not short_print:
            summary_str += "Note: " + "\n"
            summary_str += description_str
            summary_str += "+" + "-" * line_length + "+"

        return summary_str
