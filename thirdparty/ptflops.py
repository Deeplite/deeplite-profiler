'''
Source: https://github.com/sovrasov/flops-counter.pytorch 

Copyright (C) 2019 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
'''

import sys, time
from collections import OrderedDict
from functools import partial

import numpy as np
import torch
import torch.nn as nn

layer_count = 0

class HookVariables:
    def __init__(self, mac=0, params=0, activations=0, model_size=0, memory_footprint=0, summary = OrderedDict()):
        self.mac = mac # number of mac
        self.params = params # number of params
        self.activations = activations # number of activations
        self.model_size = model_size # memory occupied by the parameters
        self.memory_footprint = memory_footprint # memory occupied by the activations
        self.summary = summary

    def reset_variables(self, num_params=0):
        self.mac = 0
        self.params = num_params 
        self.activations = 0
        self.model_size = 0
        self.memory_footprint = 0
        self.summary = OrderedDict()

    def display(self):
        print("Number of MAC: ", self.mac)
        print("Number of params: ", self.params)
        print("Number of activations: ", self.activations)
        print("Number of model_size: ", self.model_size)
        print("Number of memory_footprint: ", self.memory_footprint)

def summary_str_header_footer(model_name):
    header = "\n--------------------------------------------------------------------------------------------------------------------------------------------" + "\n"
    line_new = "{:>25} {:>20} {:>15} {:>20} {:>25} {:>14} {:>14}".format(
        "Layer (" + model_name + ")", "Weight Shape", "Bias Shape", "Output Shape", "ActivationSize (Bytes)", "# Params", "Time (ms)")
    header += line_new + "\n"
    header += "============================================================================================================================================" + "\n"
    
    footer = "--------------------------------------------------------------------------------------------------------------------------------------------"
    
    return header, footer


class __Switch:
    def __init__(self):
        self.state = False
corruption_warning_switch = __Switch()


def flops_to_string(flops, units='GMac', precision=2):
    if units is None:
        if flops // 10**9 > 0:
            return str(round(flops / 10.**9, precision)) + ' GMac'
        elif flops // 10**6 > 0:
            return str(round(flops / 10.**6, precision)) + ' MMac'
        elif flops // 10**3 > 0:
            return str(round(flops / 10.**3, precision)) + ' KMac'
        else:
            return str(flops) + ' Mac'
    else:
        if units == 'GMac':
            return str(round(flops / 10.**9, precision)) + ' ' + units
        elif units == 'MMac':
            return str(round(flops / 10.**6, precision)) + ' ' + units
        elif units == 'KMac':
            return str(round(flops / 10.**3, precision)) + ' ' + units
        else:
            return str(flops) + ' Mac'


def params_to_string(params_num, units=None, precision=2):
    if units is None:
        if params_num // 10 ** 6 > 0:
            return str(round(params_num / 10 ** 6, 2)) + ' M'
        elif params_num // 10 ** 3:
            return str(round(params_num / 10 ** 3, 2)) + ' k'
        else:
            return str(params_num)
    else:
        if units == 'M':
            return str(round(params_num / 10.**6, precision)) + ' ' + units
        elif units == 'K':
            return str(round(params_num / 10.**3, precision)) + ' ' + units
        else:
            return str(params_num)


def get_model_parameters_number(model):
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params_num


def add_flops_counting_methods(net_main_module):
    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    net_main_module.start_flops_count = start_flops_count.__get__(net_main_module)
    net_main_module.stop_flops_count = stop_flops_count.__get__(net_main_module)
    net_main_module.reset_flops_count = reset_flops_count.__get__(net_main_module)
    net_main_module.compute_average_flops_cost = compute_average_flops_cost.__get__(
                                                    net_main_module)

    net_main_module.reset_flops_count()

    return net_main_module


def compute_average_flops_cost(self, model_name, start_time):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Returns current mean flops consumption per image.

    """
    batches_count = self.__batch_counter__
    total_flops = 0
    total_params = 0
    total_activations = 0
    total_model_size = 0
    total_memory_footprint = 0
    prev_total_time = start_time
    summary_str = ""
    for module in self.all_modules:
        if is_supported_instance(module) or not isinstance(module, (nn.Sequential, nn.ModuleList)):
            if hasattr(module, '__hook_variables__') and module.__hook_variables__.summary:
                total_flops += module.__hook_variables__.mac
                total_params += module.__hook_variables__.params
                total_activations += module.__hook_variables__.activations
                total_model_size += module.__hook_variables__.model_size
                total_memory_footprint += module.__hook_variables__.memory_footprint

                layer_time = (module.__hook_variables__.summary["layer_time"] - prev_total_time) * 1000

                line_new = "{:>25} {:>20} {:>15} {:>20} {:>25} {:>14} {:>14}".format(
                    module.__hook_variables__.summary["m_key"],
                    str(module.__hook_variables__.summary.get("layer_weight_size", None)),
                    str(module.__hook_variables__.summary.get("layer_bias_size", None)),
                    str(module.__hook_variables__.summary["output_shape"]),
                    "{0:,}".format(module.__hook_variables__.memory_footprint),
                    "{0:,}".format(module.__hook_variables__.params),
                    "{0:2.4f}".format(layer_time),
                )
                summary_str += line_new + "\n"
                prev_total_time = module.__hook_variables__.summary["layer_time"]

    header, footer = summary_str_header_footer(model_name)
    summary_str = header + summary_str + footer
    self.all_modules = []
    global layer_count
    layer_count = 0 

    return total_flops / batches_count, total_params, total_model_size, total_memory_footprint / batches_count, summary_str


def start_flops_count(self, **kwargs):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Activates the computation of mean flops consumption per image.
    Call it before you run the network.

    """
    add_batch_counter_hook_function(self)
    self.all_modules = []
    corruption_warning_switch.state = False

    def add_flops_counter_hook_function(module, ost, verbose, ignore_list):
        if type(module) in ignore_list:
            if is_supported_instance(module):
                module.__hook_variables__ = HookVariables()
        if hasattr(module, 'compute_module_complexity'):
            handle = module.register_forward_hook(generic_flops_counter_hook)
            module.__flops_handle__ = handle
            self.all_modules.append(module)
        elif is_supported_instance(module):
            if hasattr(module, '__flops_handle__'):
                return
            elif type(module) in CUSTOM_MODULES_MAPPING:
                handle = module.register_forward_hook(
                                        CUSTOM_MODULES_MAPPING[type(module)])
            else:
                handle = module.register_forward_hook(MODULES_MAPPING[type(module)])
            module.__flops_handle__ = handle
            self.all_modules.append(module)
        else:
            #if not type(module) in (nn.Sequential, nn.ModuleList):
            if not isinstance(module, (nn.Sequential, nn.ModuleList)):
                if hasattr(module, '__flops_handle__'):
                    return
                handle = module.register_forward_hook(no_flops_ops_counter_hook)
                module.__flops_handle__ = handle
                self.all_modules.append(module)

    self.apply(partial(add_flops_counter_hook_function, **kwargs))


def stop_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Stops computing the mean flops consumption per image.
    Call whenever you want to pause the computation.

    """
    remove_batch_counter_hook_function(self)
    self.apply(remove_flops_counter_hook_function)


def reset_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Resets statistics computed so far.

    """
    add_batch_counter_variables_or_reset(self)
    self.apply(add_flops_counter_variable_or_reset)


# ---- Internal functions
def batch_counter_hook(module, input, output):
    batch_size = 1
    if len(input) > 0:
        # Can have multiple inputs, getting the first one
        input = input[0]
        batch_size = len(input)
    else:
        pass
        print('Warning! No positional inputs found for a module,'
              ' assuming batch size is 1.')
    module.__batch_counter__ += batch_size

def get_m_key(name):
    class_name = str(name).split(".")[-1].split("'")[0]
    m_key = "%s-%i" % (class_name, layer_count + 1)
    return m_key

def get_input_shape(input):
    # note: this is done really just for the summary so it is safe to return dummy values
    if isinstance(input, (list, tuple)) and len(input) == 1:
        return list(input[0].size())
    if isinstance(input, torch.Tensor):
        return list(input.size())
    # print("unknown input tuple detected in hook for model size, size could be wrong")
    return [0]

def _parse_output(o, activation_size):
    output_shape = []
    total_activations = []
    memory_footprint = []
    corrupted = [False]

    def get_memory(oo):
        if isinstance(oo, (list, tuple)):
            for o_ in oo:
                get_memory(o_)
        elif isinstance(oo, dict):
            for o_ in oo.values():
                get_memory(o_)
        elif isinstance(oo, torch.Tensor):
            output_shape.append(list(oo.size()))
            total_activations.append(np.prod(oo.size()))
            memory_footprint.append(np.prod(oo.size()) * activation_size)
        else:
            corrupted[0] = True

    get_memory(o)
    return total_activations, memory_footprint, output_shape, corrupted[0]

def parse_module_output(module, output, activation_size):
    total_activations, memory_footprint, output_shape, corrupted = _parse_output(output, activation_size)
    if corrupted and corruption_warning_switch.state is False:
        print("Warning!! cannot parse module '{}' output types, memory footprint value is potentially".format(module) +\
              " underestimated and output shapes corrupted")
        corruption_warning_switch.state = True
    return total_activations, memory_footprint, output_shape


def upsample_flops_counter_hook(module, input, output):
    activation_size = 4
    output_size = output[0]
    batch_size = output.shape[0]
    output_elements_count = batch_size
    for val in output_size.shape[1:]:
        output_elements_count *= val

    total_activations, memory_footprint, output_shape = parse_module_output(module, output, activation_size)

    module.__hook_variables__ = HookVariables() # write fi else
    module.__hook_variables__.mac += int(output_elements_count)
    module.__hook_variables__.activations += sum(total_activations)
    module.__hook_variables__.memory_footprint += sum(memory_footprint)
    
    m_key = get_m_key(module.__class__)
    summary = OrderedDict()
    summary["m_key"] = m_key
    summary['layer_time'] = time.time()
    summary["input_shape"] = get_input_shape(input)
    summary["input_shape"][0] = batch_size
    summary["output_shape"] = output_shape
    if hasattr(module, "weight") and hasattr(module.weight, "size"):
        summary["layer_weight_size"] = list(module.weight.size())
    module.__hook_variables__.summary = summary
    global layer_count
    layer_count += 1
    
def no_flops_ops_counter_hook(module, input, output):
    activation_size = 4
    #batch_size = output.shape[0]
    #active_elements_count = output.numel()

    total_activations, memory_footprint, output_shape = parse_module_output(module, output, activation_size)

    module.__hook_variables__ = HookVariables()
    module.__hook_variables__.activations += sum(total_activations)
    module.__hook_variables__.memory_footprint += sum(memory_footprint)

    m_key = get_m_key(module.__class__)
    summary = OrderedDict()
    summary["m_key"] = m_key
    summary['layer_time'] = time.time()
    summary["input_shape"] = get_input_shape(input)
    #summary["input_shape"][0] = batch_size
    summary["output_shape"] = output_shape
    module.__hook_variables__.summary = summary
    global layer_count
    layer_count += 1

def relu_flops_counter_hook(module, input, output):
    activation_size = 4
    batch_size = output.shape[0]
    active_elements_count = output.numel()

    total_activations, memory_footprint, output_shape = parse_module_output(module, output, activation_size)

    module.__hook_variables__ = HookVariables()
    module.__hook_variables__.mac += int(active_elements_count)
    module.__hook_variables__.activations += sum(total_activations)
    module.__hook_variables__.memory_footprint += sum(memory_footprint)

    m_key = get_m_key(module.__class__)
    summary = OrderedDict()
    summary["m_key"] = m_key
    summary['layer_time'] = time.time()
    summary["input_shape"] = get_input_shape(input)
    summary["input_shape"][0] = batch_size
    summary["output_shape"] = output_shape
    if hasattr(module, "weight") and hasattr(module.weight, "size"):
        summary["layer_weight_size"] = list(module.weight.size())
    if hasattr(module, "bias") and hasattr(module.bias, "size"):
        summary["layer_bias_size"] = list(module.bias.size())
    module.__hook_variables__.summary = summary
    global layer_count
    layer_count += 1

def linear_flops_counter_hook(module, input, output):
    param_size = 4
    activation_size = 4
    input = input[0]
    output_last_dim = output.shape[-1]
    batch_size = output.shape[0]
    bias_flops = output_last_dim if module.bias is not None else 0

    total_activations, memory_footprint, output_shape = parse_module_output(module, output, activation_size)
    
    module.__hook_variables__ = HookVariables()
    module.__hook_variables__.mac += int(np.prod(input.shape) * output_last_dim + bias_flops)
    module.__hook_variables__.params += sum(p.numel() for p in module.parameters() if p.requires_grad)
    module.__hook_variables__.activations += sum(total_activations)
    module.__hook_variables__.model_size += module.__hook_variables__.params * param_size
    module.__hook_variables__.memory_footprint += sum(memory_footprint)

    m_key = get_m_key(module.__class__)
    summary = OrderedDict()
    summary["m_key"] = m_key
    summary['layer_time'] = time.time()
    summary["input_shape"] = get_input_shape(input)
    summary["input_shape"][0] = batch_size
    summary["output_shape"] = output_shape
    if hasattr(module, "weight") and hasattr(module.weight, "size"):
        summary["layer_weight_size"] = list(module.weight.size())
    if hasattr(module, "bias") and hasattr(module.bias, "size"):
        summary["layer_bias_size"] = list(module.bias.size())
    module.__hook_variables__.summary = summary
    global layer_count
    layer_count += 1

def pool_flops_counter_hook(module, input, output):
    activation_size = 4
    batch_size = output.shape[0]
    input = input[0]
    total_activations, memory_footprint, output_shape = parse_module_output(module, output, activation_size)

    module.__hook_variables__ = HookVariables()
    module.__hook_variables__.mac += int(np.prod(input.shape))
    module.__hook_variables__.activations += sum(total_activations)
    module.__hook_variables__.memory_footprint += sum(memory_footprint)

    m_key = get_m_key(module.__class__)
    summary = OrderedDict()
    summary["m_key"] = m_key
    summary['layer_time'] = time.time()
    summary["input_shape"] = get_input_shape(input)
    summary["input_shape"][0] = batch_size
    summary["output_shape"] = output_shape
    if hasattr(module, "weight") and hasattr(module.weight, "size"):
        summary["layer_weight_size"] = list(module.weight.size())
    if hasattr(module, "bias") and hasattr(module.bias, "size"):
        summary["layer_bias_size"] = list(module.bias.size())
    module.__hook_variables__.summary = summary
    global layer_count
    layer_count += 1

def bn_flops_counter_hook(module, input, output):
    activation_size = 4
    param_size = 4
    batch_size = output.shape[0]
    input = input[0]

    batch_flops = np.prod(input.shape)
    if module.affine:
        batch_flops *= 2

    total_activations, memory_footprint, output_shape = parse_module_output(module, output, activation_size)

    module.__hook_variables__ = HookVariables()
    module.__hook_variables__.mac += int(batch_flops)
    module.__hook_variables__.params += int(2 * module.num_features)
    module.__hook_variables__.activations += sum(total_activations)
    module.__hook_variables__.model_size += module.__hook_variables__.params * param_size
    module.__hook_variables__.memory_footprint += sum(memory_footprint)

    m_key = get_m_key(module.__class__)
    summary = OrderedDict()
    summary["m_key"] = m_key
    summary['layer_time'] = time.time()
    summary["input_shape"] = get_input_shape(input)
    summary["input_shape"][0] = batch_size
    summary["output_shape"] = output_shape
    if hasattr(module, "weight") and hasattr(module.weight, "size"):
        summary["layer_weight_size"] = list(module.weight.size())
    if hasattr(module, "bias") and hasattr(module.bias, "size"):
        summary["layer_bias_size"] = list(module.bias.size())
    module.__hook_variables__.summary = summary
    global layer_count
    layer_count += 1

def conv_flops_counter_hook(module, input, output):
    # Can have multiple inputs, getting the first one
    param_size = 4
    activation_size = 4
    batch_size = output.shape[0]
    input = input[0]

    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])
    kernel_dims = list(module.kernel_size)
    in_channels = module.in_channels
    out_channels = module.out_channels
    groups = module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = int(np.prod(kernel_dims)) * \
        in_channels * filters_per_channel

    active_elements_count = batch_size * int(np.prod(output_dims))
    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0
    if module.bias is not None:
        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops
    num_out_elements = output.numel()

    total_activations, memory_footprint, output_shape = parse_module_output(module, output, activation_size)

    module.__hook_variables__ = HookVariables()
    module.__hook_variables__.mac += int(overall_flops) 
    module.__hook_variables__.params += sum(p.numel() for p in module.parameters() if p.requires_grad)
    module.__hook_variables__.activations += sum(total_activations)
    module.__hook_variables__.model_size += module.__hook_variables__.params * param_size
    module.__hook_variables__.memory_footprint += sum(memory_footprint)

    m_key = get_m_key(module.__class__)
    summary = OrderedDict()
    summary["m_key"] = m_key
    summary['layer_time'] = time.time()
    summary["input_shape"] = get_input_shape(input)
    summary["input_shape"][0] = batch_size
    summary["output_shape"] = output_shape
    if hasattr(module, "weight") and hasattr(module.weight, "size"):
        summary["layer_weight_size"] = list(module.weight.size())
    if hasattr(module, "bias") and hasattr(module.bias, "size"):
        summary["layer_bias_size"] = list(module.bias.size())
    module.__hook_variables__.summary = summary
    global layer_count
    layer_count += 1

def generic_flops_counter_hook(module, input, output):
    try:
        rval = module.compute_module_complexity(input, output)
    except AttributeError:
        raise TypeError("Wrong flops hook, module '{}' does not have 'compute_module_complexity' method".format(module))

    try:
        flops = rval['flops']
        param_size = rval.get('param_size', 4)
        activation_size = rval.get('activation_size', 4)
        params = rval.get('params', None)
        assert flops is not None
    except (AssertionError, TypeError, KeyError):
        raise RuntimeError("Module '{}'.compute_module_complexity should return a dict with keys ".format(module) + \
                           "'flops', 'param_size', 'activation_size' and 'params'. 'flops' should not be None.")

    total_activations, memory_footprint, output_shape = parse_module_output(module, output, activation_size)

    module.__hook_variables__ = HookVariables()
    module.__hook_variables__.mac += int(flops)
    module.__hook_variables__.params += sum(p.numel() for p in module.parameters() if p.requires_grad) if params is None else params
    module.__hook_variables__.activations += sum(total_activations)
    module.__hook_variables__.model_size += module.__hook_variables__.params * param_size
    module.__hook_variables__.memory_footprint += sum(memory_footprint)

    m_key = get_m_key(module.__class__)
    summary = OrderedDict()
    summary["m_key"] = m_key
    summary['layer_time'] = time.time()
    summary["input_shape"] = get_input_shape(input)
    # summary["input_shape"][0] = batch_size
    summary["output_shape"] = output_shape
    if hasattr(module, "weight") and hasattr(module.weight, "size"):
        summary["layer_weight_size"] = list(module.weight.size())
    if hasattr(module, "bias") and hasattr(module.bias, "size"):
        summary["layer_bias_size"] = list(module.bias.size())
    module.__hook_variables__.summary = summary
    global layer_count
    layer_count += 1


def add_batch_counter_variables_or_reset(module):
    module.__batch_counter__ = 0

def add_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        return

    handle = module.register_forward_hook(batch_counter_hook)
    module.__batch_counter_handle__ = handle


def remove_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        module.__batch_counter_handle__.remove()
        del module.__batch_counter_handle__


def add_flops_counter_variable_or_reset(module):
    if is_supported_instance(module):
        if hasattr(module, '__hook_variables__'):
            print('Warning: variables __flops__ or __params__ are already '
                  'defined for the module' + type(module).__name__ +
                  ' ptflops can affect your code!')
        #module.__hook_variables__.reset(num_params=get_model_parameters_number(module))
        module.__hook_variables__ = HookVariables(params=get_model_parameters_number(module))

CUSTOM_MODULES_MAPPING = {
    # custom layers
}

MODULES_MAPPING = {
    # convolutions
    nn.Conv1d: conv_flops_counter_hook,
    nn.Conv2d: conv_flops_counter_hook,
    nn.Conv3d: conv_flops_counter_hook,
    # activations
    nn.ReLU: relu_flops_counter_hook,
    nn.PReLU: relu_flops_counter_hook,
    nn.ELU: relu_flops_counter_hook,
    nn.LeakyReLU: relu_flops_counter_hook,
    nn.ReLU6: relu_flops_counter_hook,
    # poolings
    nn.MaxPool1d: pool_flops_counter_hook,
    nn.AvgPool1d: pool_flops_counter_hook,
    nn.AvgPool2d: pool_flops_counter_hook,
    nn.MaxPool2d: pool_flops_counter_hook,
    nn.MaxPool3d: pool_flops_counter_hook,
    nn.AvgPool3d: pool_flops_counter_hook,
    nn.AdaptiveMaxPool1d: pool_flops_counter_hook,
    nn.AdaptiveAvgPool1d: pool_flops_counter_hook,
    nn.AdaptiveMaxPool2d: pool_flops_counter_hook,
    nn.AdaptiveAvgPool2d: pool_flops_counter_hook,
    nn.AdaptiveMaxPool3d: pool_flops_counter_hook,
    nn.AdaptiveAvgPool3d: pool_flops_counter_hook,
    # BNs
    nn.BatchNorm1d: bn_flops_counter_hook,
    nn.BatchNorm2d: bn_flops_counter_hook,
    nn.BatchNorm3d: bn_flops_counter_hook,
    # FC
    nn.Linear: linear_flops_counter_hook,
    # Upscale
    nn.Upsample: upsample_flops_counter_hook,
    # Deconvolution
    nn.ConvTranspose1d: conv_flops_counter_hook,
    nn.ConvTranspose2d: conv_flops_counter_hook,
    nn.ConvTranspose3d: conv_flops_counter_hook,
}


def is_supported_instance(module):
    if type(module) in MODULES_MAPPING or type(module) in CUSTOM_MODULES_MAPPING:
        return True
    return False


def remove_flops_counter_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops_handle__'):
            module.__flops_handle__.remove()
            del module.__flops_handle__