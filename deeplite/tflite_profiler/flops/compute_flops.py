import tensorflow as tf
import tflite
import numpy as np
from deeplite.profiler.utils import Device

from deeplite.profiler.metrics import Flops
from deeplite.profiler import ProfilerFunction


class ComputeFlops(ProfilerFunction):
    def get_bounded_status_keys(self):
        return Flops()

    def __call__(self, model, data_splits, batch_size=1, device=Device.CPU, include_weights=True):
        temp_model = tflite.Model.GetRootAsModel(model, 0)

        with tf.device('gpu' if device == Device.GPU else 'cpu'):
            return self._compute_flops(temp_model)

    def _compute_flops(self, model):
        subgraph = model.Subgraphs(0)
        total_flops = 0.0

        for op_idx in range(subgraph.OperatorsLength()):
            op = subgraph.Operators(op_idx)
            op_code = model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()

            flops = 0.0

            if op_code == tflite.BuiltinOperator.CONV_2D:
                # Convolutional layer
                input_tensor = subgraph.Tensors(op.Inputs(0))
                weight_tensor = subgraph.Tensors(op.Inputs(1))
                bias_tensor = subgraph.Tensors(op.Inputs(2))
                output_tensor = subgraph.Tensors(op.Outputs(0))

                input_shape = np.array(input_tensor.ShapeAsNumpy())
                filter_shape = np.array(weight_tensor.ShapeAsNumpy())
                output_shape = np.array(output_tensor.ShapeAsNumpy())

                flops = 2 * output_shape[1] * output_shape[2] * filter_shape[0] * filter_shape[1] * filter_shape[2] * filter_shape[3]

            elif op_code == tflite.BuiltinOperator.DEPTHWISE_CONV_2D:
                # Depthwise convolutional layer
                input_tensor = subgraph.Tensors(op.Inputs(0))
                weight_tensor = subgraph.Tensors(op.Inputs(1))
                output_tensor = subgraph.Tensors(op.Outputs(0))

                input_shape = np.array(input_tensor.ShapeAsNumpy())
                filter_shape = np.array(weight_tensor.ShapeAsNumpy())
                output_shape = np.array(output_tensor.ShapeAsNumpy())

                flops = 2 * output_shape[1] * output_shape[2] * filter_shape[0] * filter_shape[1] * filter_shape[2] * filter_shape[3]

            elif op_code == tflite.BuiltinOperator.FULLY_CONNECTED:
                # Linear (fully connected) layer
                input_tensor = subgraph.Tensors(op.Inputs(0))
                weight_tensor = subgraph.Tensors(op.Inputs(1))
                bias_tensor = subgraph.Tensors(op.Inputs(2))
                output_tensor = subgraph.Tensors(op.Outputs(0))

                input_shape = np.array(input_tensor.ShapeAsNumpy())
                weight_shape = np.array(weight_tensor.ShapeAsNumpy())

                flops = 2 * input_shape[1] * weight_shape[0]

            elif op_code in [tflite.BuiltinOperator.ADD, tflite.BuiltinOperator.AVERAGE_POOL_2D,
                             tflite.BuiltinOperator.CONCATENATION, tflite.BuiltinOperator.DEQUANTIZE,
                             tflite.BuiltinOperator.MAX_POOL_2D, tflite.BuiltinOperator.MUL,
                             tflite.BuiltinOperator.RESHAPE, tflite.BuiltinOperator.SOFTMAX,
                             tflite.BuiltinOperator.SPLIT, tflite.BuiltinOperator.SUB,
                             tflite.BuiltinOperator.TANH]:
                flops = 0.0

            total_flops += flops

        return total_flops / 1e9
