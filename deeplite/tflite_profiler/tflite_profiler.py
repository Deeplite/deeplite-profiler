import tensorflow as tf
import tflite

from deeplite.profiler import Profiler, ProfilerFunction
from deeplite.profiler.utils import Device
from deeplite.profiler.metrics import Flops
from deeplite.tf_profiler.tf_data_loader import TFDataLoader, TFForwardPass


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
try:
    tf.compat.v1.enable_eager_execution()
except:
    pass


class TFLiteProfiler(Profiler):
    def __init__(self, model, data_splits, **kwargs):
        super().__init__(model, data_splits, **kwargs)
        self.backend = 'TFLiteBackend'

    @classmethod
    def dl_cls(cls):
        return TFDataLoader

    @classmethod
    def fp_cls(cls):
        return TFForwardPass


class ComputeFlops(ProfilerFunction):
    def get_bounded_status_keys(self):
        return Flops()

    def __call__(self, model, data_splits, batch_size=1, device=Device.CPU, include_weights=True):
        temp_model = tflite.Model.GetRootAsModel(model, 0)

        with tf.device('gpu' if device == Device.GPU else 'cpu'):
            return self._compute_flops(temp_model)

    def _compute_flops(self, model):
            graph = model.Subgraphs(0)

            total_flops = 0.0
            for i in range(graph.OperatorsLength()):
                op = graph.Operators(i)
                op_code = model.OperatorCodes(op.OpcodeIndex())
                op_code_builtin = op_code.BuiltinCode()

                op_opt = op.BuiltinOptions()

                flops = 0.0
                if op_code_builtin == tflite.BuiltinOperator.CONV_2D:
                    # input shapes: in, weight, bias
                    in_shape = graph.Tensors( op.Inputs(0) ).ShapeAsNumpy()
                    filter_shape = graph.Tensors( op.Inputs(1) ).ShapeAsNumpy()
                    bias_shape = graph.Tensors( op.Inputs(2) ).ShapeAsNumpy()
                    # output shape
                    out_shape = graph.Tensors( op.Outputs(0) ).ShapeAsNumpy()
                    # ops options
                    opt = tflite.Conv2DOptions()
                    opt.Init(op_opt.Bytes, op_opt.Pos)
                    # opt.StrideH()

                    # flops. 2x means mul(1)+add(1). 2x not needed if you calculate MACCs
                    # refer to https://github.com/AlexeyAB/darknet/src/convolutional_layer.c `l.blopfs =`
                    flops = 2 * out_shape[1] * out_shape[2] * filter_shape[0] * filter_shape[1] * filter_shape[2] * filter_shape[3]

                elif op_code_builtin == tflite.BuiltinOperator.DEPTHWISE_CONV_2D:
                    in_shape = graph.Tensors( op.Inputs(0) ).ShapeAsNumpy()
                    filter_shape = graph.Tensors( op.Inputs(1) ).ShapeAsNumpy()
                    out_shape = graph.Tensors( op.Outputs(0) ).ShapeAsNumpy()
                    # flops
                    flops = 2 * out_shape[1] * out_shape[2] * filter_shape[0] * filter_shape[1] * filter_shape[2] * filter_shape[3]

                total_flops += flops

            return total_flops / 1e9 
