import time

import numpy as np
import tensorflow as tf

from deeplite.profiler import Profiler, ProfilerFunction
from deeplite.profiler.metrics import *
from deeplite.profiler.utils import Device
from deeplite.tf_profiler.tf_data_loader import TFDataLoader, TFForwardPass


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
try:
    tf.compat.v1.enable_eager_execution()
except:
    pass


def get_temp_model(model):
    weights = model.get_weights()
    new_model = tf.keras.models.clone_model(model)
    new_model.build(model.input_shape)
    new_model.set_weights(weights)
    return new_model


class TFProfiler(Profiler):
    def __init__(self, model, data_splits, **kwargs):
        super().__init__(model, data_splits, **kwargs)
        self.backend = 'TFBackend'

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
        temp_model = get_temp_model(model)
        dataloader = data_splits['train']

        with tf.device('gpu' if device == Device.GPU else 'cpu'):
            return self._compute_flops(temp_model, dataloader, batch_size=batch_size, device=device,
                                       include_weights=include_weights)

    # HAS TO RETURN A TUPLE IN THE SAME ORDER OF STATUSKEYS
    def _compute_flops(self, model, dataloader, batch_size=1, device=Device.CPU, include_weights=True):
        graph = tf.Graph()
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        session = tf.Session(graph=graph)  # , config=tf.ConfigProto(gpu_options=gpu_options))

        with graph.as_default():
            with session.as_default():
                temp_model = tf.keras.models.clone_model(model)
                loss = tf.keras.losses.MeanSquaredError()
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
                temp_model.compile(optimizer=optimizer, loss=loss)

                dataloader.sample_random_forward(temp_model, batch_size=1, device=Device.CPU)
                opts = (tf.compat.v1.profiler.ProfileOptionBuilder(
                    tf.profiler.ProfileOptionBuilder.float_operation())
                        .with_empty_output()
                        .build())
                flops = tf.profiler.profile(graph=graph, run_meta=tf.RunMetadata(), cmd='op', options=opts)
                session.close()

        del session
        flops = getattr(flops, 'total_float_ops',
                        0) / 2e9  # Giga Flops - Counting only the flops of forward pass

        return flops


class ComputeSize(ProfilerFunction):
    @classmethod
    def _get_bounded_status_keys_cls(cls):
        return ModelSize, MemoryFootprint

    def get_bounded_status_keys(self):
        sk_cls = self._get_bounded_status_keys_cls()
        rval = tuple(cls() for cls in sk_cls)
        return rval

    def __call__(self, model, data_splits, batch_size=1, device=Device.CPU, include_weights=True):
        sk_cls = self._get_bounded_status_keys_cls()
        temp_model = get_temp_model(model)
        dataloader = data_splits['train']

        with tf.device('gpu' if device == Device.GPU else 'cpu'):
            rval = self._compute_size(temp_model, dataloader, batch_size=batch_size, device=device,
                                      include_weights=include_weights)

        assert len(sk_cls) == len(rval)
        return {x.NAME: y for x, y in zip(sk_cls, rval)}

    # HAS TO RETURN A TUPLE IN THE SAME ORDER OF STATUSKEYS
    def _compute_size(self, model, dataloader, batch_size=1, device=Device.CPU, include_weights=True):
        model_vars = model.trainable_variables
        _, model_size = tf.contrib.slim.model_analyzer.analyze_vars(model_vars, print_info=False)

        activation_size = 0
        for layer in model.layers:
            output_shape = layer.output_shape
            if isinstance(output_shape, list):
                for osp in output_shape:
                    osp = [x for x in osp if x is not None]
                    activation_size += np.product(osp) * batch_size * 4  # 4 bytes
            if isinstance(output_shape, tuple):
                output_shape = [x for x in output_shape if x is not None]
                activation_size += np.product(output_shape) * batch_size * 4  # 4 bytes

        total_input_size = 0
        input_shape = model.layers[0].input_shape
        if isinstance(input_shape, list):
            for isp in input_shape:
                isp = [x for x in isp if x is not None]
                total_input_size += np.product(isp) * batch_size * 4  # 4 bytes
        if isinstance(input_shape, tuple):
            input_shape = [x for x in input_shape if x is not None]
            total_input_size += np.product(input_shape) * batch_size * 4  # 4 bytes

        memory_footprint = int(activation_size + total_input_size)
        if include_weights:
            memory_footprint += model_size
        model_size = abs(model_size / (1024 ** 2.))  # Convert bytes to MB
        memory_footprint = abs(memory_footprint / (1024 ** 2.))  # Convert bytes to MB

        return model_size, memory_footprint


class ComputeParams(ProfilerFunction):
    def get_bounded_status_keys(self):
        return TotalParams()

    def __call__(self, model, data_splits, batch_size=1, device=Device.CPU, include_weights=True):
        temp_model = get_temp_model(model)
        dataloader = data_splits['train']

        with tf.device('gpu' if device == Device.GPU else 'cpu'):
            return self._compute_params(temp_model, dataloader, batch_size=batch_size, device=device,
                                        include_weights=include_weights)

    # HAS TO RETURN A TUPLE IN THE SAME ORDER OF STATUSKEYS
    def _compute_params(self, model, dataloader, batch_size=1, device=Device.CPU, include_weights=True):
        model_vars = model.trainable_variables
        num_params, _ = tf.contrib.slim.model_analyzer.analyze_vars(model_vars, print_info=False)

        params = num_params / 1e6  # Million Flops
        return params


class ComputeLayerwiseSummary(ProfilerFunction):
    def get_bounded_status_keys(self):
        return LayerwiseSummary()

    def __call__(self, model, data_splits, batch_size=1, device=Device.CPU, include_weights=True):
        temp_model = get_temp_model(model)
        dataloader = data_splits['train']

        with tf.device('gpu' if device == Device.GPU else 'cpu'):
            return self._compute_layerwise_summary(temp_model, dataloader, batch_size=batch_size,
                                                   device=device,
                                                   include_weights=include_weights)

    # HAS TO RETURN A TUPLE IN THE SAME ORDER OF STATUSKEYS
    def _compute_layerwise_summary(self, model, dataloader, batch_size=1, device=Device.CPU,
                                   include_weights=True):
        stringlist = []
        model.summary(print_fn=stringlist.append)
        stringlist = stringlist[1:-4]
        summary_str = "\n".join(stringlist)

        return summary_str


class ComputeExecutionTime(ProfilerFunction):
    def get_bounded_status_keys(self):
        return ExecutionTime()

    def __call__(self, model, data_splits, split='train', batch_size=1, device=Device.CPU):
        dataloader = data_splits[split]
        temp_model = get_temp_model(model)
        device = 'gpu' if device == Device.GPU else 'cpu'
        with tf.device(device):
            return self._compute_exectime(temp_model, dataloader, batch_size=batch_size)

    def _compute_exectime(self, model, dataloader, batch_size=1):
        tnum = dataloader.forward_pass.create_random_model_inputs(batch_size)

        # START BENCHMARKING
        steps = 10
        fp_time = 0.

        # DRY RUNS
        for i in range(steps):
            _ = model(tnum, training=False)

        class timecallback(tf.keras.callbacks.Callback):
            def __init__(self):
                self.batch_times = 0
                self.step_time_start_batch = 0

            def on_predict_batch_begin(self, batch, logs=None):
                self.step_time_start_batch = time.perf_counter()

            def on_predict_batch_end(self, batch, logs=None):
                self.batch_times = time.perf_counter() - self.step_time_start_batch

        tt = time.perf_counter()
        ctlTime = time.perf_counter() - tt
        tcb = timecallback()
        for i in range(steps):
            _ = model.predict(tnum, batch_size=batch_size, callbacks=[tcb])
            if i > 0:
                fp_time += (tcb.batch_times - ctlTime)
        fp_time = fp_time / (steps - 1) / batch_size
        execution_time = fp_time * 1000
        return execution_time
