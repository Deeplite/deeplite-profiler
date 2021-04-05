import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
try:
    tf.compat.v1.enable_eager_execution()
except:
    pass

import numpy as np

from deeplite.profiler.evaluate import EvaluationFunction
from deeplite.profiler.utils import Device, cast_tuple


def set_device(device):
    if device == Device.GPU:
        device_name = "/gpu:0"
    else:
        device_name = "/cpu:0"
    return device_name


class _GetMissclass(EvaluationFunction):
    _NAME = 'missclass'

    def __call__(self, model, data_loader, device=Device.CPU, transform=None):
        device_name = set_device(device)
        test_accuracy = tf.keras.metrics.CategoricalAccuracy()

        with tf.device(device_name):
            for x, y in data_loader:
                if (len(x.shape) == 3):
                    x = x[None]
                if transform:
                    x, y = transform(x, y)

                logits = model(x, training=False)
                if type(logits) == tuple: logits = logits[0]
                test_accuracy(logits, y)

            return 100 - test_accuracy.result().numpy() * 100
get_missclass = _GetMissclass()


class _GetAccuracy(_GetMissclass):
    def __call__(self, model, data_loader, device=Device.CPU, transform=None):
        rval = super().__call__(model, data_loader, device=device, transform=transform)
        return 100. - rval
get_accuracy = _GetAccuracy()


class _GetTopk(EvaluationFunction):
    def __call__(self, model, data_loader, device=Device.CPU, topk=(1, 5), transform=None):
        topk = cast_tuple(topk)

        def _accuracy(output, target):
            """Computes the precision@k for the specified values of k"""
            maxk = max(topk)
            batch_size = target.shape[0]

            _, pred = tf.math.top_k(output, maxk, True)
            pred = tf.transpose(pred)
            pred = tf.cast(pred, tf.float32)

            target = tf.reshape(target, [1, -1])
            target = tf.cast(target, tf.float32)

            correct = tf.math.equal(pred, target)
            res = []

            for k in topk:
                correct_k = tf.math.reduce_sum(tf.cast(tf.reshape(correct[:k], [-1]), tf.float32), axis=0,
                                               keepdims=True)
                res.append(list(correct_k))
            return res

        correct_k = [0.] * len(topk)
        ntotal = 0
        device_name = set_device(device)

        with tf.device(device_name):
            for x, y in data_loader:
                if (len(x.shape) == 3):
                    x = x[None]
                if transform:
                    if isinstance(x, tf.Tensor): x = x.numpy()
                    if isinstance(y, tf.Tensor): y = y.numpy()
                    x, y = transform(x, y)

                if isinstance(x, tf.Tensor): x = x.numpy()
                outputs = model(x, training=False)
                correct_k = np.add(correct_k, _accuracy(outputs, y))
                ntotal += y.shape[0]

        rval = (100. * (correct_k / ntotal)).tolist()
        return {'top-' + str(k): res for k, res in zip(topk, rval)}
get_topk = _GetTopk()
