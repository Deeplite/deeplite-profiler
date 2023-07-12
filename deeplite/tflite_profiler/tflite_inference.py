import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
try:
    tf.compat.v1.enable_eager_execution()
except:
    pass

from deeplite.profiler.evaluate import EvaluationFunction
from deeplite.profiler.utils import Device


def top_k_accuracy(results, gt_labels, k, dtype="uint8"):
    accuracy = (
        np.sum(np.argsort(results, axis=1)[:, -k:] == gt_labels.reshape(-1, 1)) /
        gt_labels.size)
    return accuracy * 100.0


class _GetAccuracy(EvaluationFunction):
    def __call__(self, model, data_loader, device=Device.CPU, transform=None, k=1):

          """Evaluates tensorflow lite classification model with the given dataset."""
          interpreter = tf.lite.Interpreter(model_content=model)
          interpreter.allocate_tensors()
          input_idx = interpreter.get_input_details()[0]['index']
          output_idx = interpreter.get_output_details()[0]['index']
          input_details = interpreter.get_input_details()[0]
          dtype = input_details['dtype']
          results = []
          gt_labels = []
          for x, y in data_loader:
            if (len(x.shape) == 3):
                x = x[None]
            if transform:
                x, y = transform(x, y)
            if dtype in (np.int8, np.uint8):
                input = interpreter.get_input_details()[0]
                scale, zero_point = input['quantization']
                im = (x / scale + zero_point)
                im = tf.cast(im, tf.uint8)
                interpreter.set_tensor(input_idx, im)
            else:
                interpreter.set_tensor(input_idx, x)
            interpreter.invoke()
            results.append(interpreter.get_tensor(output_idx).flatten())
            gt_labels.append(np.argmax(y))

          results = np.array(results)
          gt_labels = np.array(gt_labels)
          return top_k_accuracy(results, gt_labels, k, dtype)
get_accuracy = _GetAccuracy()

