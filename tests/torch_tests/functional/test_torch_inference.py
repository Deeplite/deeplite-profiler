import pytest
from tests.torch_tests.functional import BaseFunctionalTest, TORCH_AVAILABLE, MODEL, DATA
from unittest import mock

class TestTorchInference(BaseFunctionalTest):
    def test_get_acc(self):
        from deeplite.torch_profiler.torch_inference import get_accuracy
        assert get_accuracy(MODEL, DATA['test']) < 100

    def test_get_topk(self):
        from deeplite.torch_profiler.torch_inference import get_topk
        assert len(get_topk(MODEL, DATA['test'])) == 2
        assert len(get_topk(MODEL, DATA['test'], topk=1)) == 1

    def test_get_missclass(self):
        from deeplite.torch_profiler.torch_inference import get_missclass
        assert get_missclass(MODEL, DATA['test']) > 0

    def test_eval_loss_fn(self):
        from deeplite.torch_profiler.torch_inference import EvalLossFunction
        def loss_fn(model, batch):
            x, y = batch
            return model(x).mean()
        loss_fn.to_device = mock.MagicMock()
        eval_loss_fn = EvalLossFunction(loss_fn)
        eval_loss_fn(MODEL, DATA['test'])

