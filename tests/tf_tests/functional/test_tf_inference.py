import pytest
from tests.tf_tests.functional import BaseFunctionalTest, TENSORFLOW_SUPPORTED, TENSORFLOW_AVAILABLE, MODEL, DATA

class TestTFInference(BaseFunctionalTest):
    def test_get_acc(self):
        from deeplite.tf_profiler.tf_inference import get_accuracy
        assert get_accuracy(MODEL, DATA['test']) < 100

    def test_get_topk(self):
        from deeplite.tf_profiler.tf_inference import get_topk
        assert len(get_topk(MODEL, DATA['test'])) == 2
        assert len(get_topk(MODEL, DATA['test'], topk=1)) == 1

    def test_get_missclass(self):
        from deeplite.tf_profiler.tf_inference import get_missclass
        assert get_missclass(MODEL, DATA['test']) > 0


