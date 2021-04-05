import pytest
from deeplite.profiler.evaluate import EvaluationFunction


def test_dict_eval_func():
    class DictReturnEval(EvaluationFunction):
        def __call__(self, model, loader, **kwargs):
            return EvaluationFunction.filter_call_rval({'a':1, 'b':2}, **kwargs)
    f = DictReturnEval()
    with pytest.raises(ValueError):
        rval = f(None, None, return_dict=False, return_keys=None)

    with pytest.raises(ValueError):
        rval = f(None, None, return_dict=False, return_keys=['a', 'b'])

    with pytest.raises(KeyError):
        rval = f(None, None, return_dict=False, return_keys='c')

    with pytest.raises(KeyError):
        rval = f(None, None, return_dict=True, return_keys='c')

    rval = f(None, None, return_dict=True, return_keys=None)
    assert rval == {'a': 1, 'b': 2}

    rval = f(None, None, return_dict=True, return_keys='a')
    assert rval == {'a': 1}

    rval = f(None, None, return_dict=False, return_keys='b')
    assert rval == 2

    class DictReturnEval(EvaluationFunction):
        def __call__(self, model, loader, **kwargs):
            return EvaluationFunction.filter_call_rval({'c':3}, **kwargs)
    g = DictReturnEval()
    assert g(None, None, return_dict=False, return_keys=None) == 3


def test_non_dict_eval_func():
    class NonDictReturnEval(EvaluationFunction):
        def __call__(self, model, loader, **kwargs):
            return EvaluationFunction.filter_call_rval(1, **kwargs)
    f = NonDictReturnEval()
    with pytest.raises(ValueError):
        rval = f(None, None, return_dict=True)

    assert f(None, None) == 1
    assert f(None, None, return_dict=False) == 1

    rval = f(None, None, return_dict=True, key_for_non_dict='a')
    assert rval == {'a': 1}

