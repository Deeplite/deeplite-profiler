import pytest
import numpy as np
from unittest import mock
from deeplite.profiler.data_loader import DataLoader, ModelInputPattern, TensorSampler, ForwardPass
from tests.profiler_tests.unit import BaseUnitTest

class TestDataLoader(BaseUnitTest):
    def test_default_loader(self):
        defaultDL = DefaultDL([np.array([1]), np.array([2])], NumpyForwardPass(model_input_pattern=(0,)))
        iterdl = iter(defaultDL)
        assert next(iterdl).tolist() == [1]
        assert next(iterdl).tolist() == [2]
        assert len(defaultDL) == 2

        model = mock.MagicMock() 
        defaultDL.timed_forward(model, None)
        defaultDL.timed_random_forward(model, 1, None)
        defaultDL._forward_pass = None
        with pytest.raises(TypeError):
            defaultDL.sample_forward(model, None)
        with pytest.raises(TypeError):
            defaultDL.sample_random_forward(model, 1, None)

    def test_valid_pattern(self):
        p = ('_', 1, 2, '_', 0)
        mip = ModelInputPattern(p)
        x = (None, 'b', 'c', None, 'a')
        x_mip = mip.rearrange_from_pattern(x)

        assert not any(x_ is None for x_ in x_mip)
        assert x_mip == ('a', 'b', 'c')

        p = ('_',)
        mip = ModelInputPattern(p)
        x = 123
        x_mip = mip.rearrange_from_pattern(x)
        assert x_mip == tuple()

        p = (0,)
        mip = ModelInputPattern(p)
        x = 123
        x_mip = mip.rearrange_from_pattern(x)
        assert x_mip == (x,)

    def test_invalid_pattern(self):
        p = ('&', 1, 2, '_', 0)
        with pytest.raises(TypeError):
            mip = ModelInputPattern(p)

        p = (1, 1, 1, '_', 0)
        with pytest.raises(TypeError):
            mip = ModelInputPattern(p)

        p = ('_', 1, 2, '_', 0)
        mip = ModelInputPattern(p)
        x = (None, 'b',)
        with pytest.raises(ValueError):
            x_mip = mip.rearrange_from_pattern(x)

    def test_simple_pass(self):
        sample = (np.array([1, 2]), np.array([[2], [1]]),)
        # this forward is considered simple as it rely on a MIP and it is expecting common inputs.
        # in this case, the mip is providing a default implementation for `extract_model_inputs` and
        # the sampler is providing a default implementation for `create_random_model_inputs`
        fp = NumpyForwardPass(model_input_pattern=('_', 0), expecting_common_inputs=True)
        fp.infer_sampler(sample)

        # because of the mip, only the second element should be returned
        x = fp.perform(None, sample, None)
        assert len(x) == 1
        assert np.all(x[0] == sample[1])

        # because of the mip and the sampler, there should be only a second element like array with a first
        # axis of 10
        x = fp.random_perform(None, 10, None)
        assert len(x) == 1
        assert x[0].shape == (10,) + sample[1].shape

        shapes = fp.get_model_input_shapes()
        assert len(shapes) == 1
        assert shapes[0] == sample[1].shape

        fp = NumpyForwardPass(model_input_pattern=(1, 0), expecting_common_inputs=True)
        fp.infer_sampler(([np.array([1, 2]), np.array([3, 4])], {'x': [4, 5]},))
        shapes = fp.get_model_input_shapes()
        assert all(shp == (2,) for shp in shapes)

    def test_user_defined_pass(self):
        # use the 'simple' pass but without a mip, it does not provide an implementation and raises
        with pytest.raises(TypeError):
            NumpyForwardPass(model_input_pattern=None, expecting_common_inputs=True)

        # use the 'simple' pass but uncommon inputs, it does not provide an implementation and raises
        with pytest.raises(TypeError):
            NumpyForwardPass(model_input_pattern=('_', 0), expecting_common_inputs=False)
        with pytest.raises(TypeError):
            NumpyForwardPass(model_input_pattern=('_', 0), expecting_common_inputs=False)

        class UserDefinedPass(ForwardPass):
            def model_call(self, model, x, device):
                return x

            def extract_model_inputs(self, batch):
                x, y, z = batch
                return z**2

            def create_random_model_inputs(self, batch_size):
                return 1

            def get_model_input_shapes(self):
                return 7

            # the user would normally subclass a framework specific where this method is being taken care of
            @property
            def _tensor_sampler_cls(self):
                return 'dummy'

        fp = UserDefinedPass(expecting_common_inputs=False)
        sample = (1,2,3)
        out = fp.perform(None, sample, None)
        assert out == 3**2

        out = fp.random_perform(None, 123, None)
        assert out == 1

        assert 7 == fp.get_model_input_shapes()

        # this should do nothing
        fp.infer_sampler(None)
        assert fp._tensor_sampler is None


class DefaultDL(DataLoader):
    def dump_native_info(self):
        return

    def load_native_info(self, native_state):
        return self

    def _create_iter(self):
        return iter(self.native_dl)

    def __len__(self):
        return len(self.native_dl)

    @property
    def batch_size(self):
        return 1


class NumpyForwardPass(ForwardPass):
    def model_call(self, model, x, device):
        return x

    @property
    def _tensor_sampler_cls(self):
        return NumpySampler


class NumpySampler(TensorSampler):
    def _standardize_tensor(self, x):
        if isinstance(x, list):
            x = np.array(x)
        elif not isinstance(x, np.ndarray):
            raise ValueError()
        return x

    def _create_random_tensor(self, x_info, batch_size):
        return np.random.random((batch_size,) + x_info.shp).astype(x_info.dtype)

    def _get_info(self, x):
        return x.shape, x.dtype
