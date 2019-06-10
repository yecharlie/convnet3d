import keras
import pytest

from convnet3d.models import (
    reductionModel,
    submodels,
    convnet3dModel
)


class TestReductionModel(object):
    def test_TimeDistributed(self):
        with pytest.raises(TypeError) :
            roi_input = keras.layers.Input(shape=(10, 25, 60, 60, 1))
            fpr_model = reductionModel()
            outputs = keras.layers.TimeDistributed(fpr_model)(roi_input)
            model   = keras.models.Model(inputs=roi_input, outputs=outputs)
            model.summary()

    def test_inputs_outputs(self):
        fpr_model = reductionModel()
        assert isinstance(fpr_model.inputs, list)
        assert isinstance(fpr_model.outputs, list)

    def test_TimeDistributed_with_submodels(self):
        roi_input = keras.layers.Input(shape=(10, 25, 60, 60, 1))
        fpr_model = reductionModel()
        sub_models = submodels(fpr_model)
        sub_models_outputs = [keras.layers.TimeDistributed(subm)(roi_input) for subm in sub_models]
        model = keras.models.Model(inputs=roi_input, outputs=sub_models_outputs)
        model.summary()


class TestConvnet3dModel(object):
    def test_simple(self):
        model = convnet3dModel()  # noqa: F401
