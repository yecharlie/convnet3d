import keras
from keras import backend as K


def ResidualUnit(coptions1, coptions2, ruid):
    '''A Residual Unit

    Structure:
     _ _ _ _ _ _ _ _ _ _ _ _
    /                       \\
    -->BN1-A1-C1-BN2-A2-C2-Add->

    The default choice are:
        * C1,C2: The 3d convolutional layer (fixed),  padding='same' (fixed), strides=1, kernel_initializer=normal(mean=0.0,stddev=0.01), bias_initializer='zeros'
        * BN!, BN2: BatchNorm layer (fixed), 'axis' is set properly.
        * A1, A2: Activation layer, 'relu'.
        * Add: Add layer
    These layers 's names are set properly based on 'ruid'.
    '''
    def _ResidualUnit(inputs):
        layers_names = ['RU{}-BN1', 'RU{}-A1', 'RU{}-C1', 'RU{}-BN2', 'RU{}-A2', 'RU{}-C2', 'RU{}-Add']
        layers_names = [ ln.format(ruid) for ln in layers_names]

        outputs = BatchNorm(name=layers_names[0])(inputs)
        outputs = keras.layers.Activation('relu', name=layers_names[1])(outputs)

        coptions1.update(
            name=layers_names[2],
            padding='same'
        )
        outputs = CBALayers(
            coptions1,
            {'name': layers_names[3]},
            name=layers_names[4]
        )(outputs)

        coptions2.update(
            name=layers_names[5],
            padding='same',
            strides=1,
            kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros'
        )
        outputs = keras.layers.Conv3D(**coptions2)(outputs)
        outputs = keras.layers.Add(name=layers_names[6])([inputs, outputs])
        return outputs
    return _ResidualUnit


def CBALayers(coptions, boptions={}, activation='relu', **kwargs):
    '''Add convolution-batchnorm-activation layer successively

    The default choice:
        * The 3d convolutional layer (fixed),  padding='valid', strides=1, kernel_initializer=normal(mean=0.0,stddev=0.01), bias_initializer='zeros'
        * BatchNorm layer (fixed), 'axis' is set properly.
        * Activation layer, 'relu'.
    Alter the default choice or add new options with coption, boption, activation respectively.
    Other arguments are sent to keras.layers.Activation

    Arguments:
        coptions
        boptions
        activation
        **kargs

    Returns:
        a functor served as CBA layer
    '''
    def _CBALayers(inputs):
        cur_coptions = {
            "padding"     : 'valid',
            "strides"     : 1,
            'kernel_initializer': keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            'bias_initializer': 'zeros'
        }
        cur_coptions.update(coptions)
        outputs = keras.layers.Conv3D(**cur_coptions)(inputs)

        outputs = BatchNorm(**boptions)(outputs)

        outputs = keras.layers.Activation(activation, **kwargs)(outputs)
        return outputs

    return _CBALayers


def BatchNorm(**kwargs):
    def _BatchNorm(inputs):
        boptions = {'axis': 1 if K.image_data_format() == 'channels_first' else -1}
        boptions.update(kwargs)
        return keras.layers.normalization.BatchNormalization(**boptions)(inputs)

    return _BatchNorm


def submodels(model):
    models = [keras.models.Model(inputs=model.inputs, outputs=output) for output in model.outputs]
    return models


def loadModel(filepath):
    import keras.models

    from .. import losses
    custom_objects = {'_reductionRegLoss': losses.reductionRegLoss()}
    return keras.models.load_model(filepath, custom_objects=custom_objects)


# import models
from .candidates_screening import (  # noqa: F401
    detectionModel,
    detectionPred)
from .false_positives_reduction import (  # noqa: F401
    reductionModel,
    reductionModel1b)
from .convnet3d import (  # noqa: F401
    convnet3dModel,
    convnet3dModel1b)
