import keras
import keras.backend as K

from . import (
    ResidualUnit,
    CBALayers,
    BatchNorm,
    loadModel
)
from deprecated import deprecated


def reductionModel1b(
    input_feature_size = 1,
    roi_size = [25, 60, 60],
    num_classes = 2,
    cs_model_path = None,
    name = 'false_positive_reduction_model_1b'
):
    input_shape = ((input_feature_size,) + tuple(roi_size)) if K.image_data_format == 'channels_first' else (tuple(roi_size) + (input_feature_size,))
    inputs = keras.layers.Input(shape=input_shape, name='fpr_input')

    outputs = CBALayers(
        coptions={
            'kernel_size' : [4, 5, 5],
            'filters'     : 32,
            'name'        : 'C1',
            'kernel_regularizer': keras.regularizers.l2(1e-4)
        },
        boptions={'name': 'BN1'},
        name='A1'
    )(inputs)

    outputs = keras.layers.MaxPooling3D(
        pool_size   = [1, 2, 2],
        strides     = [1, 2, 2],
        padding     = 'valid',
        name        = 'P1'
    )(outputs)

    outputs = CBALayers(
        coptions={
            'kernel_size' : [4, 5, 5],
            'filters'     : 32,
            'name'        : 'C2',
            'kernel_regularizer': keras.regularizers.l2(1e-4)
        },
        boptions={'name': 'BN2'},
        name='A2'
    )(outputs)

    outputs = CBALayers(
        coptions={
            'kernel_size' : [5, 5, 5],
            'filters'     : 64,
            'name'        : 'C3',
            'kernel_regularizer': keras.regularizers.l2(1e-4)
        },
        boptions={'name': 'BN3'},
        name='A3'
    )(outputs)
    # Note that the previous layers ' configuration ara consistent with corrsponding layers in candidates screening model except the l2 norm.

    outputs = keras.layers.Conv3D(
        kernel_size       =[3, 3, 3],
        strides           =1,
        filters           =64,
        name              ='C4',
        kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        bias_initializer  ='zeros',
        kernel_regularizer=keras.regularizers.l2(1e-4)
    )(outputs)

    outputs = ResidualUnit(
        coptions1={'kernel_size': [3, 3, 3], 'filters': 64, 'kernel_regularizer': keras.regularizers.l2(1e-4)},
        coptions2={'kernel_size': [3, 3, 3], 'filters': 64, 'kernel_regularizer': keras.regularizers.l2(1e-4)},
        ruid=1
    )(outputs)

    outputs = BatchNorm(name='BN4')(outputs)
    outputs = keras.layers.Activation('relu', name='A4')(outputs)
#    reg_flatten = keras.layers.Flatten(name='FReg')(outputs)
#    reg_dense = keras.layers.Dense(128,activation='sigmoid',name='DReg')(reg_flatten)
#    reg_outputs = keras.layers.Dense(4,name='regression')(reg_dense)

    outputs = keras.layers.Conv3D(
        padding           ='valid',
        strides           =2,
        kernel_size       =[3, 3, 3],
        filters           =128,
        name              ='C5',
        kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        bias_initializer  ='zeros',
        kernel_regularizer=keras.regularizers.l2(1e-4)
    )(outputs)

    outputs = ResidualUnit(
        coptions1={'kernel_size': [3, 3, 3], 'filters': 128, 'kernel_regularizer': keras.regularizers.l2(1e-4)},
        coptions2={'kernel_size': [3, 3, 3], 'filters': 128, 'kernel_regularizer': keras.regularizers.l2(1e-4)},
        ruid=2
    )(outputs)

    outputs = BatchNorm(name='BN5')(outputs)
    outputs = keras.layers.Activation('relu', name='A5')(outputs)
    cls_flatten = keras.layers.Flatten(name='FCls')(outputs)
    cls_dense = keras.layers.Dense(128, activation='sigmoid', name='DCls')(cls_flatten)
    cls_outputs = keras.layers.Dense(num_classes, activation='softmax', name='classification')(cls_dense)

#    reduction_model = keras.models.Model(inputs=inputs, outputs=[reg_outputs, cls_outputs])
    reduction_model = keras.models.Model(inputs=inputs, outputs=[cls_outputs])
    if cs_model_path is not None:
        trained_cs_model = loadModel(cs_model_path)
        # use the weights of front layers from trained
        # candidates screening model to initiate reduction model 's'
        # corresponding layers
        layers_names = ['C1', 'BN1', 'C2', 'BN2', 'C3', 'BN3']
        for ln in layers_names:
            layer_weights = trained_cs_model.get_layer(ln).get_weights()

            # have the same layers names
            reduction_model.get_layer(ln).set_weights(layer_weights)

    return reduction_model


@deprecated(version='1.0', reason='Use reductionModel1b instead')
def reductionModel(
    input_feature_size = 1,
    roi_size = [25, 60, 60],
    num_classes = 2,
    cs_model_path = None,
    name = 'false_positive_reduction_model'
):
    input_shape = ((input_feature_size,) + tuple(roi_size)) if K.image_data_format == 'channels_first' else (tuple(roi_size) + (input_feature_size,))
    inputs = keras.layers.Input(shape=input_shape, name='fpr_input')

    outputs = CBALayers(
        coptions={
            'kernel_size' : [4, 5, 5],
            'filters'     : 32,
            'name'        : 'C1',
            'kernel_regularizer': keras.regularizers.l2(1e-4)
        },
        boptions={'name': 'BN1'},
        name='A1'
    )(inputs)

    outputs = keras.layers.MaxPooling3D(
        pool_size   = [1, 2, 2],
        strides     = [1, 2, 2],
        padding     = 'valid',
        name        = 'P1'
    )(outputs)

    outputs = CBALayers(
        coptions={
            'kernel_size' : [4, 5, 5],
            'filters'     : 32,
            'name'        : 'C2',
            'kernel_regularizer': keras.regularizers.l2(1e-4)
        },
        boptions={'name': 'BN2'},
        name='A2'
    )(outputs)

    outputs = CBALayers(
        coptions={
            'kernel_size' : [5, 5, 5],
            'filters'     : 64,
            'name'        : 'C3',
            'kernel_regularizer': keras.regularizers.l2(1e-4)
        },
        boptions={'name': 'BN3'},
        name='A3'
    )(outputs)
    # Note that the previous layers ' configuration ara consistent with corrsponding layers in candidates screening model except the l2 norm.

    outputs = keras.layers.Conv3D(
        kernel_size       =[3, 3, 3],
        strides           =1,
        filters           =64,
        name              ='C4',
        kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        bias_initializer  ='zeros',
        kernel_regularizer=keras.regularizers.l2(1e-4)
    )(outputs)

    outputs = ResidualUnit(
        coptions1={'kernel_size': [3, 3, 3], 'filters': 64, 'kernel_regularizer': keras.regularizers.l2(1e-4)},
        coptions2={'kernel_size': [3, 3, 3], 'filters': 64, 'kernel_regularizer': keras.regularizers.l2(1e-4)},
        ruid=1
    )(outputs)

    outputs = BatchNorm(name='BN4')(outputs)
    outputs = keras.layers.Activation('relu', name='A4')(outputs)
    reg_flatten = keras.layers.Flatten(name='FReg')(outputs)
    reg_dense = keras.layers.Dense(128, activation='sigmoid', name='DReg')(reg_flatten)
    reg_outputs = keras.layers.Dense(4, name='regression')(reg_dense)

    outputs = keras.layers.Conv3D(
        padding           ='valid',
        strides           =2,
        kernel_size       =[3, 3, 3],
        filters           =128,
        name              ='C5',
        kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        bias_initializer  ='zeros',
        kernel_regularizer=keras.regularizers.l2(1e-4)
    )(outputs)

    outputs = ResidualUnit(
        coptions1={'kernel_size': [3, 3, 3], 'filters': 128, 'kernel_regularizer': keras.regularizers.l2(1e-4)},
        coptions2={'kernel_size': [3, 3, 3], 'filters': 128, 'kernel_regularizer': keras.regularizers.l2(1e-4)},
        ruid=2
    )(outputs)

    outputs = BatchNorm(name='BN5')(outputs)
    outputs = keras.layers.Activation('relu', name='A5')(outputs)
    cls_flatten = keras.layers.Flatten(name='FCls')(outputs)
    cls_dense = keras.layers.Dense(128, activation='sigmoid', name='DCls')(cls_flatten)
    cls_outputs = keras.layers.Dense(num_classes, activation='softmax', name='classification')(cls_dense)

    reduction_model = keras.models.Model(inputs=inputs, outputs=[reg_outputs, cls_outputs])
    if cs_model_path is not None:
        trained_cs_model = loadModel(cs_model_path)
        # use the weights of front layers from trained
        # candidates screening model to initiate reduction model 's'
        # corresponding layers
        layers_names = ['C1', 'BN1', 'C2', 'BN2', 'C3', 'BN3']
        for ln in layers_names:
            layer_weights = trained_cs_model.get_layer(ln).get_weights()

            # have the same layers names
            reduction_model.get_layer(ln).set_weights(layer_weights)

    return reduction_model
