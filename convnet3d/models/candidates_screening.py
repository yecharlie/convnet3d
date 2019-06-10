import keras
from keras import backend as K
from six import raise_from
from . import CBALayers
from .. import layers
from ..utils.index_mapping import IndexMap


def detectionModel(
    name='candidate_screening_model',
    num_classes=2,
    input_feature_size=1
):
    '''The Detection Model

    Although this model receive arbitrary shape of input, its designed espicially to be trained with patches of which input_size = [15,30,30].
    '''
    input_size = [None, None, None]

    input_shape = ((input_feature_size,) + tuple(input_size)) if K.image_data_format == 'channels_first' else (tuple(input_size) + (input_feature_size,))
    inputs = keras.layers.Input(shape=input_shape)

    outputs = CBALayers(
        coptions={
            'kernel_size' : [4, 5, 5],
            'filters'     : 32,
            'name'        : 'C1'
        },
        boptions={'name': 'BN1'},
        name='A1'
    )(inputs)

    outputs = keras.layers.MaxPooling3D(
        pool_size   = [1, 2, 2],
        strides     = [1, 2, 2],
        padding     = 'valid',
        name        = 'P'
    )(outputs)

    outputs = CBALayers(
        coptions={
            'kernel_size' : [4, 5, 5],
            'filters'     : 32,
            'name'        : 'C2'
        },
        boptions={'name': 'BN2'},
        name='A2'
    )(outputs)

    outputs = CBALayers(
        coptions={
            'kernel_size' : [5, 5, 5],
            'filters'     : 64,
            'name'        : 'C3'
        },
        boptions={'name': 'BN3'},
        name='A3'
    )(outputs)

    outputs = CBALayers(
        coptions={
            'kernel_size' : [5, 5, 5],
            'filters'     : 250,
            'name'        : 'C4'
        },
        boptions={'name': 'BN4'},
        name='A4'
    )(outputs)

    outputs = CBALayers(
        coptions={
            'kernel_size' : [1, 1, 1],
            'filters'     : num_classes,
            'name'        : 'C5'
        },
        boptions={'name': 'BN5'},
        activation = 'softmax',
        name    ='A5'
    )(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def detectionPred(
    cs_model,
    map_args         = None,
    proposal_size    = (15, 30, 30),
    max_proposals    = 20,
    score_threshold  = 0.25,
    nms_threshold    = 0.5,
    name = 'candidates_screening_model_for_prediction'
):
    if not map_args:
        try:
            imap = IndexMap(cs_model)
            map_args = imap.D, imap.C
        except ValueError as e:
            raise_from(ValueError('Couldn\'t automatically infer map arguments. Please set it manually.'), e)

    inputs = cs_model.inputs
    image = inputs[0]
    image_shape = layers.Shape()(image)
    probabilities = cs_model.outputs[0]

    boxes, probabilities = layers.Boxes(
        box_size = proposal_size,
        D        = map_args[0],
        C        = map_args[1],
        name     = 'build_boxes'
    )(probabilities)

    boxes = layers.ClipBoxes(name='clip_boxes')([boxes, image_shape])

    proposals = layers.Filter(
        score_threshold = score_threshold,
        nms             = True,
        nms_threshold   = nms_threshold,
        max_detections  = max_proposals,
        explicit_bg_class = True,
        name='filter'
    )([boxes, probabilities])
    return keras.models.Model(inputs=inputs, outputs=proposals)
