import warnings
import keras
from six import raise_from
from deprecated import deprecated
from . import (
    detectionModel,
    detectionPred,
    reductionModel,
    submodels
)
from .. import layers
from ..utils.index_mapping import IndexMap


def convnet3dModel1b(
    fpr_model        = None,
    cs_model         = None,
    map_args         = None,
    max_proposals    = 20,
    score_threshold  = 0.25,
    nms_threshold    = 0.5,
    roi_size         = (25, 60, 60),
    name             = '3d_convolutional_model_1b',
):
    '''The completed convolutional 3d model

    When training the reduction model, this function could convert it to a predicted model.

    Note that the background calss is EXPLICITLY encoded in classification outputs of these models , i.e. num_channels == num_classes. :

    Args:
        fpr_model        : The False Positive Reduction model, could be a model with multi-outputs (regression,classification) or a list of submodels. The first submodel/subouput of fpr_model is regarded as regression model, the second is classification model.
        cs_model         : The candidates screening model to generate proposals. 
        map_args         : Establishs a affine mapping (with diagonal matrix) from output indices back to input coordinates. in_coords = out_coords * map_args[0] + map_args[1]. If not given, this argument is estimated automatically according to cs model architecture.
        max_proposals    : Maximum number of proposals
        score_threshold : Scores threshold for non maximum supression (nms).
        nms_threshold    : The nms iou threshold.
        roi_size         : size of roi for fpr model.
        name             : Model name '3d_convolutional_model' by default.

    Returns:
        convnet3d model  : Which has outputs [boxes, scores, labels, regression, classification].
    '''

    if not fpr_model or isinstance(fpr_model, keras.models.Model):
        if not fpr_model:
            default_roi_size = (25, 60, 60)
            assert roi_size == default_roi_size

            # the default fpr model
            fpr_model = reductionModel()

        sub_roi_models = submodels(fpr_model)
    else:
        raise ValueError('Unrecognized argument \'fpr_model\'.')

    assert len(sub_roi_models) == 1

    if not cs_model:
        # the default cs model
        cs_model = detectionModel()

    # stage 1
    cs_pred = detectionPred(
        cs_model,
        map_args,
        roi_size,
        max_proposals,
        score_threshold,
        nms_threshold
    )
    inputs = cs_pred.inputs
    image = inputs[0]
    proposals = cs_pred.outputs

    # Stage 2
    boxes  = proposals[0]
    rois   = layers.RoiCropper(roi_size, name='roi_cropper')([image, boxes])  # roi with one channel
#    print('rois: {}; fpr_input_shape: {}.'.format(rois, fpr_input_shape))

    cls = keras.layers.TimeDistributed(sub_roi_models[0])(rois)

    # do localization by classifiacation results only
    return keras.models.Model(inputs=inputs, outputs=[cls, boxes])


@deprecated(version='1.0', reason='Use convnet3dModel1b instead')
def convnet3dModel(
    fpr_model        = None,
    cs_model         = None,
    map_args         = None,
    max_proposals    = 20,
    score_threshold  = 0.25,
    nms_threshold    = 0.5,
    roi_size         = (25, 60, 60),
    name             = '3d_convolutional_model',
):
    '''The completed convolutional 3d model

    When training the reduction model, this function could convert it to a predicted model.

    Note that the background calss is EXPLICITLY encoded in classification outputs of these models , i.e. num_channels == num_classes.

    Args:
        fpr_model        : The False Positive Reduction model, could be a model with multi-outputs (regression,classification) or a list of submodels. The first submodel/subouput of fpr_model is regarded as regression model, the second is classification model.
        cs_model         : The candidates screening model to generate proposals.
        map_args         : Establishs a affine mapping (with diagonal matrix) from output indices back to input coordinates. in_coords = out_coords * map_args[0] + map_args[1]. If not given, this argument is estimated automatically according to cs model architecture.
        max_proposals    : Maximum number of proposals
        score_threshold : Scores threshold for non maximum supression (nms).
        nms_threshold    : The nms iou threshold.
        roi_size         : size of roi for fpr model.
        name             : Model name '3d_convolutional_model' by default.

    Returns:
        convnet3d model  : Which has outputs [boxes, scores, labels, regression, classification].
    '''

    if not fpr_model or isinstance(fpr_model, keras.models.Model):
        if not fpr_model:
            default_roi_size = (25, 60, 60)
            assert roi_size == default_roi_size

            # the default fpr model
            fpr_model = reductionModel()

        sub_roi_models = submodels(fpr_model)
    elif isinstance(fpr_model, (list, tuple)):
        sub_roi_models = fpr_model
    else:
        raise ValueError('Unrecognized argument \'fpr_model\'.')

    if len(sub_roi_models) < 2:
        raise ValueError('The FPR model must has a regression batch as well as a classification batch.')
    elif len(sub_roi_models) > 2:
        warnings.warn('The FPR model has more than two outputs batch, the trailing batchs are discarded.')
        sub_roi_models = sub_roi_models[:2]
    roi_models_names = ['regression', 'classification']

    if not cs_model:
        # the default cs model
        cs_model = detectionModel()

    if not map_args:
        try:
            imap = IndexMap(cs_model)
            map_args = imap.D, imap.C
        except ValueError as e:
            raise_from(ValueError('Couldn\'t automatically infer map arguments. Please set it manually.'), e)

    # Stage 1
    inputs = cs_model.inputs
    image = inputs[0]
    image_shape = layers.Shape()(image)
    probabilities = cs_model.outputs[0]

    boxes, probabilities = layers.Boxes(
        box_size = roi_size,
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

    # Stage 2
    boxes  = proposals[0]
    labels = proposals[2]
    rois    = layers.RoiCropper(roi_size, name='roi_cropper')([image, boxes])  # roi with one channel
#    print('rois: {}; fpr_input_shape: {}.'.format(rois, fpr_input_shape))
    vision_field, labels = layers.ResizeBoxes(roi_size, mode='all', name='cnn_vision_field')([boxes, labels])

    other_outputs = []
    for idx, (name, roi_model) in enumerate(zip(roi_models_names, sub_roi_models)):
        # run all the roi models
        # Indeed, in our framework there are only teo submodels
        output = keras.layers.TimeDistributed(roi_model)(rois)
        if name == 'regression':
            reg = output
        elif name == 'classification':
            cls = output
        else:
            other_outputs.append(output)

    reg_boxes  = layers.RegressBoxes(name='regress_boxes')([vision_field, reg])

    return keras.models.Model(inputs=inputs, outputs=[reg_boxes] + [reg, cls] + other_outputs)
