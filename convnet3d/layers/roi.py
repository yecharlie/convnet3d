import numpy as np

import keras
import keras.backend as K
from .. import backend


class RoiCropper(keras.layers.Layer):
    '''Cropping roi
    '''

    def __init__(self, roi_size, parallel_iterations=32, **kwargs):
        self.roi_size = np.array(roi_size, dtype='int32')
        self.parallel_iterations = parallel_iterations
        super(RoiCropper, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        image, boxes = inputs
        image = K.stop_gradient(image)
        boxes = K.stop_gradient(K.cast(boxes, dtype='int32'))
#        fpr_input = K.stop_gradient(fpr_input)
#
#        fpr_input_shape = K.cast(K.shape(fpr_input), dtype='int32')
#        fpr_int_shape = K.int_shape(fpr_input)

#        assert len(fpr_int_shape) == 5

#        target_size = self.roi_size[:3]
#        self.roi_shape = fpr_int_shape[1:]

        def _crop(args, target_size = self.roi_size):
            image, boxes = args

            def wrapped_crop_to_bounding_box(box):
                return backend.crop_to_bounding_box_3d(image, box, target_size)

            cropped = backend.map_fn(
                wrapped_crop_to_bounding_box,
                elems=boxes,
                dtype = K.floatx()
            )
            return cropped

        cropped_batch = backend.map_fn(
            _crop,
            elems=[image, boxes],
            dtype = K.floatx(),
            parallel_iterations=self.parallel_iterations
        )
        return cropped_batch

    def get_config(self):
        config = super(RoiCropper, self).get_config()
        config.update(

            roi_size            = self.roi_size.tolist(),
            parallel_iterations = self.parallel_iterations
        )

    def compute_output_shape(self, input_shape):
        if K.image_data_format() == 'channels_last':
            channels = input_shape[0][-1]
            roi_shape = tuple(self.roi_size) + (channels,)
        else:
            channels = input_shape[0][1]
            roi_shape = (channels,) + tuple(self.roi_size)
        return (input_shape[0][0] , input_shape[1][1]) + roi_shape
