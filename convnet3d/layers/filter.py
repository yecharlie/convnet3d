import keras
import keras.backend as K
from .. import backend


def filterDetections(
    boxes,
    probs,
    score_threshold   = 0.05,
    nms               = True,
    nms_threshold     = 0.5,
    max_detections    = 100,
    explicit_bg_class = True
):
    print('filterDetection-boxes-shape',boxes)
    # get the reference
    scores  = K.max(probs, axis=1)
    labels  = K.argmax(probs, axis=1)

    # perform score thresholding 
    indices = backend.where(K.greater(scores, score_threshold))  # 2-D

    if explicit_bg_class:
        # assume that backgroud class is labeled as 0, we will then remove those "detections" 
        bg_class_label = 0

        # repick the reference
        filtered_labels = K.gather(labels, indices)[:, 0]
        selected_indices = backend.where(K.not_equal(filtered_labels, bg_class_label))[:, 0]
        indices = K.gather(indices, selected_indices)

    if nms:  # perform non-max-suppersion
        # repick the reference
        filtered_boxes  = backend.gather_nd(boxes, indices)  # 2-D

        print('filterDetection-nms-indices', indices)
        print('filterDetection-nms-filtered_boxes', filtered_boxes)
        filtered_scores = K.gather(scores,indices)[:, 0]  # 1-D

        overlaps         = backend.computeOverlaps(filtered_boxes, filtered_boxes)
        selected_indices = backend.non_max_suppression_overlaps(overlaps, filtered_scores, max_detections, nms_threshold)
        indices = K.gather(indices, selected_indices)  # 2-D

    # select top-k
    labels    = backend.gather_nd(labels, indices)  # 1-D
    indices   = K.stack([indices[:, 0], labels], axis = 1)  # 2-D
    scores    = backend.gather_nd(probs, indices)
    scores, top_indices  = backend.top_k(scores, k=K.minimum(max_detections, K.shape(scores)[0]))

    # get final valid indices
    indices  = K.gather(indices[:, 0], top_indices)
    boxes    = K.gather(boxes, indices)
    labels   = K.gather(labels, top_indices)

    # pad the outputs
    pad_size = K.maximum(0, max_detections - K.shape(scores)[0])

    # boxes are padded with zero because they will be supplied d to crop rois
    boxes    = backend.pad(boxes, [[0, pad_size], [0, 0]], constant_values=0)
    scores   = backend.pad(scores, [[0, pad_size]], constant_values=-1)
    labels   = backend.pad(labels, [[0, pad_size]], constant_values=-1)
    labels   = K.cast(labels, 'int32')

    # set shape, since we know what they are
    boxes.set_shape([max_detections, 6])
    scores.set_shape([max_detections])
    labels.set_shape([max_detections])

    return [boxes, scores, labels]


class Filter(keras.layers.Layer):
    '''Keras layer for filtering detection
    '''
    def __init__(
        self,
        score_threshold     = 0.05,
        nms                 = True,
        nms_threshold       = 0.5,
        max_detections      = 100,
        explicit_bg_class   = True,
        parallel_iterations = 32,
        **kwargs
    ):
        self.score_threshold     = score_threshold
        self.nms                 = nms
        self.nms_threshold       = nms_threshold
        self.max_detections      = max_detections
        self.explicit_bg_class   = explicit_bg_class
        self.parallel_iterations = 32
        super(Filter, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        boxes, probs = inputs

        def _filterDetection(args):
            return filterDetections(
                boxes = args[0],
                probs = args[1],
                score_threshold   = self.score_threshold,
                nms               = self.nms,            
                nms_threshold     = self.nms_threshold,
                max_detections    = self.max_detections,
                explicit_bg_class = self.explicit_bg_class
            )

        outputs = backend.map_fn(
            _filterDetection,
            elems=[boxes, probs],
            dtype=[K.floatx(), K.floatx(), 'int32'],
            parallel_iterations = self.parallel_iterations
        )
        return outputs

    def compute_output_shape(self, input_shapes):
        return [
            (input_shapes[0][0], self.max_detections, 6),
            (input_shapes[0][0], self.max_detections),
            (input_shapes[0][0], self.max_detections)
        ]
    
    def compute_mask(self, inputs, mask):
        '''Just the copy from other code
        '''
        return (len(inputs) + 1) * [None]

    def get_config(self):
        config = super(Filter, self).get_config()
        config.update({
            'score_threshold'     :  self.score_threshold,        
            'nms'                 :  self.nms,                
            'nms_threshold'       :  self.nms_threshold,      
            'max_detections'      :  self.max_detections,     
            'explicit_bg_class'   :  self.explicit_bg_class,
            'parallel_iterations' :  self.parallel_iterations
        })
        return config
