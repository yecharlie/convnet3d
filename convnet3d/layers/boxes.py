#import warnings
import keras
import keras.backend as K
import numpy as np

from .. import backend

class Boxes(keras.layers.Layer):
    def __init__(
        self, 
        box_size, 
        D=1,
        C=0, 
#        num_classes=1, 
        parallel_iterations=32, 
        **kwargs
    ):
        self.D = np.array(D)
        self.C = np.array(C)
        self.size = np.array(box_size)
        self.parallel_iterations = parallel_iterations
        super(Boxes, self).__init__(**kwargs)

    def call(self, inputs):
        def _generateBoexs(probs):
            shape = K.shape(probs)

            a = K.arange(shape[0])
            b = K.arange(shape[1])
            c = K.arange(shape[2])
            A,B,C = backend.meshgrid(a,b,c, indexing='ij')
            indices = K.concatenate([
                K.expand_dims(A,axis=-1),
                K.expand_dims(B,axis=-1),
                K.expand_dims(C,axis=-1),
                ],axis=-1
            )
            indices = K.reshape(indices,(-1,3))
            indices = K.cast(indices,K.floatx())
            mapped = indices * self.D + self.C
            boxes = backend.tobbox(mapped, self.size)
            probs = K.reshape(probs,(-1,shape[3]))
            return [boxes, probs]#list! list! list!


        probs = inputs#Notes: inputs is not a list
        if K.image_data_format() == 'channels_first':
            probs = K.permute_dimensions(probs,(0,2,3,4,1))
            #Notes: from now on, data is inconsistent with image_data_format
#            warnings.warn('The boxes layer has permuted data, which is no longer consistant with keras.backend.image_data_format()')
        probs_shape = K.int_shape(probs)
        assert len(probs_shape) == 5,'The outputs shape of candidates screening model is inconsitent: {}.'.format(probs_shape)

#        if probs_shape[-1] == num_classes + 1:
#            print('Its detected that the backgroud class has been encoded into the result probabilites, retriving the leading {} channels probabilities out of {} chanels outputs.'.format(num_classes, probs_shape[-1]))
#            probs = probs[:,:,:,:,:-1]
#        elif probs_shape[-1] != num_classes:
#            raise ValueError('num_classes{} don\'t match the outputs shape {} of cs model'.format(num_classes, probs_shape))

        outputs = backend.map_fn(
            _generateBoexs,
            elems=probs,
            dtype=[K.floatx(), K.floatx()],
            parallel_iterations = self.parallel_iterations
        )
        return outputs

    def compute_output_shape(self, input_shape):
        prob_shape = input_shape
        if K.image_data_format() == 'channels_first':
            channels = prob_shape[1]
        else:
            channels = prob_shape[4]

        batch_size = prob_shape[0]

        num_boxes = 1
        for dim in prob_shape[1:]:
            if dim is None:
                num_boxes = None
                break
            else:
                num_boxes *= dim
        if num_boxes is not None:
            num_boxes /= channels

#        if channels = self.num_classes + 1:
#            probs_channels = channels - 1
#        else:
#            probs_channels = channels
            
        return [(batch_size, num_boxes, 6), (batch_size,num_boxes,channels)]

    def get_config(self):
        config = super(Boxes,self).get_config()
        config.update({
            'D'                   : self.D.tolist(),
            'C'                   : self.C.tolist(),
            'box_size'            : self.size.tolist(),
            'parallel_iterations' : self.parallel_iterations
        })
        return config

class ResizeBoxes(keras.layers.Layer):
    '''keras layers for resizing the existed bounding boxes
    '''

    def __init__(
        self, 
        target_size,
        mode='partial',
        parallel_iterations=32,
        **kwargs
    ):
        if mode not in ['partial', 'all']:
            raise ValueError("There are two kind of mode:1-'partial' (by default), 2-'all'." )

        self.mode = mode
        self.target_size = np.array(target_size, dtype=K.floatx())

        self.parallel_iterations = parallel_iterations
        super(ResizeBoxes, self).__init__(**kwargs)

    def call(self, inputs):
        boxes, labels = inputs

        def _resizeAll(elems, new_size = self.target_size):
            boxes = elems[0]
            labels = elems[1]
            centroids = (boxes[:,1::2] + boxes[:,0::2]) // 2
            resized   = backend.tobbox(centroids, new_size)
            return [resized, labels]

        def _resizePartial(elems, new_size = self.target_size):
            boxes = elems[0]
            labels = elems[1]
            indices = backend.where(K.greater_equal(labels, 0))
            filtered_boxes = backend.gather_nd(boxes, indices)
            centroids = (filtered_boxes[:,1::2] + filtered_boxes[:, 0::2]) // 2
            resized   = backend.tobbox(centroids, new_size)
            ignored_indices = backend.where(K.less(labels, 0))
            ignored   = backend.gather_nd(boxes, ignored_indices)
            outputs_boxes   = K.concatenate([resized, ignored], axis = 0)

            #The following operation is redundant
            #because we know that the ignored boxes are
            #at the trail.
            outputs_labels  = K.concatenate([
                K.gather(labels, indices)[:,0],
                K.gather(labels, ignored_indices[:,0])
            ])
            return [outputs_boxes, outputs_labels]

        outputs = backend.map_fn(
            _resizePartial if self.mode == 'partial' else _resizeAll,
            elems               = [boxes, labels],
            parallel_iterations = self.parallel_iterations
        )
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[:2]

    def get_config(self):
        config = super(ResizeBoxes, self).get_config()
        config.update({
            'target_size'         : self.target_size.tolist(),
            'mode'                : self.mode,
            'parallel_iterations' : self.parallel_iterations
        })

class ClipBoxes(keras.layers.Layer):
    '''Return Keras layers to clip box values to lie inside a given shape.
    '''
    def __init__(self, **kwargs):
        super(ClipBoxes, self).__init__(**kwargs)

    def call(self, inputs):
        def _clipBoxes(inputs):
            boxes, shape = inputs
            shape = K.cast(shape, K.floatx())#to avoid ValueError when do clip
            if K.image_data_format() == 'channels_first':
                max_a = shape[2]
                max_b = shape[3]
                max_c = shape[4]
            else:
                max_a = shape[1]
                max_b = shape[2]
                max_c = shape[3]

            #Use backend.clip_by_value instead of K.clip
            #because it raises a TypeError in some version of keras
            a1 = backend.clip_by_value(boxes[:,:,0],0,max_a)
            a2 = backend.clip_by_value(boxes[:,:,1],0,max_a)
            b1 = backend.clip_by_value(boxes[:,:,2],0,max_b)
            b2 = backend.clip_by_value(boxes[:,:,3],0,max_b)
            c1 = backend.clip_by_value(boxes[:,:,4],0,max_c)
            c2 = backend.clip_by_value(boxes[:,:,5],0,max_c)
            return K.stack([a1,a2,b1,b2,c1,c2],axis=2)
        return _clipBoxes(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

class RegressBoxes(keras.layers.Layer):
    '''Keras layer for applying regression values to boxes
    '''

    def __init__(self, mean=None, std=None, **kwargs):
        '''Initializer
        
        Args:
            mean  : The mean value used when computing deltas (defaults to [0, 0, 0, 0]).
            std   : The standard deviation used when computing deltas (defaults to [0.2, 0.2, 0.2, 0.2]).
        '''
        if mean is None:
            mean = [0, 0, 0, 0]
        if std is None:
            std = [0.2, 0.2, 0.2, 0.2 ]
        self.mean = np.array(mean)
        self.std  = np.array(std)
        super(RegressBoxes, self).__init__(**kwargs)
        
    def call(self, inputs, **kwargs):
        boxes, regression = inputs
        return backend.bboxTransformInv_v2(boxes, regression, mean=self.mean, std=self.std)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(RegressBoxes, self).get_config()
        config.update({
            'mean': self.mean.tolist(),
            'std' : self.std.tolist()
        })
