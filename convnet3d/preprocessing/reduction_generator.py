import numpy as np
import keras

from deprecated import deprecated
from .generator import Generator

def _computeTargets(image_group, annotations_group, mean=None, std=None):
    assert len(image_group)==len(annotations_group),"The length of the images and annotations should be equal."

    if mean is None:
        mean = [0, 0, 0, 0]
    if std is None:
        std = [0.2, 0.2, 0.2, 0.2]

    batch_size = len(image_group)
    regression_batch = np.zeros((batch_size, 5),dtype=keras.backend.floatx())#the last elem stores label
    labels_batch = np.zeros((batch_size,))
    for i,(img, an) in enumerate(zip(image_group, annotations_group)):
        if keras.backend.image_data_format() == 'channels_last':
            depth, width, height, channels = img.shape
        else:
            channels, depth, width, height = img.shape

        #NOTE: because we desire the output to directly regress coordinates related to x,y,z,
        #here we use (height, width, depth) to form the target.
        img_cent = np.array([height // 2, width // 2, depth // 2])
        diagonal = np.sqrt(np.square(height) + np.square(width) + np.square(depth))
        if an['bboxes'].shape[0] > 0:
            bboxes = an['bboxes'][0]
            box_cent = (bboxes[::2] + bboxes[1::2]) / 2
            obj_d = (bboxes[1::2] - bboxes[::2]).max()
            tc = (box_cent - img_cent) / img_cent
            td = np.log(obj_d / diagonal)

            regression_batch[i][:3] = (tc - mean[:3]) / std[:3]
            regression_batch[i][3] = (td - mean[3]) / std[3]
        else:
            #negative sample
            regression_batch[i][:4] = -1

        regression_batch[i][4] = an['labels'][0]
        labels_batch[i] = an['labels'][0]

    return [regression_batch, labels_batch]

@deprecated(reason='this generator is maded for convnet3d2b. use Generator for convnet3d1b')
class ReductionGenerator(Generator):
    def computeTargets(self, image_group, annotations_group):
        return _computeTargets(image_group, annotations_group)
