import keras

import numpy as np
import SimpleITK as sitk

from .generator import Generator
from ..utils.image import (isotropic, readSeries)
from ..utils.tobbox import tobbox

def makeIsotropic(image, annotations):
    resampled, transform = isotropic(image)
    annotations['bboxes'] = annotations['bboxes'].copy()
    for index in range(annotations['bboxes'].shape[0]):
        annotations['bboxes'][index, :] = _transformBbox(annotations['bboxes'][index,:], image, transform)
    return resampled, annotations

def _transformBbox(box, image, transform):
    centroid = (box[::2] + box[1::2]) / 2
    sides = box[1::2] - box[::2]
    assert sides[0] == sides[1] == sides[2]

    #the diameter will not change as its the voxel coordinates in standard domain already.
    newc = image.TransformContinuousIndexToPhysicalPoint(centroid.astype(np.float64))
    newc = np.array(transform.GetInverse().TransformPoint(newc))
    return tobbox(newc, sides)

class ValidationGenerator(Generator):
    #@overrides
    def loadImage(self,image_index):
#        if getattr(self,'first_use_load', None) is None:
#            self.first_use_load = True
#            print('Use LoadImage in ValidationGenerator.')
        
        image_path = self.image_names[image_index]
        image = readSeries(image_path)
        #tricky: return a SimpleITK Image object intead of ndarray
        return image


    #@overrides
    def preprocessGroupEntry(self,image,annotations):
#        if getattr(self,'first_use_preprocess', None) is None:
#            self.first_use_preprocess = True
#            print('Use preprocessGroupEntry in ValidationGenerator.')
        image, annotations = makeIsotropic(image,annotations)
        #convert image to ndarray
        imgarr = sitk.GetArrayFromImage(image)
        if len(imgarr.shape) == 3:
            imgarr = imgarr.reshape(imgarr.shape+(1,))
        elif len(imgarr.shape) != 4:
            raise ValueError('Unsupported series data')

        return super(ValidationGenerator, self).preprocessGroupEntry(imgarr, annotations)
        

