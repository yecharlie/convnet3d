import numpy as np
import keras

from deprecated import deprecated
from .generator import Generator

@deprecated(reason='use Generator instead')
class DetectionGenerator(Generator):
   def computeTargets(self,image_group, annotations_group):
        assert len(image_group)==len(annotations_group),"The length of the images and annotations should be equal."

        labels_batch = np.zeros((self.batch_size,),dtype=keras.backend.floatx())
        for idx,an in enumerate(annotations_group):
            labels_batch[idx]=an["labels"][0]
        return labels_batch
