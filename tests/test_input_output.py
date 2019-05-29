import numpy as np
import os

import tensorflow as tf

from convnet3d.models.candidates_screening import candidates_screening_model
from convnet3d.models.false_positives_reduction import fp_reduction_model

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#sess = get_session()
#sess.run(tf.initialize_all_variables())
#
#data = np.random.random((1,50,30,30,1))
#model = candidates_screening_model()
#model2 = fp_reduction_model(model)
#result2 = model2.predict(data)
#print(result2.shape)
#model2.summary()
#result = model.predict(data)
#print("result shape:",result.shape)
