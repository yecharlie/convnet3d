import keras
import keras.backend as K

class Shape(keras.layers.Layer):
    def call(self, inputs):
        return K.shape(inputs)

    def compute_output_shape(self, input_shape):
        return (len(input_shape),)

class Cast(keras.layers.Layer):
    def __init__(self, dtype, **kwargs):
        self.dtype = dtype
        super(Cast, self).__init__(**kwargs)

    def call(self, inputs):
        return K.cast(inputs, self.dtype)

    def get_config(self):
        config = super(Cast, self).get_config()
        config.update(dtype=self.dtype)
        return config

