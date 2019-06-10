import numpy as np
from numpy.testing import assert_array_equal
from convnet3d.utils.window import windowing


def test_windowing():
    image_size    = (50, 200, 200)
    window_size   = (50, 120, 120)
    slide_strides = (50, 100, 100)

    actual = windowing(image_size, window_size, slide_strides)
    expected = np.array([
        [0, 0,  0],
        [0, 0,  80],
        [0, 80, 0],
        [0, 80, 80]
    ])
    assert_array_equal(actual, expected)
