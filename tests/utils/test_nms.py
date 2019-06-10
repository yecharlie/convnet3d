import numpy as np
from numpy.testing import assert_array_equal

from convnet3d.utils.nms import nmsOverlaps
from convnet3d.utils.annotations import computeOverlaps


def test_nms_overlaps():
    boxes    = np.array([
        [0, 2, 0, 2, 0, 2],  # suppressed
        [0, 2, 0, 2, 0, 2],
        [2, 5, 2, 5, 4, 6],
        [1, 4, 1, 4, 4, 6]
    ])
    scores   = np.array([0.5, 0.7, 0.9, 0.8])

    overlaps = computeOverlaps(boxes, boxes)
    actual   = nmsOverlaps(overlaps, scores, threshold=0.5)
    expected = [2, 3, 1]
    assert_array_equal(actual, expected)
