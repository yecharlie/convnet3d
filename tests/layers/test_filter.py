import keras
import keras.backend as K

import numpy as np

from convnet3d import layers


class TestFilter(object):
    def test_simple(self):
        filter_layer = layers.Filter(score_threshold=0.65)
        boxes = np.array([[
            [0, 2, 0, 2, 0, 2],  # suppressed
            [0, 2, 0, 2, 0, 2],
            [0, 2, 0, 2, 4, 6],  # filtered:bg
            [2, 4, 2, 4, 4, 6]   # filtered: score
        ]],dtype = K.floatx())
        probs = np.array([[
            [0.1, 0.9],
            [0,   1],
            [0.7, 0.3],
            [0.4, 0.6]
        ]],dtype=K.floatx())

        actual_boxes,actual_scores,actual_labels = filter_layer.call([boxes, probs])
        actual_boxes = K.eval(actual_boxes)
        actual_scores = K.eval(actual_scores)
        actual_labels = K.eval(actual_labels)
        expected_boxes  = np.zeros((1, 100, 6), dtype=K.floatx())
        expected_boxes[0, 0, :] = boxes[0, 1, :]
        expected_scores = -1 * np.ones((1, 100), dtype=K.floatx())
        expected_scores[0, 0] = probs[0, 1, 1]
        expected_labels = -1 * np.ones((1, 100), dtype=K.floatx())
        expected_labels[0, 0] = 1

        np.testing.assert_array_equal(actual_boxes, expected_boxes)
        np.testing.assert_array_equal(actual_scores, expected_scores)
        np.testing.assert_array_equal(actual_labels, expected_labels)

    def test_mini_batch(self):
        filter_layer = layers.Filter(explicit_bg_class=False)
        boxes = np.array([
            [
                [2, 4, 2, 4, 0, 2],  # suppressed
                [2, 4, 2, 4, 0, 2],  # will not be filtered although its bg 
            ],
            [
                [6, 8, 6, 8, 6, 8],
                [6, 8, 6, 8, 6, 8],  # suppressed
            ]
        ], dtype=K.floatx())
        probs = np.array([
            [
                [0.1, 0.9],
                [1,  0]
            ],
            [
                [0,   1],
                [0.1, 0.9]
            ]
        ], dtype= K.floatx())
        actual_boxes, actual_scores,actual_labels = filter_layer.call([boxes, probs])
        actual_boxes = K.eval(actual_boxes)
        actual_scores = K.eval(actual_scores)
        actual_labels = K.eval(actual_labels)

        expected_boxes = np.zeros((2, 100, 6), dtype=K.floatx())
        expected_boxes[0, 0, :] = boxes[0, 1, :]
        expected_boxes[1, 0, :] = boxes[1, 0, :]

        expected_scores = -1 * np.ones((2,100), dtype=K.floatx())
        expected_scores[0, 0] = probs[0, 1, 0]
        expected_scores[1, 0] = probs[1, 0, 1]

        expected_labels = -1 * np.ones((2, 100), dtype=K.floatx())
        expected_labels[0, 0] = 0
        expected_labels[1, 0] = 1

        np.testing.assert_array_equal(actual_boxes, expected_boxes)
        np.testing.assert_array_equal(actual_scores, expected_scores)
        np.testing.assert_array_equal(actual_labels, expected_labels)
