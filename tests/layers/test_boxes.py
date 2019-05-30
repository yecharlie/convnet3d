import pytest 
import keras
import keras.backend as K
from convnet3d import layers
from convnet3d.utils.tobbox import tobbox

import numpy as np
from numpy.testing import assert_almost_equal

import os


class TestBoxes(object):
    def test_simple(self):
        boxes_layer = layers.Boxes(
            box_size = (10,20,20),
            D = (3,1,1),
            C = 1
        )

        probs = np.zeros((1,1,1,2,2),dtype=K.floatx())
        probs = K.constant(probs)

        actual_boxes, actual_probs = boxes_layer.call(probs)
        actual_boxes = K.eval(actual_boxes)
        actual_probs = K.eval(actual_probs)
        expected_boxes = np.zeros((1,2,6), dtype=K.floatx())
        #(0,0,0) * (3,1,1) + 1 = (1,1,1)
        #(1,1,1) -> (-4,6,-9,11,-9,11)
        expected_boxes[0,0] = np.array([-4,6,-9,11,-9,11],dtype=K.floatx())

        #(0,0,1) -> (1,1,2)
        #(1,1,2) -> (-4,6,-9,11,-8,12)
        expected_boxes[0,1] = np.array([-4,6,-9,11,-8,12],dtype=K.floatx())
        expected_probs = np.zeros((1,2,2),dtype=K.floatx())
        np.testing.assert_array_equal(actual_boxes,expected_boxes)
        np.testing.assert_array_equal(actual_probs,expected_probs)

    def test_mini_batch(self):
        boxes_layer = layers.Boxes(
            box_size = (10,20,20),
            D = (3,1,1),
            C = 1
        )

        probs = np.zeros((2,1,1,2,2),dtype=K.floatx())
        probs[1,0,0,0,1] = 1
        probs = K.constant(probs)

        actual_boxes, actual_probs = boxes_layer.call(probs)
        actual_boxes = K.eval(actual_boxes)
        actual_probs = K.eval(actual_probs)
        expected_boxes = np.zeros((2,2,6), dtype=K.floatx())
        #(0,0,0) * (3,1,1) + 1 = (1,1,1)
        #(1,1,1) -> (-4,6,-9,11,-9,11)
        expected_boxes[0,0] = np.array([-4,6,-9,11,-9,11],dtype=K.floatx())

        #(0,0,1) -> (1,1,2)
        #(1,1,2) -> (-4,6,-9,11,-8,12)
        expected_boxes[0,1] = np.array([-4,6,-9,11,-8,12],dtype=K.floatx())

        #the same
        expected_boxes[1] = expected_boxes[0,:,:] 
        expected_probs = np.zeros((2,2,2),dtype=K.floatx())
        expected_probs[1,0,1] = 1
        np.testing.assert_array_equal(actual_boxes,expected_boxes)
        np.testing.assert_array_equal(actual_probs,expected_probs)

       
        
class TestClipBoxes(object):
    def test_mini_batch(self):
        clip_layer = layers.ClipBoxes()
        boxes = np.array([
            [[-2,8,2,8,-4,6]],
            [[90,105,180,210,180,210]]
        ], dtype=K.floatx())
        shape = np.array([2,100,200,200,1])
        boxes = K.constant(boxes)
        shape = K.constant(shape)

        actual_boxes = clip_layer.call([boxes,shape])
        actual_boxes = K.eval(actual_boxes)

        expected_boxes = np.array([
            [[0,8,2,8,0,6]],
            [[90,100,180,200,180,200]]
        ],dtype=K.floatx())
        np.testing.assert_array_equal(actual_boxes, expected_boxes)

    def test_load_model(self):
        inputs = keras.layers.Input(shape=(4,10,10,2))

        boxes, probs = layers.Boxes(
            box_size = (2,2,2),
            D        = (1,1,1),
            C        = (0,0,0)
        )(inputs)
        image_shape = layers.Shape()(inputs)
        outputs = layers.ClipBoxes()([boxes, image_shape])
        model = keras.models.Model(inputs=inputs, outputs=outputs)
        model.summary()
        save_path = '../clip_box_model.h5'
#        keras.models.save_model(model, save_path)

#        with pytest.raises(Exception):#RecursionError, TypeError
        model.save(save_path)
        keras.models.load_model(save_path,custom_objects={
            'Boxes'     : layers.Boxes,
            'ClipBoxes' : layers.ClipBoxes,
            'Shape'     : layers.Shape
        })



class TestResizeBoxes(object):
    def test_resize_partial(self):
        target_size = np.array([10,30,30])
        resize_layer = layers.ResizeBoxes(target_size)
        boxes = np.array([
            [[0,2,0,2,0,2],
            [0,0,0,0,0,0]]
        ], dtype=K.floatx())
        labels = np.array([
            [1,-1]
        ])
        boxes = K.constant(boxes)
        labels = K.constant(labels)

        actual_boxes, actual_labels = resize_layer.call([boxes, labels])
        actual_boxes = K.eval(actual_boxes)
        actual_labels = K.eval(actual_labels)

        expected_boxes = np.array([[
            [-4,6,-14,16,-14,16],
            [0,0,0,0,0,0]
        ]],dtype=K.floatx())
        expected_labels = K.eval(labels)

        np.testing.assert_array_equal(actual_boxes, expected_boxes)
        np.testing.assert_array_equal(actual_labels,expected_labels)
        
    def test_resize_all_and_min_batch(self):
        target_size = np.array([10,30,30])
        resize_layer = layers.ResizeBoxes(target_size, mode='all')
        boxes = np.array([
            [[0,2,0,2,0,2]],
            [[0,0,0,0,0,0]]
        ], dtype=K.floatx())
        labels = np.array([
            [1],
            [-1]
        ])
        boxes = K.constant(boxes)
        labels = K.constant(labels)

        actual_boxes, actual_labels = resize_layer.call([boxes, labels])
        actual_boxes = K.eval(actual_boxes)
        actual_labels = K.eval(actual_labels)

        expected_boxes = np.array([
            [[-4,6,-14,16,-14,16]],
            [[-5,5,-15,15,-15,15]]
        ],dtype=K.floatx())
        expected_labels = K.eval(labels)

        np.testing.assert_array_equal(actual_boxes, expected_boxes)
        np.testing.assert_array_equal(actual_labels,expected_labels)

class TestRegressBoxes(object):
    def test_simple(self):
        mean = [0,0,0,0]
        std = [0.2,0.2,0.2,0.2]

        regress_boxes_layer = layers.RegressBoxes(mean=mean, std=std)

        boxes = np.array([[
            [0,10,0,10,0,10],
            [50,100,50,100,50,100],
            [50,100,50,100,50,100]
        ]], dtype=K.floatx())
        boxes = K.constant(boxes)

        regression = np.array([[
            [0,0,0,0],
            [0.1,0.1,0.1,0],
            [-0.1,-0.1,-0.1,0.1]
        ]], dtype=K.floatx())
        regression = K.constant(regression)
        actual_boxes = regress_boxes_layer.call([boxes,regression])
        actual_boxes = K.eval(actual_boxes)

        expb1 = tobbox([5,5,5],    np.sqrt(3) * 10)
        expb2 = tobbox([75.5,75.5,75.5], np.sqrt(3) * 50)
        expb3 = tobbox([74.5,74.5,74.5], np.exp(0.02) * np.sqrt(3) * 50)
        expected_boxes = np.array([[
            expb1,
            expb2,
            expb3
        ]], dtype=K.floatx())

        assert_almost_equal(actual_boxes, expected_boxes, decimal=6)

    def test_min_batch(self):

        mean = [0,0,0,0]
        std = [0.2,0.2,0.2,0.2]

        regress_boxes_layer = layers.RegressBoxes(mean=mean, std=std)
    
        boxes = np.array([
           [[0,10,0,10,0,10],
            [50,100,50,100,50,100],
            [50,100,50,100,50,100]],

           [[50,100,50,100,50,100],#3
            [0,10,0,10,0,10],#1
            [50,100,50,100,50,100]]#2
        ], dtype=K.floatx())
        boxes = K.constant(boxes)

        regression = np.array([
           [[0,0,0,0],
            [0.1,0.1,0.1,0],
            [-0.1,-0.1,-0.1,0.1]],
 
           [[-0.1,-0.1,-0.1,0.1], #3
            [0,0,0,0], #1
            [0.1,0.1,0.1,0]] #2
        ], dtype=K.floatx())
        regression = K.constant(regression)
        actual_boxes = regress_boxes_layer.call([boxes,regression])
        actual_boxes = K.eval(actual_boxes)

        expb1 = tobbox([5,5,5],    np.sqrt(3) * 10)
        expb2 = tobbox([75.5,75.5,75.5], np.sqrt(3) * 50)
        expb3 = tobbox([74.5,74.5,74.5], np.exp(0.02) * np.sqrt(3) * 50)
        expected_boxes = np.array([
           [expb1, expb2, expb3],

           [expb3, expb1, expb2]
        ], dtype=K.floatx())

        assert_almost_equal(actual_boxes, expected_boxes, decimal=6)


