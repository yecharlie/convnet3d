from convnet3d.utils.transform import (
    randomTranslation,
    scaling,
    randomScaling,
    horizontalRotation,
    randomHorRotation,
    randomFlip,
    transformBbox,
    randomTransform,
    randomTransformGenerator
)

import SimpleITK as sitk
import numpy as np
from numpy.testing import assert_almost_equal
from math import pi


def colvec(*args):
    return np.array([args]).T


def assrt_is_translation(translation, min, max):
    assert np.greater_equal(translation, min).all()
    assert np.less(translation, max).all()


def test_random_translation():
    prng = np.random.RandomState(0)
    min = (-10, 10)
    max = (10, 30)
    for i in range(100):
        assrt_is_translation(randomTranslation(min, max, prng), min, max)


def test_horizontal_rotation():
    assert_almost_equal(colvec( 1,  0, 1),horizontalRotation(0.0 * pi).dot(colvec(1, 0, 1)))
    assert_almost_equal(colvec( 0,  1, 1),horizontalRotation(0.5 * pi).dot(colvec(1, 0, 1)))
    assert_almost_equal(colvec(-1,  0, 1),horizontalRotation(1.0 * pi).dot(colvec(1, 0, 1)))
    assert_almost_equal(colvec( 0, -1, 1),horizontalRotation(1.5 * pi).dot(colvec(1, 0, 1)))
    assert_almost_equal(colvec( 1,  0, 1),horizontalRotation(2.0 * pi).dot(colvec(1, 0, 1)))

    assert_almost_equal(colvec( 0,  1, 1),horizontalRotation(0.0 * pi).dot(colvec(0, 1, 1)))
    assert_almost_equal(colvec(-1,  0, 1),horizontalRotation(0.5 * pi).dot(colvec(0, 1, 1)))
    assert_almost_equal(colvec( 0, -1, 1),horizontalRotation(1.0 * pi).dot(colvec(0, 1, 1)))
    assert_almost_equal(colvec( 1,  0, 1),horizontalRotation(1.5 * pi).dot(colvec(0, 1, 1)))
    assert_almost_equal(colvec( 0, 1, 1), horizontalRotation(2.0 * pi).dot(colvec(0, 1, 1)))

def test_random_hor_rotation():
    prng = np.random.RandomState(0)
    for i in range(100):
        assert_almost_equal(1, np.linalg.det(randomHorRotation(-i, i, prng)))

def test_scacling():
    assert_almost_equal(colvec(1, 2, 1), scaling((1.0, 1.0, 1.0)).dot(colvec(1, 2, 1)))
    assert_almost_equal(colvec(2, 2, 1), scaling((2.0, 1.0, 1.0)).dot(colvec(1, 2, 1)))
    assert_almost_equal(colvec(1, 1, 1), scaling((1.0, 0.5, 1.0)).dot(colvec(1, 2, 1)))
    assert_almost_equal(colvec(1, 2, 3), scaling((1.0, 1.0, 3.0)).dot(colvec(1, 2, 1)))


def assert_is_scaling(transform, min, max):
    assert transform.shape == (3, 3)
    assert_almost_equal(transform, transform.T)
    assert_almost_equal(np.diagonal(transform, 1), 0)
    assert_almost_equal(np.diagonal(transform, 2), 0)
    assert np.greater_equal(np.diagonal(transform), min).all()
    assert np.less(np.diagonal(transform), max).all()


def test_random_scaling():
    prng = np.random.RandomState(0)
    min = (0.1, 0.1, 0.2)
    max = (10, 30, 20)
    for i in range(100):
        assert_is_scaling(randomScaling(min, max, prng), min, max)


def test_random_flip():
    assert_almost_equal(randomFlip(0, 0).dot(colvec(1, 1, 1)), colvec(1, 1, 1))
    assert_almost_equal(randomFlip(1, 0).dot(colvec(1, 1, 1)), colvec(-1, 1, -1))
    assert_almost_equal(randomFlip(0, 1).dot(colvec(1, 1, 1)), colvec(1, -1, -1))
    assert_almost_equal(randomFlip(1, 1).dot(colvec(1, 1, 1)), colvec(-1, -1, 1))


def box_to_point(box):
    x1,x2,y1,y2,z1,z2 = box
    points = np.array([
        [x1, x2, x1, x2, x1, x2, x1, x2],
        [y1, y1, y2, y2, y1, y1, y2, y2],
        [z1, z1, z1, z1, z2, z2, z2, z2]
    ])
    return points


def test_transform_bbox():
    box = [0, 10, 0, 10, 0, 10]
    center = (2, 2, 2)
    matrix = randomFlip(1, 1).astype(np.float64)
    translation = [3, 3, 3]

    affine = sitk.AffineTransform(3)
    affine.SetMatrix(matrix.ravel())
    affine.SetTranslation(translation)
    affine.SetCenter(center)

    points_actual = box_to_point(transformBbox(box, affine))

#    affine = sitk.AffineTransform(3)
#    affine_mattrix = np.array(affine.GetMatrix()).reshape((3,3))
#    affine_mattrix[:,:] = matrix.copy()
#    affine.SetMatrix(affine_mattrix.ravel())
#    affine.SetTranslation(translation)
#    affine.SetCenter(center)
#    affine_inverse = affine.GetInverseTransform() #NOT FOUND!
#    points_expected= affine_inverse.TransformPoint(points)

    my_expected = box_to_point([-3, 7, -3, 7, -3, 7])
#    assert_almost_equal(points_expected, my_expected)

    assert_almost_equal(points_actual, my_expected)


def test_random_transform():
    prng = np.random.RandomState(0)
    for i in range(100):
        linear, translation = randomTransform(prng=prng)
        assert np.array_equal(linear, np.identity(3))
        assert np.array_equal(translation, np.zeros(3))

    for i, (linear, translation) in zip(range(100), randomTransformGenerator(prng=np.random.RandomState())):
        assert np.array_equal(linear, np.identity(3))
        assert np.array_equal(translation, np.zeros(3))
