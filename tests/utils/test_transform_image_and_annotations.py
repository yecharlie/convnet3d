import os
import pytest
import numpy as np
import SimpleITK as sitk
from convnet3d.utils.image import transformImage
from convnet3d.utils.transform import (transformBbox, randomTransformGenerator)
from convnet3d.utils.tobbox import tobbox

SAVE_DIR = 'tests/utils/transformed_image/'
@pytest.fixture
def simple_image_patch():
    path = '/mnt/aneurysm-kfoldCV/demo/patches/reg_sample0211.npy'
    arr = np.load(path)
    c = 36.645,36.645,14.5
    d = 11.647
    bbox = tobbox(c,d)
    return arr,bbox

def test_flip(simple_image_patch):
    img, bbox = simple_image_patch
    linear, translation = next(randomTransformGenerator(flip_x_chance=1))
    transformed,affine = transformImage(img, linear, translation)
    bbox_t = transformBbox(bbox, affine)

    transformed = sitk.GetImageFromArray(transformed)
    svpth = os.path.join(SAVE_DIR, 'flipx.mhd')
    sitk.WriteImage(transformed, svpth)
    print('Transform type={}, params={}, bbox={}, bbox_t={}'.format('flip-x',1 , bbox, bbox_t))

def test_horizontal_rotation(simple_image_patch):
    from math import pi
    img, bbox = simple_image_patch
    degree = 45
    linear, translation = next(randomTransformGenerator(min_horizontal_rotation= pi * degree / 180, max_horizontal_rotation= pi * degree / 180 ))
    transformed,affine = transformImage(img,linear, translation)
    bbox_t = transformBbox(bbox, affine)

    transformed = sitk.GetImageFromArray(transformed)
    svpth = os.path.join(SAVE_DIR, 'rotation.mhd')
    sitk.WriteImage(transformed, svpth)
    print('Transform type={}, params={}, bbox={}, bbox_t={}'.format('rotation','45 degree' , bbox, bbox_t))

def test_scaling(simple_image_patch):
    img, bbox = simple_image_patch
    linear, translation = next(randomTransformGenerator(min_scaling=(2,2,1), max_scaling=(2,2,1)))

    transformed,affine = transformImage(img,linear, translation)
    bbox_t = transformBbox(bbox, affine)

    transformed = sitk.GetImageFromArray(transformed)
    svpth = os.path.join(SAVE_DIR, 'scaling.mhd')
    sitk.WriteImage(transformed, svpth)
    print('Transform type={}, params={}, bbox={}, bbox_t={}'.format('scaling', (2,2,1), bbox, bbox_t))

def test_translation(simple_image_patch):
    img, bbox = simple_image_patch
    linear, translation = next(randomTransformGenerator(min_translation=(0,0,5), max_translation=(0,0,5)))

    transformed,affine = transformImage(img,linear, translation, relativeTranslatiuon=False)
    bbox_t = transformBbox(bbox, affine)

    transformed = sitk.GetImageFromArray(transformed)
    svpth = os.path.join(SAVE_DIR, 'translation.mhd')
    sitk.WriteImage(transformed, svpth)

    print('Transform type={}, params={}, bbox={}, bbox_t={}'.format('Translation', (0,0,5), bbox, bbox_t))
    print('linear ={}, translation={}'.format(linear, translation))

#    print('Transform type={}, params={}, bbox={}, bbox_t={}'.format())



