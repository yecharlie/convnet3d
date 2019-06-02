import os
import numpy as np

import SimpleITK as sitk

from .tobbox import tobbox


def readSeries(spath):
    '''Read Series, the arg should be a directory/a single dcm file/a mhd file.
    '''
    if os.path.isdir(spath):
        reader = sitk.ImageSeriesReader()
        dcmNames = reader.GetGDCMSeriesFileNames(spath)
        reader.SetFileNames(dcmNames)
        img = reader.Execute()  # x,y,z
    elif os.path.isfile(spath):
        img = sitk.ReadImage(spath)
    else:
        raise ValueError("invalid series path", spath)

    return img


def readImage(path, sides, *centroids, reader=readSeries, convert_centroids=True, verbose=False):
    '''Read image patch from series path.

    Args:
        path        : Series path.
        sides       : Respective side length, could be a single value(diameter) or a tuple (x,y,z).
        centroids   : Image patch 's centroids, zero-based.
        verbose     : Note that it should take the keyword form verbose=True  when set this option.
    Returns:
        arr_list    : List of image patch in form of ndarray with shape tuole(sides.reverse()) + (channels,).
        indices     : Indices list of valid patches.
        new_centroids : centroids corresponding to indices in the isotropic space (convert_centroids == true) or the original space
        isotropicSpace : function to convert point of this series to isotropic space.
    '''
    arr_list = []
    indices = []
    image = reader(path)
    resampled, transform = isotropic(image)
    resampled_size = np.array(resampled.GetSize())
    new_centroids = []
    
    def isotropicSpace(centroid):
        newc = image.TransformContinuousIndexToPhysicalPoint(centroid.astype(np.float64))
        newc = np.array(transform.GetInverse().TransformPoint(newc))
        return newc
        
    def isBboxValid(bbox, size):
        return np.all(bbox[::2] >= 0) and np.all(bbox[1::2] <= size ) 

    for idx, centroid in enumerate(centroids):
        # compute the new centroid in the resampled image
        if convert_centroids:
            newc = isotropicSpace(centroid)
        else:
            # centroid must have been converted
            newc = centroid

        x = tobbox(newc, sides).astype(int)
        if not isBboxValid(x, resampled_size):
            if verbose:
                print('sample {} with sides {} is out of image boundary {}.'.format(newc, sides, resampled_size))
            continue

        img_patch = resampled[x[0]:x[1], x[2]:x[3], x[4]:x[5]]
        arr = sitk.GetArrayFromImage(img_patch)

        if len(arr.shape) == 3:
            # add channel dim
            arr = arr.reshape(arr.shape+(1,))
        elif len(arr.shape) != 4:
            raise ValueError('Unexpected series data')
            
        arr_list.append(arr)
        new_centroids.append(newc)
        indices.append(idx)
    return arr_list, indices, new_centroids, isotropicSpace


def huwindowing(imgarr, level = 80, window = 600, outmin = 0, outmax=1):
    '''housefield unit windowing for CT images
    '''
    if window < 1:
        raise ValueError("window value should be not less than 1 but %.2f was received" % (window))

    imgarr = ((imgarr - level + 0.5 ) / (window - 1) + 0.5) \
        * (outmax - outmin) + outmin
    imgarr[imgarr <= outmin] = outmin
    imgarr[imgarr >= outmax] = outmax
    return imgarr


def isotropic(image):
    dim = image.GetDimension()
    ref = createRefDomain(image)

    # aligning use
    transform = sitk.AffineTransform(dim)
    transform.SetMatrix(image.GetDirection())
    transform.SetTranslation(np.array(image.GetOrigin()) - np.array(ref.GetOrigin()))

    resampled = sitk.Resample(image, ref, transform)
    return resampled, transform


def createRefDomain(image):
    '''Create a standard reference domain
    '''
    dim = image.GetDimension()
    phys_sz = [ (sz - 1) * sp for sz, sp in zip(image.GetSize(), image.GetSpacing())]

    ref_origin = np.zeros(dim)
    ref_direction = np.identity(dim).flatten()
    ref_spacing = np.ones(dim)
    ref_size = [int(psz / sp + 1) for psz, sp in zip(phys_sz, ref_spacing)]

    ref_image = sitk.Image(ref_size, image.GetPixelIDValue())
    ref_image.SetOrigin(ref_origin)
    ref_image.SetSpacing(ref_spacing)
    ref_image.SetDirection(ref_direction)
    return ref_image

    
def transformImage(
    imgarr, 
    matrix, 
    translation, 
    relativeTranslatiuon=True,  
    interpolator=sitk.sitkLinear,
    defaultPixelValue=0.0,
    outputPixelType=sitk.sitkUnknown,
):
    '''Apply affine transform on a image.

    This function require the input image has been placed on the standard domain.

    Args:
        image                : ndarray of (d, w, h, c)
        matrix               : linear part of affine transform.
        translation          : translation
        relativeTranslatiuon : a flag to indicate that the translation parameter is relative to image size
        interpolator         : See SimpleITK.Resample.
        defaultPixelValue    : See SimpleITK.Resample.
        outputPixelType      : See SimpleITK.Resample.
    Returns:
        transformed          : The transformed image, ndaary with same shap as input image.
        affine               : The AffineTransform object which has been applied on the image.
    '''
    depth, width, height, channels = imgarr.shape
    if channels == 1:
        imgarr = np.squeeze(imgarr, axis = 3)
        
    image = sitk.GetImageFromArray(imgarr)
    center = np.array([height // 2, width // 2, depth // 2])
    if relativeTranslatiuon:
        translation *= [height, width, depth]
    affine = sitk.AffineTransform(3)
    affine.SetMatrix(matrix.ravel())
    affine.SetTranslation(translation)

    # take proper precision
    affine.SetCenter(center.astype('float64'))

    # resample and convert it to array
    resampled = sitk.Resample(image, affine, interpolator, defaultPixelValue, outputPixelType)
    resampled = sitk.GetArrayFromImage(resampled)
    if len(resampled.shape) == 3:
        resampled = resampled.reshape(resampled.shape + (1,))
    elif len(resampled.shape) != 4:
        raise ValueError('Unexpected image data')
 
    return resampled, affine
