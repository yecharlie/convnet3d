import keras.backend as K
from .. import backend

def tobbox(centroids,sides):
    '''To Bonding box

    Convert a cube coordinates to the form like [x1,x2,y1,y2,z1,z2]

    Args:
        centroids   : Tensor, the centroids of cube with shape (N,3).
        sides       : Respective side length, could be a int value or a tuple.
    Returns:
        x           : tensor of (N, 6), one could get a patch from array[x[0]:x[1],x[2]:x[3],x[4]:x[5]].
    '''
 
    xi = centroids - sides / 2
    xj = xi + sides
    x = K.stack((xi,xj), axis = -1)
    boxes = K.reshape(x, (-1,6))
    return boxes

def computeOverlaps(boxes, query_boxes):
    '''Compute the IoU overlaps 

    Args:
        boxes       : Tensor of shape (N, 6) of boxes where each box is (a1,a2,b1,b2,c1,c2).  
        query_boxes  : Tensor of shape (K, 6) of boxes

    Returns:
        overlaps of shape (N, K)
    '''
#    print('overlaps-boxes',boxes)
    b_volumes = (boxes[:,1] - boxes[:,0]) * (boxes[:,3] - boxes[:,2]) * (boxes[:,5] - boxes[:,4])
    ia = K.minimum(K.expand_dims(boxes[:,1], axis=1),query_boxes[:,1]) - K.maximum(K.expand_dims(boxes[:,0], axis=1),query_boxes[:,0])
    ib = K.minimum(K.expand_dims(boxes[:,3], axis=1),query_boxes[:,3]) - K.maximum(K.expand_dims(boxes[:,2], axis=1),query_boxes[:,2])
    ic = K.minimum(K.expand_dims(boxes[:,5], axis=1),query_boxes[:,5]) - K.maximum(K.expand_dims(boxes[:,4], axis=1),query_boxes[:,4])

    ia = K.maximum(ia,0)
    ib = K.maximum(ib,0)
    ic = K.maximum(ic,0)
    intersection_volumes = ia * ib * ic

    qb_volumes = (query_boxes[:,1] - query_boxes[:,0]) * (query_boxes[:,3] - query_boxes[:,2]) * (query_boxes[:,5] - query_boxes[:,4])
    union_volumes = K.expand_dims(qb_volumes, axis=1) + b_volumes - intersection_volumes
    union_volumes = K.maximum(union_volumes,K.epsilon())

    return intersection_volumes / union_volumes

def bboxTransformInv_v2(boxes, deltas, mean=None, std=None):
    """ Applies deltas (usually regression results) to boxes (usually anchors).

    Before applying the deltas to the boxes, the normalization that was previously applied (in the generator) has to be removed.
    The mean and std are the mean and std as applied in the generator. They are unnormalized in this function and then applied to the boxes.

    Args
        boxes : np.array of shape (B, N, 6), where B is the batch size, N the number of boxes and 6 values for (x1, x2, y1, y2, z1, z2).
        deltas: np.array of shape (B, N, 4). These deltas (d_x, d_y, d_z, d_d) are a factor of the height/width/depth and aneurysm diameter.
        mean  : The mean value used when computing deltas (defaults to [0, 0, 0, 0).
        std   : The standard deviation used when computing deltas (defaults to [0.2, 0.2, 0.2, 0.2]).

    Returns
        A np.array of the same shape as boxes, but with deltas applied to each box.
        The mean and std are used during training to normalize the regression values (networks love normalization).
    """

    if mean is None:
        mean = [0, 0, 0, 0]
    if std is None:
        std = [0.2, 0.2, 0.2, 0.2]

    #Note the reversed coordinates
    centroidz  =( boxes[:, :, 1] + boxes[:, :, 0]) / 2
    centroidy  =( boxes[:, :, 3] + boxes[:, :, 2]) / 2
    centroidx  =( boxes[:, :, 5] + boxes[:, :, 4]) / 2
    depth  = boxes[:, :, 1] - boxes[:, :, 0]
    width  = boxes[:, :, 3] - boxes[:, :, 2]
    height = boxes[:, :, 5] - boxes[:, :, 4]
    diagonal = K.sqrt(K.square(height) + K.square(width) + K.square(depth))
    
    #Note mean/std is the same as it is in generator.
    predx = centroidx + (deltas[:, :, 0] * std[0] + mean[0]) * height /2
    predy = centroidy + (deltas[:, :, 1] * std[1] + mean[1]) * width /2
    predz = centroidz + (deltas[:, :, 2] * std[2] + mean[2]) * depth /2
    predd =  K.exp(deltas[:, :, 3] * std[3] + mean[3]) * diagonal 

    def wrapped_tobbox(args):
        return tobbox(centroids=args[0], sides=args[1])
        
    pred_centroids = K.concatenate([
        K.expand_dims(predx, axis=-1),
        K.expand_dims(predy, axis=-1),
        K.expand_dims(predz, axis=-1),
    ],axis=-1)
    pred_centroids = K.reshape(pred_centroids,(-1, 1, 3))
    pred_sides = K.reshape(predd, (-1,))
    pred_boxes = backend.map_fn(
        wrapped_tobbox,
        elems = [pred_centroids, pred_sides],
        dtype = K.floatx(),
        parallel_iterations = 32
    )
    pred_boxes = K.reshape(pred_boxes, K.shape(boxes))
    return pred_boxes


def bboxTransformInv(boxes, deltas, mean=None, std=None):
    """ Applies deltas (usually regression results) to boxes (usually anchors).

    (modified) From https://github.com/fizyr/keras-retinanet/blob/master/keras_retinanet/backend/common.py
    Before applying the deltas to the boxes, the normalization that was previously applied (in the generator) has to be removed.
    The mean and std are the mean and std as applied in the generator. They are unnormalized in this function and then applied to the boxes.

    Args
        boxes : np.array of shape (B, N, 6), where B is the batch size, N the number of boxes and 6 values for (x1, x2, y1, y2, z1, z2).
        deltas: np.array of same shape as boxes. These deltas (d_x1, d_x2, d_y1, d_y2, d_z1, d_z2) are a factor of the height/width/depth.
        mean  : The mean value used when computing deltas (defaults to [0, 0, 0, 0, 0, 0]).
        std   : The standard deviation used when computing deltas (defaults to [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]).

    Returns
        A np.array of the same shape as boxes, but with deltas applied to each box.
        The mean and std are used during training to normalize the regression values (networks love normalization).
    """
    if mean is None:
        mean = [0, 0, 0, 0, 0 ,0]
    if std is None:
        std = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]

    height = boxes[:, :, 1] - boxes[:, :, 0]
    width  = boxes[:, :, 3] - boxes[:, :, 2]
    depth  = boxes[:, :, 5] - boxes[:, :, 4]


    x1 = boxes[:, :, 0] + (deltas[:, :, 0] * std[0] + mean[0]) * height
    x2 = boxes[:, :, 1] + (deltas[:, :, 1] * std[1] + mean[1]) * height
    y1 = boxes[:, :, 2] + (deltas[:, :, 2] * std[2] + mean[2]) * width
    y2 = boxes[:, :, 3] + (deltas[:, :, 3] * std[3] + mean[3]) * width
    z1 = boxes[:, :, 4] + (deltas[:, :, 4] * std[4] + mean[4]) * depth
    z2 = boxes[:, :, 5] + (deltas[:, :, 5] * std[5] + mean[5]) * depth

    pred_boxes = K.stack([x1, x2, y1, y2, z1, z2], axis=2)

    return pred_boxes
