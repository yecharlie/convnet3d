import tensorflow as tf


def _is_tensor(x):
    """Returns `True` if `x` is a symbolic tensor-like object.
 
    From http://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/image_ops_impl.py
 
    Args:
        x: A python object to check.
 
    Returns:
        `True` if `x` is a `tf.Tensor` or `tf.Variable`, otherwise `False`.
    """
  return isinstance(x, (tf.Tensor, tf.Variable))


def _ImageDimensions(image, rank):
    """Returns the dimensions of an image tensor.
  
    From http://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/image_ops_impl.py
  
    Args:
        image: A rank-D Tensor. For 3-D  of shape: `[height, width, channels]`.
        rank: The expected rank of the image
  
    Returns:
        A list of corresponding to the dimensions of the
        input image.  Dimensions that are statically known are python integers,
        otherwise they are integer scalar tensors.
    """
    if image.get_shape().is_fully_defined():
        return image.get_shape().as_list()
    else:
        static_shape = image.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(image), rank)
        return [
            s if s is not None else d for s, d in zip(static_shape, dynamic_shape)
        ]


def _CheckAtLeast4DImage(image, require_static=True):
    """Assert that we are working with properly shaped image.

    (modified) From http://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/image_ops_impl.py

    Args:
        image: >= 4-D Tensor of size [*, height, width, depth, channels]
        require_static: If `True`, requires that all dimensions of `image` are
          known and non-zero.

    Raises:
        ValueError: if image.shape is not a [>= 4] vector.

    Returns:
        An empty list, if `image` has fully defined dimensions. Otherwise, a list
        containing an assert op is returned.
    """
    try:
        if image.get_shape().ndims is None:
            image_shape = image.get_shape().with_rank(4)
        else:
            image_shape = image.get_shape().with_rank_at_least(4)
    except ValueError:
        raise ValueError("'image' must be at least four-dimensional.")
    if require_static and not image_shape.is_fully_defined():
        raise ValueError('\'image\' must be fully defined.')
    if any(x == 0 for x in image_shape):
        raise ValueError(
            'all dims of \'image.shape\' must be > 0: %s' % image_shape)
    if not image_shape.is_fully_defined():
        return [
            tf.assert_positive(
              tf.shape(image),
              ["all dims of 'image.shape' "
               'must be > 0.'])
        ]
    else:
        return []


def uniform(*args, **kwargs):
    return tf.random.uniform(*args, **kwargs)


def pad(*args, **kwargs):
    return tf.pad(*args, **kwargs)


def top_k(*args, **kwargs):
    return tf.math.top_k(*args, **kwargs)


def non_max_suppression_overlaps(*args, **kwargs):
    return tf.image.non_max_suppression_overlaps(*args, **kwargs)


def gather_nd(*args, **kwargs):
    return tf.gather_nd(*args, **kwargs)


def clip_by_value(*args, **kwargs):
    return tf.clip_by_value(*args, **kwargs)


def meshgrid(*args, **kwargs):
    return tf.meshgrid(*args, **kwargs)


def map_fn(*args, **kwargs):
    return tf.map_fn(*args, **kwargs)


def where(*args, **kwargs):
    return tf.where(*args, **kwargs)


def crop_to_bounding_box_3d(image, box, target_size):
    '''Crops an 3d image to a specificed bounding box. When the size of box is smaller than 'target_size', then the surroundings of image is evenly (approximately) padded with zero. The 'box' with size = 0 is allowed.

    Args:
        image: 5-D Tensor of shape '[batch, heigh, width, depth, channels]' or
               4-D Tensor of shape '[heights, width, depth, channels]'
        box:  1-D Tensor of shape '[6,]' representing the cropped area.
        target_size: The ultimate bounding box size.

    Returns:
        if 'image' was 5-D, a 5-D float Tensor of shape '[batch_size] + target_size + [channels]'
        if 'image' was 4-D, a 5-D float Tensor of shape 'target_size + [channels]'
    '''

    with tf.name_scope(None, 'crop_to_bounding_box_3d', [image]):
        image = tf.convert_to_tensor(image, name='image')

        is_batch = True
        image_shape = image.get_shape()
        if image_shape.ndims == 4:
            is_batch = False
            image = tf.expand_dims(image, 0)
        elif image_shape.ndims is None:
            is_batch = False
            image = tf.expand_dims(image, 0)
            image.set_shape([None] * 5)
        elif image_shape.ndims != 5:
            raise ValueError('\'image\' must have either 4 or 5 dimensions.')

        assert_ops = _CheckAtLeast4DImage(image, require_static=False)

        # Never mind what are the real meaning of  height/width/depth. They are mimics from the tensorflow API 's writting convention.
        batch, height, width, depth, channels = _ImageDimensions(image, rank=5)
#        print('crop_to_bounding_box_3d height:',height)
        box_size = box[1::2] - box[::2]
        assert_ops.append(tf.assert_greater_equal([height, width, depth], box[1::2], ['The remote corner of box must not exceed image boundaries.']))
        assert_ops.append(tf.assert_non_negative(box[::2], ['The near corner of box must be non negative.']))
        assert_ops.append(tf.assert_non_negative(box_size, ['The box size should be non negative.']))
        assert_ops.append(tf.assert_greater_equal(target_size, box_size, ['The target size should be not less than box size. ']))

        with tf.control_dependencies(assert_ops):
            image = image
#        tf.with_dependencies(assert_ops, image)

        cropped = tf.slice(
            image, tf.stack([0, box[0], box[2], box[4], 0]),
            tf.stack([-1, box_size[0], box_size[1], box_size[2] , -1])
        )

        def _max(x, y):
            if _is_tensor(x) or _is_tensor(y):
                return tf.maximum(x, y)
            else:
                return max(x, y)

        padding_offsets = _max((target_size - box_size) // 2, 0)
        after_padding_size = target_size - padding_offsets - box_size
        paddings = tf.reshape(
            tf.stack([
                0, 0, padding_offsets[0], after_padding_size[0],
                      padding_offsets[1], after_padding_size[1], # noqa: E131
                      padding_offsets[2], after_padding_size[2], 0, 0 # noqa: E131
        ]), [5, 2])
        padded = tf.pad(cropped, paddings)

        result_shape = [
            None if _is_tensor(i) else i
            for i in [batch, target_size[0], target_size[1], target_size[2], channels]
        ]
        padded.set_shape(result_shape)

        if not is_batch:
            padded = tf.squeeze(padded, axis=[0])

        return padded
