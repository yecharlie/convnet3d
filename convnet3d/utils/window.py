import numpy as np


def windowing(image_size, window_size, sliding_strides, mode='fill'):
    assert len(image_size) == len(window_size) == len(sliding_strides) == 3
    window_size = np.array(window_size)
    sliding_strides = np.array(sliding_strides)
    if mode == 'fill' and (window_size < sliding_strides).all():
        raise ValueError('invalid arguments with "fill" mode window_size {} < side_strides {}'.format(window_size, sliding_strides))

    ia = np.arange(0, image_size[0] - window_size[0], sliding_strides[0])
    ib = np.arange(0, image_size[1] - window_size[1], sliding_strides[1])
    ic = np.arange(0, image_size[2] - window_size[2], sliding_strides[2])

    if mode == 'fill':
        ia = np.append(ia, image_size[0] - window_size[0])
        ib = np.append(ib, image_size[1] - window_size[1])
        ic = np.append(ic, image_size[2] - window_size[2])

    gia, gib, gic = np.meshgrid(ia, ib, ic, indexing='ij')

    offsets = np.concatenate([
        np.expand_dims(gia, axis=-1),
        np.expand_dims(gib, axis=-1),
        np.expand_dims(gic, axis=-1)
    ], axis=-1)
    return offsets.reshape((-1, 3))
