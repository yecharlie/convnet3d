import numpy as np

DEFAULT_PRNG = np.random

# def transformBbox(matrix, translate, center, box):
#
#     '''
#     Note that in SimpleITK transform parameters are applied from output sapce to input space.
#         xi = A(Xo - C) + T + C
#     where A:linear transform matrix, C:center, T:translate, which implies:
#         xo = A^-1(xi - C - T) +  C
#     '''
#     x1,x2,y1,y2,z1,z2 = box
#     points = np.array([
#         [x1, x2, x1, x2, x1, x2, x1, x2],
#         [y1, y1, y2, y2, y1, y1, y2, y2],
#         [z1, z1, z1, z1, z2, z2, z2, z2]
#     ])
#     matrix = np.array(matrix.copy())
#     translate = np.array(translate)
#     center = np.array(center)
#
#     points -= np.expand_dims(translate + center, axis=-1)
#     inv_ma = np.linalg.inv(matrix)
#     points = inv_ma.dot(points)
#     points += np.expand_dims(center, axis=-1)
#
#     min_corner = points.min(axis=1)
#     max_corner = points.max(axis=1)
#     print('min_corner shape',min_corner.shape)
#
#     transformed = np.zeros(6)
#     transformed[::2] = min_corner[:]
#     transformed[1::2] = max_corner[:]
#     return transformed


def transformBbox(box, transform):
    '''
    Note that in SimpleITK transform parameters are applied from output sapce to input space.
        xi = A(Xo - C) + T + C
    where A:linear transform matrix, C:center, T:translate, which implies:
        xo = A^-1(xi - C - T) +  C
    '''
    x1, x2, y1, y2, z1, z2 = box
    points = np.array([
        [x1, x2, x1, x2, x1, x2, x1, x2],
        [y1, y1, y2, y2, y1, y1, y2, y2],
        [z1, z1, z1, z1, z2, z2, z2, z2]
    ], dtype=np.float64)  # double for TransformPoint
    inverse = transform.GetInverse()
    for i in range(points.shape[1]):
        points[:, i] = np.array(inverse.TransformPoint(points[:, i]))

#    matrix = np.array(matrix.copy())
#    translate = np.array(translate)
#    center = np.array(center)
#
#    points -= np.expand_dims(translate + center, axis=-1)
#    inv_ma = np.linalg.inv(matrix)
#    points = inv_ma.dot(points)
#    points += np.expand_dims(center, axis=-1)

    min_corner = points.min(axis=1)
    max_corner = points.max(axis=1)
#    print('min_corner shape',min_corner.shape)

    transformed = np.zeros(6)
    transformed[::2] = min_corner[:]
    transformed[1::2] = max_corner[:]
    return transformed


def _randomVector(min, max, prng=DEFAULT_PRNG):
    min = np.array(min)
    max = np.array(max)
    assert min.shape == max.shape
    assert len(min.shape) == 1
    return prng.uniform(min, max)


def randomTranslation(min, max, prng=DEFAULT_PRNG):
    return _randomVector(min, max, prng)


def scaling(factor):
    return np.array([
        [factor[0], 0,  0],
        [0, factor[1],  0],
        [0, 0,  factor[2]]
    ])


def randomScaling(min, max, prng=DEFAULT_PRNG):
    return scaling(_randomVector(min, max, prng))


def horizontalRotation(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle),  0],
        [0, 0, 1]
    ])


def randomHorRotation(min, max, prng=DEFAULT_PRNG):
    return horizontalRotation(prng.uniform(min, max))


def randomFlip(flip_x_chance, flip_y_chance, prng=DEFAULT_PRNG):
    flip_x = prng.uniform(0, 1) < flip_x_chance
    flip_y = prng.uniform(0, 1) < flip_y_chance
    '''
        1 - 2 * flip_x  == -1 if flip_x else 1
        1 - 2 * flip_y ...
        1 - 2 * (flip_x ^ flip_y)) ...
    '''
    return scaling((1 - 2 * flip_x, 1 - 2 * flip_y, -2 * (flip_x ^ flip_y) + 1))


def randomTransform(
    min_scaling=(1, 1, 1),
    max_scaling=(1, 1, 1),
    min_horizontal_rotation=0,
    max_horizontal_rotation=0,
    flip_x_chance=0,
    flip_y_chance=0,
    min_translation=(0, 0, 0),
    max_translation=(0, 0, 0),
    prng=DEFAULT_PRNG
):
    linear = np.linalg.multi_dot([
        randomScaling(min_scaling, max_scaling, prng),
        randomHorRotation(min_horizontal_rotation, max_horizontal_rotation, prng),
        randomFlip(flip_x_chance, flip_y_chance, prng)
    ])
    translation = randomTranslation(min_translation, max_translation)
    return linear, translation


def randomTransformGenerator(
    prng=None,
    min_scaling=(1, 1, 1),
    max_scaling=(1, 1, 1),
    min_horizontal_rotation=0,
    max_horizontal_rotation=0,
    flip_x_chance=0,
    flip_y_chance=0,
    min_translation=(0, 0, 0),
    max_translation=(0, 0, 0),
):
    if prng is None:
        prng = np.random.RandomState()

    # adjust params for transform
    # Internally, parameters are set for mapping from output space to input sapce.
    min_scaling = np.array(min_scaling)
    max_scaling = np.array(max_scaling)
    min_scaling_inv = np.min([1 / min_scaling, 1 / max_scaling], axis=0)
    max_scaling_inv = np.max([1 / min_scaling, 1 / max_scaling], axis=0)

    min_rotation_inv = np.min([-min_horizontal_rotation, -max_horizontal_rotation], axis=0)
    max_rotation_inv = np.max([-min_horizontal_rotation, -max_horizontal_rotation], axis=0)

    # Note that the flip parameters are equivalent (A = ~A)

    min_translation = np.array(min_translation)
    max_translation = np.array(max_translation)
    min_translation_inv = np.min([-min_translation, -max_translation], axis=0)
    max_translation_inv = np.max([-min_translation, -max_translation], axis=0)

    while True:
        yield randomTransform(
            min_scaling = min_scaling_inv,
            max_scaling = max_scaling_inv,
            min_horizontal_rotation = min_rotation_inv,
            max_horizontal_rotation = max_rotation_inv,
            flip_x_chance = flip_x_chance,
            flip_y_chance = flip_y_chance,
            min_translation = min_translation_inv,
            max_translation = max_translation_inv,
            prng = prng
        )
