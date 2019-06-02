import numpy as np
from collections import Iterable


def tobbox(centroid, sides):
    if isinstance(sides, Iterable):
        assert len(sides) == len(centroid)
        sides = np.asarray(sides)
    centroid = np.asarray(centroid)

    xi = centroid - sides / 2
    xj = xi + sides
    x = np.zeros(len(xi) * 2)
    x[0::2] = xi
    x[1::2] = xj
    return x
