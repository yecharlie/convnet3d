'''
The defination of non-maximum-suppresion taked in this script is referred to:
https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
'''

import numpy as np

def nmsOverlaps(overlaps, scores, threshold):
    assert (np.diagonal(overlaps) == 1).all()
    assert 0 < threshold < 1

    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        inds = np.where(overlaps[i, order] <= threshold)[0]
        order = order[inds]

    return keep

