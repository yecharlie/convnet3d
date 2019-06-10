import pytest
from numpy.testing import assert_allclose

from convnet3d.utils.index_mapping import IndexMap
from convnet3d.models import (
    detectionModel,
    reductionModel
)


def test_init():
    with pytest.raises(ValueError) as einfo:
        model = reductionModel()
        imap = IndexMap(model)  # noqa: F841
    assert 'Multi inputs/outputs' in str(einfo.value)
# After modification, the error has been eliminated.
#     with  pytest.raises(ValueError) as einfo:
#         model = reductionModel()
#         imap = IndexMap(model)
#     assert 'illegal configuration' in str(einfo.value)


def test_mapping():
    model = detectionModel()
    imap = IndexMap(model)
    assert imap.D == pytest.approx([1, 2, 2])
    assert imap.C == pytest.approx([6, 14, 14])

    indices = [[1, 1, 1], [10, 10, 10]]
    mapped = [[7, 16, 16], [16, 34, 34]]

    assert_allclose(imap(indices), mapped)
# approx is not for nested array
#    assert imap(indices) == pytest.approx(mappped)
