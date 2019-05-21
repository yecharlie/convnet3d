import os

if 'KERAS_BACKEND' in os.environ:
    _backend = os.environ('KERAS_BACKEND')

    assert _backend == 'tensorflow'

from .tensorflow_backend  import *
from .common import *

