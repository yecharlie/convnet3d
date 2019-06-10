# import convnet3d.bin.train
# import keras.backend as K
#
# import pytest
#
# @pytest.fixture(autouse=True)
# def clear_session():
#     #run before test
#     yield
#
#     #run after test, clear keras session
#     K.clear_session()
#
# def test_windowing():
#     convnet3d.bin.train.main([
#         '--epochs=1',
#         '--no-snapshots',
#         '--val-annotations=/mnt/aneurysm-kfoldCV/3d/KFoldsCV_series0_test',
#         '--gpu=1',
#         'fpr',
#         '--val-cs-model=/mnt/aneurysm-kfoldCV/3d/snapshots/CS0/cs_31.h5',
#         '/mnt/aneurysm-kfoldCV/3d/KFoldsCV_patches0_reduction_train',
#         '/mnt/aneurysm-kfoldCV/3d/mapping.csv'
#     ])
