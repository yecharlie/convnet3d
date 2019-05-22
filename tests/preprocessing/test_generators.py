from convnet3d.preprocessing.reduction_generator import ReductionGenerator

def test_reduction_generator():
    csv_data_file = '/mnt/aneurysm-kfoldCV/3d/KFoldsCV_patches0_reduction_train'
    mapping = '/mnt/aneurysm-kfoldCV/3d/mapping.csv'
    generator = ReductionGenerator(csv_data_file, mapping, batch_size=32)
    for i in range(len(generator)):
        inputs, targets = generator[i]
