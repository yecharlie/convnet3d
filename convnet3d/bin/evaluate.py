import argparse
import os
import sys

import keras
import tensorflow as tf

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..','..',))
    import convnet3d.bin
    __package__ = "convnet3d.bin"

from .. import models
from ..preprocessing.val_generator import ValidationGenerator
#from ..utils.eval import evaluate
from ..callbacks.eval import eval_mAP

def get_session():
    '''Construct a modified tf session
    '''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def create_generator(args):
#    #preporcess for lung nodule (temporary)
#    from ..utils.image import huwindowing
#    def windowing(image, annotations):
#        x = image
#        x = x.astype(keras.backend.floatx())
#        x = huwindowing(x, level=-600, window=1200, outmin = 0, outmax = 255)
#        return x, annotations

    validation_generator = ValidationGenerator(
        args.val_annotations,
        args.classes,
#        preprocessImage=windowing,
        batch_size = 1
    )
    return validation_generator

def parse_args(args):
    '''Parse the arguments
    '''
    parser = argparse.ArgumentParser(description='Evaluation script for convnet3d model.')
    parser.add_argument('cs_model')
    parser.add_argument('fpr_model')

    parser.add_argument('val_annotations')
    parser.add_argument('classes')

    def args_list(string):
        args = string.split(',')
        return [int(a) for a in args]

    parser.add_argument('--window-size', type=args_list, default=(25,60,60))
    parser.add_argument('--sliding-strides',type=args_list, default=[13,30,30])

    parser.add_argument('--gpu')
 
    return parser.parse_args(args)

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES']  = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    generator = create_generator(args)

    print('loading model, this may take a second.')
    cs_model  = models.loadModel(args.cs_model)
    fpr_model = models.loadModel(args.fpr_model) 
    convnet3d1b = models.convnet3dModel1b(fpr_model, cs_model)

    mean_ap, descp = eval_mAP(
        generator,
        convnet3d1b,
        window_size = args.window_size,
        sliding_strides = args.sliding_strides,
        nms=True
    )

    print('{}: {:.4f}'.format('mAP', mean_ap), descp)


if __name__ == '__main__':
    main()
 



