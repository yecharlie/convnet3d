import argparse
import os
import sys

import keras
import tensorflow as tf
# print(__name__,'__package__:',__package__)

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import convnet3d.bin # noqa: F401
    __package__ = "convnet3d.bin"

from .. import models
from .. import losses
from ..preprocessing.generator import Generator
from ..preprocessing.val_generator import ValidationGenerator

from ..utils.transform import randomTransformGenerator
from ..callbacks import (RedirectModel, Evaluate)


def get_session():
    '''Construct a modified tf session
    '''
#    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_generators(args):
    if args.random_transform:
        transform_generator = randomTransformGenerator(
            min_scaling = (0.9, 0.9, 0.9),
            max_scaling = (1.1, 1.1, 1.1),
            min_horizontal_rotation = -0.1,
            max_horizontal_rotation = 0.1,
            flip_x_chance = 0.5,
            flip_y_chance = 0.5,
            min_translation = (-0.1, -0.1, -0.1),
            max_translation = (0.1, 0.1, 0.1)
        )
    else:
        transform_generator = None

    train_generator = Generator(
        args.annotations,
        args.classes,
        batch_size = args.batch_size,
        transform_generator=transform_generator
    )

    if args.val_annotations and getattr(args, 'val_cs_model', None):
        validation_generator = ValidationGenerator(
            args.val_annotations,
            args.classes
        )
    elif args.val_annotations:
        validation_generator = Generator(
            args.val_annotations,
            args.classes,
            batch_size = args.batch_size
        )
    else:
        validation_generator = None
    return train_generator, validation_generator
 

def create_models(num_classes, args):
    if args.model_type == 'cs':
        model = models.detectionModel(num_classes = num_classes, input_feature_size = args.data_channels)
        detections = model.outputs[0]
        reshaped = keras.layers.Reshape((num_classes,), name='classification' )(detections)
        training_model = keras.models.Model(inputs=model.inputs, outputs=reshaped)

        prediction_model = training_model

        training_model.compile(
            loss={'classification' : losses.detectionLossOHEM()},
            optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
        )
    elif args.model_type == 'fpr':
        if args.val_cs_weights:
            # initiate fpr model with cs model weights
            model = models.reductionModel1b(num_classes = num_classes, input_feature_size = args.data_channels, cs_model_path=args.val_cs_model)
        else:
            model = models.reductionModel1b(num_classes = num_classes, input_feature_size = args.data_channels)

        training_model = model
#        prediction_model = model
        if args.val_cs_model and args.val_annotations:
            cs_model = models.loadModel(args.val_cs_model)
            prediction_model = models.convnet3dModel1b(model, cs_model)
        else:
            prediction_model = model

        training_model.compile(
            loss={
                'classification' : keras.losses.sparse_categorical_crossentropy
            },
            optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
        )

    return model, training_model, prediction_model


def create_callbacks(model, training_model, prediction_model, validation_generator, args):
    callbacks = []

    if args.snapshots:
        os.makedirs(args.snapshot_path, exist_ok=True)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                args.snapshot_path,
                '{model_type}_{{epoch:02d}}.h5'.format(model_type=args.model_type)
            ),
            verbose=1
        )
        checkpoint = RedirectModel(checkpoint, model)
        callbacks.append(checkpoint)

    tensorboard_callback = None
    if args.tensorboard_dir:
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir = args.tensorboard_dir,
            histogram_freq         = 0,
            batch_size             = args.batch_size,
            write_graph            = True,
            write_grads            = False,
            write_images           = False,
            embeddings_freq        = 0,
            embeddings_layer_names = None,
            embeddings_metadata    = None
        )
        callbacks.append(tensorboard_callback)

    if args.val_annotations:
        if args.model_type == 'cs':
            evaluation = Evaluate(validation_generator, mode='recall', tensorboard=tensorboard_callback)
        elif args.val_cs_model:
            evaluation = Evaluate(
                validation_generator,
                mode='mAP',
                tensorboard=tensorboard_callback,
                window_size = (25, 60, 60),
                sliding_strides = (13, 30, 30),
                nms=True
            )
        else:
            evaluation = Evaluate(validation_generator, mode='accuracy', tensorboard=tensorboard_callback)

        evaluation = RedirectModel(evaluation, prediction_model)
        callbacks.append(evaluation)

    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor   = 'loss',
        factor    = 0.1,
        patience  = 2,
        verbose   = 1,
        mode      = 'auto',
        epsilon   = 0.0001,
        cooldown  = 0,
        min_lr    = 0
    ))
    return callbacks


def parse_args(args):
    '''Parse the arguments
    '''
    parser = argparse.ArgumentParser(description='Simple training script for training the candidate screening model & false positive reduction model.')
    subparsers = parser.add_subparsers(help='Specitic the model type: cs/fpr.', dest='model_type')
    subparsers.required = True
    cs_parser = subparsers.add_parser('cs') # noqa: F841
    fpr_parser = subparsers.add_parser('fpr')
    fpr_parser.add_argument('--val-cs-model', help='Path to candidate screening model, then the two model are combined to biuld a convnet3d model for validation.')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--snapshot')
    group.add_argument('--no-weights', dest='val_cs_weights', action='store_const', const=False)
    group.add_argument('--val-cs-weights', action='store_true')

    parser.add_argument('annotations')
    parser.add_argument('classes')
    parser.add_argument('--val-annotations')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)

    def devices(string):
        return string.split(',')

    parser.add_argument('--gpu', metavar='GPUs', type=devices, default=None)
    parser.add_argument('--snapshot-path', default='./snapshots')
    parser.add_argument('--tensorboard-dir', default='./logs')
    parser.add_argument('--no-snapshots',dest='snapshots', action='store_false')
    parser.add_argument('--random-transform', action='store_true')
    parser.add_argument('--data-channels', default=1, type=int)
    return check_args(parser.parse_args(args))


def check_args(parsed_args):
    if parsed_args.val_cs_weights:
        if parsed_args.model_type == 'cs':
            raise ValueError(
                '"--val-cs-model" is an option for fpr model only. ')
        elif not parsed_args.val_cs_model:
            raise ValueError(
                'Validation cs model has not benn set yet. (See "--val-cs-model")')

    return parsed_args


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu[0]

    keras.backend.tensorflow_backend.set_session(get_session())

    train_generator, validation_generator = create_generators(args)

    if args.snapshot is not None:
        raise NotImplementedError('Snapshot option  is unsupported now.')
    else:
        print('Creating model, this may take a second...')
        model, training_model, prediction_model = create_models(
            num_classes = train_generator.numClasses(),
            args = args
        )

    print(model.summary())

    callbacks = create_callbacks(
        model,
        training_model,
        prediction_model,
        validation_generator,
        args
    )


    #training
    training_model.fit_generator(
        generator=train_generator,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks
    )


if __name__ == '__main__':
    main()
