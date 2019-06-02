import keras
import numpy as np
from ..utils.eval import evaluate


def _getResults(generator, model):
    all_annotations = np.zeros((generator.size(),))
    all_detections  = np.zeros((generator.size(),))
    for i in range(generator.size()):
        annotations = generator.loadAnnotations(i)
        all_annotations[i] = annotations['labels']
        raw_image = generator.loadImage(i)
        image, annotations = generator.preprocessGroupEntry(raw_image.copy(), annotations)

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((3, 0, 1, 2))
        predictions = model.predict_on_batch(np.expand_dims(image, axis=0))
        all_detections[i] = predictions.argmax(axis=1)[0]
    return all_detections, all_annotations

   
def eval_accuracy(generator, model):
    def _accuracy(all_detections, all_annotations):
        T  = np.count_nonzero(all_detections = all_annotations)
        P  = np.count_nonzero(all_annotations != 0)
        descp = '{:.0f} positives out of {:.0f} samples'.format(P, all_annotations.size),

        return T / all_annotations.size, descp

    all_detections, all_annotations = _getResults(generator, model)
    return _accuracy(all_detections, all_annotations)


def eval_recall(generator, model):
    def _recall(all_detections, all_annotations):
        TP = np.count_nonzero(np.logical_and(all_annotations != 0, all_detections == all_annotations))
        P  = np.count_nonzero(all_annotations != 0)
        descp = '{:.0f} positives out of {:.0f} samples'.format(P, all_annotations.size),
        return TP / P, descp

    all_detections, all_annotations = _getResults(generator, model)
    return _recall(all_detections, all_annotations)


def eval_mAP(generator, model, **kwargs):
    recording = {'recall': {}, 'fps': {}}
    average_precisions = evaluate(generator, model, recording = recording, **kwargs)

    total_instances = []
    precisions = []
    for label, (average_precision, num_annotations ) in average_precisions.items():
        total_instances.append(num_annotations)
        precisions.append(average_precision)
    mean_ap = sum(precisions) / sum(x > 0 for x in total_instances)

    recall = recording['recall'][1]
    fps    = recording['fps'][1]
    descp  = '{:.0f} instances of class {}, of which the recall is {}, and false positives per scan(FPs) is {}'.format(average_precisions[1][1], generator.labelToName(1), recall, fps)
    return mean_ap, descp


class Evaluate(keras.callbacks.Callback):
    """ Evaluation callback for arbitrary datasets.
    """

    def __init__(
        self,
        generator,
        mode,
        tensorboard=None,
        verbose=1,
        **kwargs
    ):
        """ Evaluate a given dataset using a given model at the end of every epoch during training.

        # Arguments
            generator        : The generator that represents the dataset to evaluate.
            mode             : 'recall' or 'accuracy'.
            tensorboard      : Instance of keras.callbacks.TensorBoard used to log the mAP value.
            verbose          : Set the verbosity level, by default this is set to 1.
        """
        self.generator       = generator

        if mode == 'recall':
            self.evaluate = eval_recall
        elif mode == 'accuracy':
            self.evaluate = eval_accuracy
        elif mode == 'mAP':
            self.evaluate = eval_mAP
        else:
            raise ValueError('unsupported evaluation callback mode')
        self.mode = mode

        self.tensorboard     = tensorboard
        self.verbose         = verbose
        self.kwargs          = kwargs

        super(Evaluate, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # run evaluation
        self.metric, descp  = self.evaluate(self.generator, self.model, **self.kwargs)
        if self.verbose > 0:
            print('{}: {:.4f}'.format(self.mode, self.metric), descp)

        if self.tensorboard is not None and self.tensorboard.writer is not None:
            import tensorflow as tf
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = self.metric
            summary_value.tag = self.mode
            self.tensorboard.writer.add_summary(summary, epoch)

        logs[self.mode] = self.metric
