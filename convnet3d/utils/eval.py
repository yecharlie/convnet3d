"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
"""
THIS FILE HAS BEEN MODIFIED.
"""

import keras
import keras.backend as K
import numpy as np
import os

import cv2
import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."


from .window import windowing
from .annotations import computeOverlaps
from .nms import nmsOverlaps


def _computeAP(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))


    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])


    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]


    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap




def getResults(
    generator,
    model,
    transfer='convnet3d-1b',
    window_size=None,
    sliding_strides=None,
    nms=False,
    score_threshold=0.05,
    max_detections=100,
    overlap_threshold = 0.5
):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 6 + 1]

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[np.empty((0,7)) for i in range(generator.numClasses()) if generator.hasLabel(i)] for j in range(generator.size())]


    all_annotations = [[np.empty((0,6)) for i in range(generator.numClasses())] for j in range(generator.size())]
    for i in progressbar.progressbar(range(generator.size()), prefix='Running network: '):
        # load the annotations
        annotations  = generator.loadAnnotations(i)
        raw_image    = generator.loadImage(i)
        image, annotations  = generator.preprocessGroupEntry(raw_image, annotations)

        # copy detections to all_annotations
        for label in range(generator.numClasses()):
            if not generator.hasLabel(label):
                continue

            all_annotations[i][label] = annotations['bboxes'][annotations['labels'] == label, :].copy()


        image_size = image.shape[:3]

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((3, 0, 1, 2))

#        print('IMG-SHAPE',image.shape)

        #while directly apply model to a whole volume will raise an OOM error, we add a prior-windowing operation instead.
        if window_size is not None and sliding_strides is not None:
            #Note the order z,y,x
            windows_offset = windowing(image_size, window_size, sliding_strides)
        else:#i.e. no windowing
            windows_offset = np.zeros((1,3), dtype=int)
            window_size = image_size

        #create window
        if keras.backend.image_data_format() == 'channels_last':
            window = np.zeros(tuple(window_size) + (image.shape[3],), dtype=K.floatx())
        else:
            window = np.zeros((image.shape[0],) + tuple(window_size), dtype=K.floatx())

        image_detections = np.empty((0,6+1+1))#boxes + socres + labels
        for ofs in windows_offset:
            #padding window
            if keras.backend.image_data_format() == 'channels_last':
                window[:,:,:,:] = image[
                    ofs[0]:ofs[0] + window_size[0],
                    ofs[1]:ofs[1] + window_size[1],
                    ofs[2]:ofs[2] + window_size[2],:] 
            else:
                window[:,:,:,:] = image[:,
                    ofs[0]:ofs[0] + window_size[0],
                    ofs[1]:ofs[1] + window_size[1],
                    ofs[2]:ofs[2] + window_size[2]] 
            
            def convnet3dOneBatchdOutputsTransfer(outputs):
                classification, boxes = outputs[:2]

                #different from regressed boxes (at below, commented), these boxes used the coordinates (z,y,x)
                boxes = boxes[:,:,[4,5,2,3,0,1]]

                scores = classification.max(axis=1)
                labels = classification.argmax(axis=1)
                return boxes, scores, labels

            def detectionPredTransfer(outputs):
                boxes, scores, labels =  outputs[:3]

                #different from regressed boxes (at below), these boxes used the coordinates (z,y,x)
                boxes = boxes[:,:,[4,5,2,3,0,1]]
                return boxes, scores, labels

            def convnet3dTwoBatchdOutputsTransfer(outputs):
                #JUST FOR DEMON, NOT USE!

                reg_boxes, regression, classification = outputs[:3]
                scores = classification.max(axis=1)
                labels = classification.argmax(axis=1)
                return reg_boxes, scores, labels

            if transfer=='convnet3d-1b':
                transfer = convnet3dOneBatchdOutputsTransfer
            elif transfer == 'detection-pred':
                transfer = detectionPredTransfer

            # run network
            boxes, scores, labels = transfer(model.predict_on_batch(np.expand_dims(window, axis=0)))

            # select indices which have a score above the threshold
            indices = np.where(scores[0, :] > score_threshold)[0]

            # select those scores
            scores = scores[0][indices]

            # find the order with which to sort the scores
            scores_sort = np.argsort(-scores)

            # select detections
            window_boxes      = boxes[0, indices[scores_sort], :]
            window_boxes     += np.array([ofs[2], ofs[2], ofs[1], ofs[1], ofs[0], ofs[0]])#Note the order
#            print('ofs={} window_size={} when the first thress boxes are {}'.format(ofs, window_size, window_boxes[:3]))
            window_scores     = scores[scores_sort]
            window_labels     = labels[0, indices[scores_sort]]
            window_detections = np.concatenate([window_boxes, np.expand_dims(window_scores, axis=1), np.expand_dims(window_labels, axis=1)], axis=1)
            image_detections  = np.concatenate([image_detections, window_detections], axis = 0)

        def nms(image_detections, overlap_threshold):
            boxes = image_detections[:, :6]
            scores = image_detections[:, 6]
            overlaps = computeOverlaps(boxes, boxes)

            #get valid boxes
            vi = np.where(overlaps.diagonal()==1)[0]
            overlaps = overlaps[np.expand_dims(vi, axis=-1) ,vi]
            #filter
            image_detections = image_detections[vi]
            scores = scores[vi]

            indices = nmsOverlaps(overlaps, scores, overlap_threshold)
            return image_detections[indices]

        #non-max-suppression could eliminate the side effect of overlapped windowing
        if nms:
            image_detections = nms(image_detections, overlap_threshold)

        #at last, get top_k detections
        image_scores = image_detections[:,6]
        scores_sort = np.argsort(-image_scores)[:max_detections]
        image_detections = image_detections[scores_sort]

        # copy detections to all_detections
        for label in range(generator.numClasses()):
            if not generator.hasLabel(label):
                continue


            all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]


    return all_detections, all_annotations




#def _getAnnotations(generator):
#    """ Get the ground truth annotations from the generator.
#
#    The result is a list of lists such that the size is:
#        all_annotations[num_images][num_classes] = annotations[num_annotations, 6]
#
#    # Arguments
#        generator : The generator used to retrieve ground truth annotations.
#    # Returns
#        A list of lists containing the annotations for each image in the generator.
#    """
#    all_annotations = [[np.empty((0,6)) for i in range(generator.numClasses())] for j in range(generator.size())]
#
#    for i in progressbar.progressbar(range(generator.size()), prefix='Parsing annotations: '):
#        # load the annotations
#        annotations = generator.loadAnnotations(i)
#
#
#        # copy detections to all_annotations
#        for label in range(generator.numClasses()):
#            if not generator.hasLabel(label):
#                continue
#
#
#            all_annotations[i][label] = annotations['bboxes'][annotations['labels'] == label, :].copy()
#
#
#    return all_annotations



def hitTesting(query_boxes, boxes):
    testing_points = (query_boxes[:,1::2] + query_boxes[:,::2]) / 2
    targets_points = (boxes[:,1::2] + boxes[:,::2]) / 2
    diameters = (boxes[:,1::2] - boxes[:,::2]).max()
    norm = np.linalg.norm(
        -np.expand_dims(testing_points, axis=1) 
        +targets_points, axis=2)
    return norm > diameters

def evaluate(
    generator,
    model,
    transfer='convnet3d-1b',
    window_size = None,
    sliding_strides = None,
    nms = False,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100,
    recording = {}
):
    """ Evaluate a given dataset using a given model.

    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
        recording       : If provided, record additional detection informarion as indicated.
    # Returns
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    all_detections, all_annotations     = getResults(generator, model, transfer,  window_size=window_size, sliding_strides=sliding_strides, nms=nms,  score_threshold=score_threshold, max_detections=max_detections)
#    print('all_detections\n',all_detections)
#    print('all_annotations\n',all_annotations)

#    all_annotations    = _getAnnotations(generator)
    average_precisions = {}


    # all_detections = pickle.load(open('all_detections.pkl', 'rb'))
    # all_annotations = pickle.load(open('all_annotations.pkl', 'rb'))
    # pickle.dump(all_detections, open('all_detections.pkl', 'wb'))
    # pickle.dump(all_annotations, open('all_annotations.pkl', 'wb'))


    # process detections and annotations
    for label in range(generator.numClasses()):
        if not generator.hasLabel(label):
            continue


        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0


        for i in range(generator.size()):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []


            for d in detections:
                scores = np.append(scores, d[4])


                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue


#                overlaps            = computeOverlaps(np.expand_dims(d, axis=0), annotations)
                overlaps            = hitTesting(np.expand_dims(d[:6], axis=0), annotations)

                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]


                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)

            #record in a image
            def recorderAtImageLevel(recording):
                if 'fpd' in recording: 
                   # fps tag for newest set of detections
                   fps  = false_positives[-len(detections):]
                   fpd = recording['fpd']
                   if label not in fpd:
                        fpd[label] = []
                   fpd[label].append(
                       np.array([ d for fp, d in zip(fps, detections) if fp])
                   )
            recorderAtImageLevel(recording)
                    

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue


        # sort by score
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]


        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)


        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
        def recorderAtLabelLevel(recording):
            if 'recall' in recording:
                ultimate_recall = recall.max()
                rc = recording['recall']
                rc[label]=ultimate_recall

            if 'fps' in recording:
                fps = false_positives.max() / generator.size()
                recording['fps'][label]=fps
                
        recorderAtLabelLevel(recording)

        # compute average precision
        average_precision  = _computeAP(recall, precision)
        average_precisions[label] = average_precision, num_annotations


    return average_precisions
