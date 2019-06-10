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

THIS FILE HAS BEEN MODIFIED.
"""

import random
import csv

import numpy as np
from six import raise_from

import keras

from ..utils.transform import transformBbox
from ..utils.image import (
    transformImage,
    huwindowing
)
from ..utils.annotations import (
    readAnnotations,
    readClasses,
    openForCsv
)
from ..utils.tobbox import tobbox


def windowingAndMinusMean(image, annotations):
    x = image
    x = x.astype(keras.backend.floatx())
    x = huwindowing(x, level=80, window=600, outmin = 0, outmax = 255)
    x -= 60.039

    return x, annotations


class Generator(keras.utils.Sequence):
    def __init__(
        self,
        csv_data_file,
        csv_clsses_file,
        batch_size=1,
        shuffle_groups=True,
        preprocessImage = windowingAndMinusMean,
        transform_generator=None,
        **kwargs
    ):
        self.batch_size = batch_size
        self.shuffle_groups = shuffle_groups
        self.preprocessImage = preprocessImage
        self.transform_generator = transform_generator
        self.kwargs = kwargs

        try:
            with openForCsv(csv_clsses_file) as file:
                self.classes = readClasses(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise_from(ValueError('invalid CSV class file {}: {}'.format(csv_clsses_file, e)), None)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        try:
            with openForCsv(csv_data_file) as file:
                self.image_data = readAnnotations(csv.reader(file, delimiter=','), self.classes)
        except ValueError as e:
            raise_from(ValueError('invalid CSV annotations file {}: {}'.format(csv_data_file, e)), None)
        self.image_names = list(self.image_data.keys())

        self.groupImages()

        if self.shuffle_groups:
            self.onEpochEnd()

    def onEpochEnd(self):
        random.shuffle(self.groups)

    def size(self):
        return len(self.image_names)

    def numClasses(self):
        return max(self.classes.values()) + 1

    def nameToLabel(self, name):
        return self.classes[name]

    def labelToName(self, label):
        return self.labels[label]

    def hasLabel(self, label):
        return label in self.labels

    def hasName(self, name):
        return name in self.classes

    def loadImage(self, image_index):
        image_path = self.image_names[image_index]
        return np.load(image_path)

    def loadAnnotations(self, image_index):
        annotations = self.image_data[self.image_names[image_index]]
        std_annotations = {
            "bboxes": np.empty((0, 6)),
            "labels": np.empty((0,))
        }
        for idx, an in enumerate(annotations):
            std_annotations["labels"] = np.concatenate((std_annotations['labels'], [self.nameToLabel(an["class"])]), axis=0)
            if 'coords' in an and 'diameter' in an:
                centroid = an['coords']
                diameter = an['diameter']
                bbox = tobbox(centroid, diameter)
                std_annotations['bboxes'] = np.concatenate(
                    [std_annotations['bboxes'],
                        np.expand_dims(bbox, axis = 0)], axis=0
                )

        assert std_annotations['bboxes'].shape[0] == 0 or std_annotations['labels'].shape[0] == std_annotations['bboxes'].shape[0], 'unsupported dataset format with labels {}, bboxes {} for an image.'.format(std_annotations['labels'], std_annotations['bboxes'])

        return std_annotations

    def loadAnnotationsGroup(self, group):
        return [self.loadAnnotations(image_index) for image_index in group]

    def loadImageGroup(self, group):
        return [self.loadImage(image_index) for image_index in group]

    def randomTransformGroupEntry(self, image, annotations):
        def isPositiveSample(annotations):
            return self.nameToLabel('bg') not in annotations['labels']

        if self.transform_generator and isPositiveSample(annotations):
            matrix, translation = next(self.transform_generator)

            image, transform = transformImage(image, matrix, translation, **self.kwargs)
            annotations['bboxes'] = annotations['bboxes'].copy()
            for index in range(annotations['bboxes'].shape[0]):
                annotations['bboxes'][index, :] = transformBbox(annotations['bboxes'][index, :], transform)
        return image, annotations

    def randomTransformGroup(self, image_group, annotation_group):
        assert (len(image_group) == len(annotation_group))

        for index in range(len(image_group)):
            image_group[index], annotation_group[index] = self.randomTransformGroupEntry(image_group[index], annotation_group[index])
        return image_group, annotation_group

    def preprocessGroupEntry(self, image, annotations):
        """ Preprocess image and its annotations.
        """
        # preprocess the image
        image, annotations = self.preprocessImage(image, annotations)
        # convert to the wanted keras floatx
        image = keras.backend.cast_to_floatx(image)
        return image, annotations

    def preprocessGroup(self, image_group, annotations_group):
        """ Preprocess each image and its annotations in its group.
        """
        assert(len(image_group) == len(annotations_group))

        for index in range(len(image_group)):
            # preprocess a single group entry
            image_group[index], annotations_group[index] = self.preprocessGroupEntry(image_group[index], annotations_group[index])

        return image_group, annotations_group

    def groupImages(self):
        order = list(range(self.size()))
        random.shuffle(order)

        # one group, one batch
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size) ]

    def computeInputs(self, image_group):
        """ Compute inputs for the network using an image_group.
        """
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(4))

        # construct an image batch object
        image_batch = np.zeros((self.batch_size,) + max_shape, dtype=keras.backend.floatx())

        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group):
            image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2], :image.shape[3]] = image

        if keras.backend.image_data_format() == 'channels_first':
            image_batch = image_batch.transpose((0, 4, 1, 2, 3))

        return image_batch

    def computeTargets(self, image_group, annotations_group):
        assert len(image_group) == len(annotations_group), "The length of the images and annotations should be equal."

        labels_batch = np.zeros((self.batch_size,), dtype=keras.backend.floatx())
        for idx, an in enumerate(annotations_group):
            labels_batch[idx] = an["labels"][0]
        return labels_batch

    def computeInputOutput(self, group):
        """ Compute inputs and target outputs for the network.
        """
        # load images and annotations
        image_group       = self.loadImageGroup(group)
        annotations_group = self.loadAnnotationsGroup(group)

        # perform preprocessing steps
        image_group, annotations_group = self.preprocessGroup(image_group, annotations_group)

        # randomly transform data
        image_group, annotations_group = self.randomTransformGroup(image_group, annotations_group)

        # compute network inputs
        inputs = self.computeInputs(image_group)

        # compute network targets
        targets = self.computeTargets(image_group, annotations_group)

        return inputs, targets

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        group = self.groups[index]
        inputs, targets = self.computeInputOutput(group)

        return inputs, targets
