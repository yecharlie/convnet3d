import os
import csv
import numpy as np
import warnings
import glob

from six import raise_from
from sklearn.model_selection import GroupKFold

from .annotations import (
    readAnnotations,
    readClasses,
    openForCsv
)
from .image import (readImage, readSeries)

DATASET_CLASSES = {"bg": 0, "aneurysm": 1}
DATASET_LABELS = {0: "bg", 1: "aneurysm"}
SERIES_CSV_NAME = "kfold_series{id}_{phase}"
PATCHES_CSV_NAME = 'kfold_patches{id}_{stage}_{phase}'
MAPPING_NAME = 'mapping.csv'


def makeKFold(nfolds, csv_file, out_dir):
    gkf = GroupKFold(n_splits=nfolds)
    try:
        with open(csv_file, "r", newline="") as file:
            reader = csv.reader(file, delimiter=',')
            X = list(reader)

            groups = [x[-1] for x in X]
    except ValueError as e:
        raise_from(ValueError('invalid CSV annotations file {}: {}'.format(csv_file, e)), None)

    global SERIES_CSV_NAME
    for f, (train_indices, test_indices) in enumerate(gkf.split(X, groups=groups)):
        train_set_name = os.path.join(out_dir, SERIES_CSV_NAME.format(id=f, phase="train"))
        test_set_name = os.path.join(out_dir, SERIES_CSV_NAME.format(id=f,  phase="test") )
        train_set = [X[i] for i in train_indices]
        test_set = [X[i] for i in test_indices]
        try:
            with open(train_set_name, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(train_set)
        except ValueError as e:
            raise_from(ValueError('invalid CSV annotations file {}: {}'.format(train_set_name, e)), None)
        try:
            with open(test_set_name, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(test_set)
        except ValueError as e:
            raise_from(ValueError('invalid CSV annotations file {}: {}'.format(test_set_name, e)), None)


def _makeSureFileClear(path):
    if os.path.isfile(path):
        warnings.warn('The object file {} has already existed, which is automatically removed.'.format(path))
        os.remove(path)


def makePatchesForDetection(
    dataset3d_dir,
    cls_sides=(30, 30, 15),
    cls_sample_rate = 4,
    volume_reader = readSeries,
    dataset_classes = DATASET_CLASSES,
    dataset_labels  = DATASET_LABELS,
    verbose=0
):
    out_dir = dataset3d_dir  # save result under the same directory with series dataset

    # find alll series3d dataset in the directory
    global SERIES_CSV_NAME, MAPPING_NAME
    search_template = os.path.join(dataset3d_dir, SERIES_CSV_NAME.format(id='*', phase='*'))
    series_csv_list  = glob.glob(search_template)
    cls_imgindex = 0  # patch-image-id across all CV dataset
    pathsmps = os.path.join(out_dir, "patches")  # patches directory
    if not os.path.exists(pathsmps):
        print('Create images directories {}'.format(pathsmps))
        os.makedirs(pathsmps)

    # Mapping file
    pathmp = os.path.join(out_dir, MAPPING_NAME)
    createMappingCSV(pathmp, dataset_classes)

    cls_sides = np.array(cls_sides)

    def _makePatches(series_csv, detection_csv):
        nonlocal cls_imgindex, pathsmps
        imgname = 'cs{:0>4d}_{}.npy'  # padding 0s by left side

        try:
            with open(series_csv, "r", newline="") as file:
                csv_reader = csv.reader(file)
                series_annotations = readAnnotations(csv_reader, dataset_classes)
        except ValueError as e:
            raise_from(ValueError('invalid csv annotations file{}: {}'.format(series_csv, e)), None)

        if verbose > 0:
            print('preparing kfoldCV patches at {} for detection model.'.format(detection_csv))

        def reproducePos(pos_annotations, reproduction_rate = 4):
            # reproduce new samples by shifting original samples
            rpd = []
            for anno in pos_annotations:
                coords = anno['coords']
                offsets = np.random.randint(-5, 6, (reproduction_rate, coords.size))
                rpd += [{'coords': newc, 'diameter': anno['diameter'], 'class':anno['class']} for newc in coords + offsets]
                if verbose > 1:
                    print('from {} reproducing positives \n{}'.format(coords, rpd[-reproduction_rate:]))

            return pos_annotations + rpd

        def sampleNeg(series_path, pos_annotations, cls_sides, rate_to_pos=3, reader=volume_reader):
            # randomly generate negatives samples within the whole volume

            negs = []

            image_size = reader(series_path).GetSize()
            num_pos = len(pos_annotations)
            randx = np.random.randint(cls_sides[0] // 2, image_size[0] - cls_sides[0] // 2, rate_to_pos * num_pos )
            randy = np.random.randint(cls_sides[1] // 2, image_size[1] - cls_sides[1] // 2, rate_to_pos * num_pos)
            randz = np.random.randint(cls_sides[2] // 2, image_size[2] - cls_sides[2] // 2, rate_to_pos * num_pos)
            negs += list(np.concatenate([
                np.expand_dims(randx, axis=-1),
                np.expand_dims(randy, axis=-1),
                np.expand_dims(randz, axis=-1)
            ], axis=-1))
            if verbose > 1:
                print('random sample negative \n{}'.format(negs[-num_pos * rate_to_pos:]))
            return negs

        for name, pos_annotations in series_annotations.items():
            num_original_pos = len(pos_annotations)

            # original positive + reproduced positive
            pos_annotations = reproducePos(pos_annotations)
            neg_cls_samples = sampleNeg(name, pos_annotations, cls_sides)

            # the leading samples are positives, and the rest are negatives
            all_cls_samples = [np.asarray(pan["coords"]) for pan in pos_annotations] + list(neg_cls_samples)
            all_cls_images, valid_indices, all_cls_samples, _ = readImage(name, cls_sides, *all_cls_samples, reader=volume_reader, verbose=verbose)

            detection_csv_rows = []
            for idx, img, smp in zip(valid_indices, all_cls_images, all_cls_samples):
                if idx < num_original_pos:
                    marker = 'op'
                elif idx < len(pos_annotations):
                    marker = 'rp'
                else:
                    marker = 'ng'
                # the main purpose of marker is to debug
                img_saved_path = os.path.join(pathsmps,imgname.format(cls_imgindex, marker))
                lbl = pos_annotations[idx]['class'] if idx < len(pos_annotations) else dataset_labels[0]
                detection_csv_rows.append([img_saved_path, lbl, *smp, '', name])  # the name field mark source, which will be useful in reduction model training

                np.save(img_saved_path, img)
                cls_imgindex += 1
            try:
                with open(detection_csv, "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerows(detection_csv_rows)
            except ValueError as e:
                raise_from(ValueError('invalid csv annotations file{}: {}'.format(detection_csv, e)), None)

    # ------------------------end of _makePatches def--------------------------
    for sidx, series_csv in enumerate(series_csv_list):
        series_csv_name = os.path.basename(series_csv)
        print('{} processing series csv file at {}'.format(sidx, series_csv))

        detection_csv_name = series_csv_name.replace('series', 'patches').replace('train', 'detection_train').replace('test', 'detection_test')  # Note that either 'test' or 'train will be replaced
        detection_csv = os.path.join(out_dir, detection_csv_name)
        _makeSureFileClear(detection_csv)

        _makePatches(series_csv, detection_csv)


def makePatchesForReduction(
    dataset3d_dir,
    cs_dataset_id,
    cs_model,
    patches_sides = (60, 60, 25),
    rate_to_pos = 3,
    volume_reader = readSeries,
    gpu = -1,
    verbose=0
):
    if gpu != -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    # retrive related files in the directory
    global SERIES_CSV_NAME, PATCHES_CSV_NAME, MAPPING_NAME
    detection_train = os.path.join(dataset3d_dir, SERIES_CSV_NAME.format(id=cs_dataset_id, phase='train'))
    reduction_train = os.path.join(dataset3d_dir, PATCHES_CSV_NAME.format(id=cs_dataset_id, stage='reduction', phase='train'))
    classes_file = os.path.join(dataset3d_dir, MAPPING_NAME)
    samples_outdir = os.path.join(dataset3d_dir, 'patches')

    if not os.path.exists(samples_outdir):
        print('Create images directories {}'.format(samples_outdir))
        os.makedirs(samples_outdir)
    _makeSureFileClear(reduction_train)

    imgname = 'fpr{}_{:0>4d}_{}.npy'  # padding 0s by left side
    imgindex = 0

    def _remakePatchesFromSource(
        reduction_csv,
        series,
        samples,
        labels,
        markers,
        convert_centroids
    ):
        '''remake patches selected by names in annotations with new sides parameters
        '''

        reduction_csv_rows = []
        nonlocal imgindex
        samples = np.array(samples)

        images, vi, samples, _ = readImage(series, patches_sides, *samples, reader=volume_reader, convert_centroids=convert_centroids, verbose=verbose)
        labels = [labels[i] for i in vi]  # dilter
        markers = [markers[i] for i in vi]  # filter
        for img, smp, lbl, mk in zip(images, samples, labels, markers):
            img_saved_path = os.path.join(
                samples_outdir,
                imgname.format(cs_dataset_id, imgindex, mk)
            )
            reduction_csv_rows.append([img_saved_path, lbl, *smp, '', series]) 

            np.save(img_saved_path, img)
            imgindex += 1

        try:
            with open(reduction_csv, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(reduction_csv_rows)
        except ValueError as e:
            raise_from(ValueError('invalid csv annotations file{}: {}'.format(reduction_csv, e)), None)

        return len(vi)
    # ------------------end of defination _remakePatchesFromSource----------------

    def _readCSV(csv_file, reader_fn, *args, **kwargs):
        '''read csv using specific processer
        '''
        try:
            with openForCsv(csv_file) as file:
                data = reader_fn(csv.reader(file, delimiter=','), *args, **kwargs)
        except ValueError as e:
            raise_from(ValueError('invalid CSV class file {}: {}'.format(csv_file, e)), None)
        return data
    # ------------end of defination _readCSV ----------

    def _writeCSVWithOneSeries(csv_file, annotations, series):
        series_anno = annotations[series]
        rows = []
        for anno in series_anno:
            rows.append([series, anno['class'], *anno['coords'], anno['diameter']])
        try:
            with open(csv_file, 'w', newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerows(rows)
        except ValueError as e:
            raise_from(ValueError('invalid CSV class file {}: {}'.format(csv_file, e)), None)
    # ------------end of defination _writeCSVWithOneSeries ----------

    def _numOriginalPosAnnotions(annotations):
        num = 0
        for series in annotations.keys():
            num += len(annotations[series])
        return num
    # -----------------end of defination _numOriginalPosAnnotions---------------

    def _reproducePos(pos_annotations, reproduction_rate = 4):
        # reproduce new samples by shifting original samples
        rpd = []
        for anno in pos_annotations:
            coords = anno['coords']
            offsets = np.random.randint(-5, 6, (reproduction_rate, coords.size))
            rpd += [{'coords': newc, 'diameter': anno['diameter'], 'class':anno['class']} for newc in coords + offsets]
            if verbose > 1:
                print('from {} reproducing positives \n{}'.format(coords, rpd[-reproduction_rate:]))

        return pos_annotations + rpd
    # ----------------end of defination _reproducePos-------------------------

    from ..preprocessing.val_generator import ValidationGenerator
    from ..models import (loadModel, detectionPred)
    from .eval import evaluate
    import tempfile

    # classes
    dataset_classes = _readCSV(classes_file, readClasses)
    dataset_labels = {}
    for key, value in dataset_classes.items():
        dataset_labels[value] = key

    # original series annotations
    annotations = _readCSV(detection_train, readAnnotations, dataset_classes.keys())
    series_names = list(annotations.keys())
    sidx = 0

    # learn the number of original positive annotations and infer wanted fp numbers
    numop  = _numOriginalPosAnnotions(annotations)
    reproduction_rate = 4
    numpos = numop * (1 + reproduction_rate)
    numfp  = numpos * rate_to_pos
    curfp  = 0

    # prepare our model
    cs_model = loadModel(cs_model)
    cs_pred  = detectionPred(cs_model)

    print('processing series csv file at {}'.format(detection_train))
    with tempfile.TemporaryDirectory() as tmpdir:
        print('create temporary directory', tmpdir)
        while sidx < len(series_names) and curfp < numfp:
            sname = series_names[sidx]
            print('\n{} evaluating series {}'.format(sidx, sname))

            tmpcsv_name = os.path.join(tmpdir, '{}.csv'.format(sidx))
            _writeCSVWithOneSeries(tmpcsv_name, annotations, sname)
            print('create temporary file', tmpcsv_name)

            # create our generator with one series only
            generator = ValidationGenerator(tmpcsv_name, classes_file)
            recording = {'fpd':{}}  # record false positive detections

            # evaluate
            evaluate(
                generator,
                cs_pred,
                transfer='detection-pred',
                window_size = (25, 60, 60),
                sliding_strides = (25, 60, 60),
                score_threshold = 0.8,
                nms             = True,
                recording = recording
            )

            # false positives at label 1 image 0
            # boxes + scores
            false_positives = recording['fpd'][1][0]
            false_positives = false_positives[:, :6]
            fp_samples = (false_positives[:, ::2] + false_positives[:, 1::2]) / 2
            print('return {} false positive samples with the leading three\n{} '.format(len(fp_samples), fp_samples[:3]))

            pos_annotations = annotations[sname]
            num_pos_annotations = len(pos_annotations)
            print('positive annotations\n {}'.format(pos_annotations))
            pos_samples = _reproducePos(pos_annotations, reproduction_rate)  # original positives followed by reproduced samples

#            #op: original positive; rp: reproduced positive; fp: false positive

            _remakePatchesFromSource(
                reduction_train,
                sname,
                samples = [psm['coords'] for psm in pos_samples],
                labels  = [psm['class'] for psm in pos_samples],
                markers = ['op'] * num_pos_annotations + ['rp'] * (len(pos_samples) - num_pos_annotations),
                convert_centroids = True
            )

            # Note that the coordinates has benn converted when the image is being preprocessing
            _remakePatchesFromSource(
                reduction_train,
                sname,
                samples = fp_samples,
                labels = ['bg'] * len(fp_samples),
                markers = ['fp'] * len(fp_samples),
                convert_centroids = False
            )

            # move to next series
            sidx += 1
            curfp += len(fp_samples)


def createMappingCSV(csvPath, dataset_classes=DATASET_CLASSES):
    rows = []
    for k, v in dataset_classes.items():
        rows.append([k, v])
    with open(csvPath, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(rows)
