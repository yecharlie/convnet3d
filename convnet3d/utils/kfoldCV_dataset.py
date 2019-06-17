import os
import sys
import csv
import math
import numpy as np
import warnings
import glob
import matplotlib
matplotlib.use("Agg")#for web server;this can avoid TclError
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D#Axes3D is imported but not used. Otherwise a ValueError is raised: "Unknown projection '3d'." 

import keras 
import tensorflow as tf

from six import raise_from
from deprecated import deprecated
from sklearn.model_selection import GroupKFold
from .visualization import (
    drawCircle,
    circleFacecolors,
    clusterColor,
    sizeHistgram
)
from .dataset import (datasetOp, readSeries)
from .reconstruct3d import (
    Record3dPiece,
    AdaptiveClustering,
    ClusteringError
)
from .annotations import (
    readAnnotations,
    readClasses,
    openForCsv,
    sampleNegative,
    sampleRegressors
)
from .image import readImage

DATASET_CLASSES = {"bg":0,"aneurysm":1}
DATASET_LABELS = {0:"bg",1:"aneurysm"}
SERIES_CSV_NAME = "kfold_series{id}_{phase}"
PATCHES_CSV_NAME = 'kfold_patches{id}_{stage}_{phase}'
MAPPING_NAME = 'mapping.csv'

def generate3d(
    originds,
    outds,
    minDiameter = 3.5,
    nfolds = 1,
    visualized = True
):
    #image index the variable which will be used in gencsvOp function
#    imgindex = 0
#    pathsmps = os.path.join(outds,"samples")
    pathcsv = os.path.join(outds,"allLabels.csv")
    pathpngs = os.path.join(outds,"visualized") if visualized else None
    pathlog = os.path.join(outds,'log')

    def gen3dOp(img,rcds,sidx,series,outds):
        try:
            adpc = AdaptiveClustering(series[sidx][1],series[sidx][0])
            pieces, centroids = adpc.clustering()
            clusters = AdaptiveClustering.getAllClusters(pieces,len(centroids),series[sidx][0])
        except ClusteringError as e:
            print(e.message)
            print("pieces: ",adpc.pieces)
            return

        global DATASET_LABELS
        nonlocal pathcsv,pathpngs,pathlog
        rows = []
        log = []
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        for idxc,c in enumerate(clusters):
            c.sort(key=lambda piece:piece.record[2])
            spacing = img.GetSpacing()
            center_piece, diameter, plane_bound, depth_bound = _estimateClusterDiameter(c,spacing)
            diameter = diameter / 1  + 1#vocel coordinates in the standard domain 

            #positive samples
            rows.append([series[sidx][0],DATASET_LABELS[1]] + list(c[center_piece].record[:3])+[diameter,sidx])
            #the log is saved to detemine the input size of model
            log.append([plane_bound,depth_bound])
            print("Rcord: ",list(c[center_piece].record[:3])+[diameter])
            for idxp,piece in enumerate(c):
                x,y,z,r = piece.record
                r = r / 1 + 1 #vocel radius in the standard domain 
                drawCircle(x,y,z,r,ax=ax,color=np.array(clusterColor(idxc)) / 255)

                if idxp == center_piece:#render a ranbow ring to indiactes 3d record from a cluster
                    fcolors = circleFacecolors(x,y,z,diameter+2,diameter,cmap="hsv")
                    drawCircle(x,y,z,diameter+2,diameter,ax=ax,facecolors=fcolors,shade=False) 

        if pathpngs is not None:
            pngname = "{}{}.png"
            patname = os.path.basename(
                        os.path.dirname(
                            os.path.dirname(series[sidx][0]))
                      )
            plt.savefig(os.path.join(pathpngs,pngname.format(sidx, patname.replace(" ",'_'))))
            
        with open(pathcsv,"a",newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(rows)

        with open(pathlog,'a',newline='') as logfile:
            logwriter = csv.writer(logfile)
            logwriter.writerows(log)
   #---------- end of gen3dOp 's defination --------

    if pathpngs is not None and not os.path.exists(pathpngs):
        os.makedirs(pathpngs)

    #Because csv is open with 'a' option, this can avoid duplicate writting.
    _makeSureFileClear(pathcsv)
    _makeSureFileClear(pathlog)

    #generate visualized dataset in 'outds/visualized' and labels in 'outds/allLabels.csv'
    series,_ = datasetOp(originds,gen3dOp,minDiameter,True,outds)

    if pathpngs is not None:
    #dataset information
        pathhist = os.path.join(pathpngs,'__dataset_size_hist__.png')
        showSizeInfo(pathcsv,pathlog,pathhist)

    if nfolds > 1:
    #k-fold Cross Validation
        _makeKFold(nfolds, pathcsv, outds )

def _estimateClusterDiameter(pieces,spacing):
    '''Estimate diameter

    Args:
        pieces          : records belongs to a cluster in a seriess, sorted with ascending z-coords.
        spacing         : image spacing.
    Returns:
        center          : index of center pieces
        diameter        : diameter in 3d space
        plane_bound     : max diameter in 2d plane
        depth_bound     : depth in pixel 
    '''
    center = len(pieces) // 2
    
    lower_bounds = [math.sqrt(p.record[3]**2 +
        (abs(pieces[center].record[2] - p.record[2])*spacing[2])**2)
        for p in pieces]

    plane_bound = math.ceil(1 + max([ p.record[3] / spacing[0] for p in pieces]))#pixel unit
    depth_bound = pieces[-1].record[2] - pieces[0].record[2]  + 1#pixel unit


    return center,2 * max(*lower_bounds),plane_bound,depth_bound

def _makeKFold( nfolds, csv_file, out_dir):
    gkf = GroupKFold(n_splits=nfolds)
    try:
        with open(csv_file,"r", newline="") as file:
            reader = csv.reader(file,delimiter=',')
            X = list(reader)

            groups = [x[-1] for x in X]
    except ValueError as e:
        raise_from(ValueError('invalid CSV annotations file {}: {}'.format(csv_file,e)),None)

    global SERIES_CSV_NAME
    for f, (train_indices, test_indices) in enumerate(gkf.split(X,groups=groups)):
        train_set_name = os.path.join(out_dir, SERIES_CSV_NAME.format(id=f, phase="train"))
        test_set_name = os.path.join(out_dir, SERIES_CSV_NAME.format(id=f,  phase="test") )
        train_set = [X[i] for i in train_indices]
        test_set = [X[i] for i in test_indices]
        try:
            with open(train_set_name,"w",newline="") as file:
                writer = csv.writer(file)
                writer.writerows(train_set)
        except ValueError as e:
            raise_from(ValueError('invalid CSV annotations file {}: {}'.format(train_set_name,e)),None)
        try:
            with open(test_set_name,"w",newline="") as file:
                writer = csv.writer(file)
                writer.writerows(test_set)
        except ValueError as e:
            raise_from(ValueError('invalid CSV annotations file {}: {}'.format(test_set_name,e)),None)


def _makeSureFileClear(path):
    if os.path.isfile(path):
        warnings.warn('The object file {} has already existed, which is automatically removed.'.format(path))
        os.remove(path)

def showSizeInfo(alllabels,log,hist_saved_path):
    try:
        with open(alllabels,"r",newline="") as labels_file, open(log, 'r', newline='') as log_file:
            labels_reader = csv.reader(labels_file)
            diameters_3d = np.array([float(line[-2]) for line in labels_reader])
            print('max diameter is {}, mean is {}'.format(diameters_3d.max(),diameters_3d.mean()))

            log_reader = csv.reader(log_file)
            X = list(log_reader)

            diameters_2d,depths = (np.array([int(x[i]) for x in X]) for i in range(2)) 
            print('max plane diameter is {}, mean is {}.\nmax depth is {}, mean is {}'.format(diameters_2d.max(),diameters_2d.mean(),depths.max(),depths.mean()))
    except ValueError as e:
        raise_from(ValueError('invalid CSV file {} or {}: {}'.format(alllabels,log,e)),None)

    sizeHistgram(diameters_3d,diameters_2d,depths,hist_saved_path)

#    try:
#        with open(log,'r',newline='') as file:
#            reader = csv.reader(file)
#            X = list(reader)
#            plane_d,nax_depth = (max([int(x[i]) for x in X]) for i in range(2)) 
#    except ValueError as e:
#        raise_from(ValueError('invalid CSV annotations file {}: {}'.format(log,e)),None)

#def _getSeriesDataset(
#    dataset3d_dir,
#    
#):
#    global SERIES_CSV_NAME
#    search_template = os.path.join(dataset3d_dir,SERIES_CSV_NAME.format('*','*'))
#    series_csv_list  = glob.glob(search_template)
#    return series_csv_list

def makePatchesForDetection(
    dataset3d_dir,
    cls_sides=(30,30,15),
    cls_sample_rate = 4,
    volume_reader = readSeries,
    dataset_classes = DATASET_CLASSES,
    dataset_labels  = DATASET_LABELS,
    verbose=0
):
    out_dir = dataset3d_dir#save result under the same directory with series dataset

    #find alll series3d dataset in the directory
    global SERIES_CSV_NAME, MAPPING_NAME
    search_template = os.path.join(dataset3d_dir,SERIES_CSV_NAME.format(id='*', phase='*'))
    series_csv_list  = glob.glob(search_template)
    cls_imgindex = 0#patch-image-id across all CV dataset
    pathsmps = os.path.join(out_dir,"patches")#patches directory
    if not os.path.exists(pathsmps):
        print('Create images directories {}'.format(pathsmps))
        os.makedirs(pathsmps)

    #Mapping file
    pathmp = os.path.join(out_dir, MAPPING_NAME)
    createMappingCSV(pathmp, dataset_classes)

    cls_sides = np.array(cls_sides)
    def _makePatches(series_csv, detection_csv):
        nonlocal cls_imgindex, pathsmps
        imgname = 'cs{:0>4d}_{}.npy'#padding 0s by left side
        
        try:
            with open(series_csv,"r",newline="") as file:
                csv_reader = csv.reader(file)
                series_annotations = readAnnotations(csv_reader,dataset_classes)
        except ValueError as e:
            raise_from(ValueError('invalid csv annotations file{}: {}'.format(series_csv,e)),None)

        if verbose > 0:
            print('preparing kfoldCV patches at {} for detection model.'.format(detection_csv))

        def reproducePos(pos_annotations, reproduction_rate = 4):
            #reproduce new samples by shifting original samples
            rpd = []
            for anno in pos_annotations:
                coords = anno['coords']
                offsets = np.random.randint(-5,6,(reproduction_rate, coords.size))
                rpd += [{'coords':newc, 'diameter':anno['diameter'],'class':anno['class']} for newc in coords+offsets]
                if verbose > 1:
                    print('from {} reproducing positives \n{}'.format(coords, rpd[-reproduction_rate:]))

            return pos_annotations + rpd

        def sampleNeg(series_path, pos_annotations, cls_sides, rate_to_pos=3, reader=volume_reader):
            negs = []

            image_size = reader(series_path).GetSize()
            num_pos = len(pos_annotations)
            randx = np.random.randint(cls_sides[0] //2, image_size[0] - cls_sides[0]//2, rate_to_pos * num_pos )
            randy = np.random.randint(cls_sides[1] //2, image_size[1] - cls_sides[1]//2, rate_to_pos * num_pos)
            randz = np.random.randint(cls_sides[2] //2, image_size[2] - cls_sides[2]//2, rate_to_pos * num_pos)
            negs += list(np.concatenate([
                np.expand_dims(randx,axis=-1),
                np.expand_dims(randy,axis=-1),
                np.expand_dims(randz,axis=-1)
                ],axis=-1
            ))
            if verbose > 1:
                print('random sample negative \n{}'.format(negs[-num_pos * rate_to_pos:]))
            return negs
                
        for name, pos_annotations in series_annotations.items():
            num_original_pos = len(pos_annotations)

            #original positive + reproduced positive
            pos_annotations = reproducePos(pos_annotations)
            neg_cls_samples = sampleNeg(name, pos_annotations, cls_sides)

            #the leading samples are positives, and the rest are negatives
            all_cls_samples = [np.asarray(pan["coords"]) for pan in pos_annotations] + list(neg_cls_samples)
            all_cls_images, valid_indices, all_cls_samples, _ = readImage(name,cls_sides,*all_cls_samples,reader=volume_reader, verbose=verbose)

            detection_csv_rows = [] 
            for idx, img, smp in zip(valid_indices, all_cls_images, all_cls_samples):
                if idx < num_original_pos:
                    marker = 'op'
                elif idx < len(pos_annotations):
                    marker = 'rp'
                else:
                    marker = 'ng'
                #the main purpose of marker is to debug
                img_saved_path = os.path.join(pathsmps,imgname.format(cls_imgindex, marker))
                lbl = pos_annotations[idx]['class'] if idx < len(pos_annotations) else dataset_labels[0]
                detection_csv_rows.append([img_saved_path,lbl,*smp,'',name])#the name field mark source, which will be useful in reduction model training

                np.save(img_saved_path,img)
                cls_imgindex += 1
            try:
                with open(detection_csv,"a",newline="") as file:
                    writer = csv.writer(file)
                    writer.writerows(detection_csv_rows)
            except ValueError as e:
                raise_from(ValueError('invalid csv annotations file{}: {}'.format(detection_csv,e)),None)
            
    #------------------------end of _makePatches def--------------------------
    for sidx,series_csv in enumerate(series_csv_list):
        series_csv_name = os.path.basename(series_csv)
        print('{} processing series csv file at {}'.format(sidx,series_csv))

        detection_csv_name = series_csv_name.replace('series','patches').replace('train','detection_train').replace('test','detection_test')#Note that either 'test' or 'train will be replaced
        detection_csv = os.path.join(out_dir,detection_csv_name)
        _makeSureFileClear(detection_csv)

        _makePatches(series_csv,detection_csv)



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
        
    #retrive related files in the directory
    global SERIES_CSV_NAME, PATCHES_CSV_NAME, MAPPING_NAME
    detection_train = os.path.join(dataset3d_dir, SERIES_CSV_NAME.format(id=cs_dataset_id, phase='train'))
    reduction_train = os.path.join(dataset3d_dir, PATCHES_CSV_NAME.format(id=cs_dataset_id, stage='reduction', phase='train'))
    classes_file = os.path.join(dataset3d_dir, MAPPING_NAME)
    samples_outdir = os.path.join(dataset3d_dir, 'patches')

    if not os.path.exists(samples_outdir):
        print('Create images directories {}'.format(samples_outdir))
        os.makedirs(samples_outdir)
    _makeSureFileClear(reduction_train)

    imgname = 'fpr{}_{:0>4d}_{}.npy'#padding 0s by left side
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

        images, vi, samples , _ = readImage(series,  patches_sides, *samples, reader=volume_reader, convert_centroids=convert_centroids,verbose=verbose)
        labels = [labels[i] for i in vi]#dilter
        makers = [markers[i] for i in vi]#filter
        for img, smp, lbl, mk in zip(images, samples, labels, markers):
            img_saved_path = os.path.join(samples_outdir,
                imgname.format(cs_dataset_id, imgindex, mk))
            reduction_csv_rows.append([img_saved_path,lbl,*smp,'',series]) 

            np.save(img_saved_path, img)
            imgindex += 1

        try:
            with open(reduction_csv,"a",newline="") as file:
                writer = csv.writer(file)
                writer.writerows(reduction_csv_rows)
        except ValueError as e:
            raise_from(ValueError('invalid csv annotations file{}: {}'.format(reduction_csv, e)),None)

        return len(vi)
    #------------------end of defination _remakePatchesFromSource---------------- 
    def _readCSV(csv_file, reader_fn, *args, **kwargs):
        '''read csv using specific processer
        '''
        try:
            with openForCsv(csv_file) as file:
                data = reader_fn(csv.reader(file, delimiter=','),*args,**kwargs)
        except ValueError as e:
            raise_from(ValueError('invalid CSV class file {}: {}'.format(csv_file, e)), None)
        return data
    #------------end of defination _readCSV ----------
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
    #------------end of defination _writeCSVWithOneSeries ----------
    def _numOriginalPosAnnotions(annotations):
        num = 0
        for series in annotations.keys():
            num += len(annotations[series])
        return num
    #-----------------end of defination _numOriginalPosAnnotions---------------
    def _reproducePos(pos_annotations, reproduction_rate = 4):
        #reproduce new samples by shifting original samples
        rpd = []
        for anno in pos_annotations:
            coords = anno['coords']
            offsets = np.random.randint(-5,6,(reproduction_rate, coords.size))
            rpd += [{'coords':newc, 'diameter':anno['diameter'],'class':anno['class']} for newc in coords+offsets]
            if verbose > 1:
                print('from {} reproducing positives \n{}'.format(coords, rpd[-reproduction_rate:]))

        return pos_annotations + rpd
    #----------------end of defination _reproducePos-------------------------

    from ..preprocessing.val_generator import ValidationGenerator 
    from ..models import (loadModel, detectionPred)
    from .eval import evaluate
    import tempfile

    #classes 
    dataset_classes = _readCSV(classes_file, readClasses)
    dataset_labels = {}
    for key, value in dataset_classes.items():
        dataset_labels[value] = key

    #original series annotations
    annotations = _readCSV(detection_train, readAnnotations, dataset_classes.keys())
    series_names = list(annotations.keys())
    sidx = 0

    #learn the number of original positive annotations and infer wanted fp numbers
    numop  = _numOriginalPosAnnotions(annotations)
    reproduction_rate = 4
    numpos = numop * (1 + reproduction_rate)
    numfp  = numpos * rate_to_pos
    curfp  = 0

    #prepare our model
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

            #create our generator with one series only
            generator =  ValidationGenerator(tmpcsv_name, classes_file)
            recording = {'fpd':{}}#record false positive detections

            #evaluate
            evaluate(
                generator,
                cs_pred,
                transfer='detection-pred',
                window_size = (25,60,60),
                sliding_strides = (25,60,60),
                score_threshold = 0.8,
                nms             = True,
                recording = recording
            )

            #false positives at label 1 image 0
            #boxes + scores 
            false_positives = recording['fpd'][1][0]
            false_positives = false_positives[:,:6]
            fp_samples = (false_positives[:,::2] + false_positives[:,1::2]) / 2
            print('return {} false positive samples with the leading three\n{} '.format(len(fp_samples), fp_samples[:3]))

            pos_annotations = annotations[sname]
            num_pos_annotations = len(pos_annotations)
            print('positive annotations\n {}'.format(pos_annotations))
            pos_samples = _reproducePos(pos_annotations, reproduction_rate)#original positives followed by reproduced samples
#            all_samples = [psm['coords'] for psm in pos_samples] + list(fp_samples)
#            all_labels  = [psm['class'] for psm in pos_samples] + ['bg'] * len(fp_samples)
#
#            #op: original positive; rp: reproduced positive; fp: false positive
#            all_markers = ['op'] * num_pos_annotations + ['rp'] * (len(pos_samples) - num_pos_annotations) + ['fp'] * len(fp_samples)

            _remakePatchesFromSource(
                reduction_train,
                sname,
                samples = [psm['coords'] for psm in pos_samples],
                labels  = [psm['class'] for psm in pos_samples],
                markers = ['op'] * num_pos_annotations + ['rp'] * (len(pos_samples) - num_pos_annotations),
                convert_centroids = True
            )

            #Note the coordinates has benn converted when the image is being preprocessing
            _remakePatchesFromSource(
                reduction_train,
                sname,
                samples = fp_samples,
                labels = ['bg'] * len(fp_samples),
                markers = ['fp'] * len(fp_samples),
                convert_centroids = False
            )

            #move to next series 
            sidx +=1
            curfp += len(fp_samples)


@deprecated(version=1.0, reason='Use makePatchesForDetection and makePatchesForReduction instead')
def makePatches(
    dataset3d_dir,
    cls_sides,
    reg_sides, 
    cls_sample_rate = 6,
    reg_sample_strides = (10, 10, 5), 
    volume_reader = readSeries,
    dataset_classes = DATASET_CLASSES,
    dataset_labels  = DATASET_LABELS,
    verbose=0
):
    '''Make patches dataset

    Make patches dataset based on the previous generated series dataset. The new dataset will be maded under the same directory.

    Args:
        dataset3d_dir
        cls_sides
        reg_sides
        strides
        cls_sample_rate
    '''

    out_dir = dataset3d_dir#save result under the same directory with series dataset

    global SERIES_CSV_NAME
    search_template = os.path.join(dataset3d_dir,SERIES_CSV_NAME.format('*','*'))
    series_csv_list  = glob.glob(search_template)
    cls_imgindex = reg_imgindex = 0
    pathsmps = os.path.join(out_dir,"patches")
    if not os.path.exists(pathsmps):
        print('Create images directories {}'.format(pathsmps))
        os.makedirs(pathsmps)
    cls_sides = np.array(cls_sides)
    reg_sides = np.array(reg_sides)
    strides   = np.array(reg_sample_strides)

    def _makePatches(series_csv, detection_csv, reduction_csv, verbose):
        nonlocal cls_imgindex,reg_imgindex, pathsmps
        imgname = '{}_sample{:0>4d}.npy'#padding 0s by left side
        
        try:
            with open(series_csv,"r",newline="") as file:
                reader = csv.reader(file)
                series_annotations = readAnnotations(reader,dataset_classes)
        except ValueError as e:
            raise_from(ValueError('invalid csv annotations file{}: {}'.format(series_csv,e)),None)

        names = series_annotations.keys()

        if verbose:
            print('preparing kfoldCV patches at {} for detection model.'.format(detection_csv))

        #Make Patches-of-image dataset for candidates proposal network
        for name in names:
            pos_annotations = series_annotations[name]
            neg_cls_samples = sampleNegative(pos_annotations,cls_sides,verbose)

            #down-sampling negatives
            choices = np.random.choice(
                len(neg_cls_samples),
                min(len(neg_cls_samples),
                    cls_sample_rate * len(pos_annotations) ),
                replace = False
            )
            neg_cls_samples = [neg_cls_samples[ch] for ch in choices]

            all_cls_samples = [np.asarray(pan["coords"]) for pan in pos_annotations] + list(neg_cls_samples)
            all_cls_images, valid_indices, all_cls_samples, _ = readImage(name,cls_sides,*all_cls_samples,reader=volume_reader, verbose=verbose)
#            all_cls_samples = [all_cls_samples[vidx] for vidx in valid_indices]#filter annotations

            detection_csv_rows = [] 
            for idx, img, smp in zip(valid_indices, all_cls_images, all_cls_samples):
                img_saved_path = os.path.join(pathsmps,imgname.format("cls",cls_imgindex))
                lbl = pos_annotations[idx]['class'] if idx < len(pos_annotations) else dataset_labels[0]
                detection_csv_rows.append([img_saved_path,lbl,'','','',''])

                np.save(img_saved_path,img)
                cls_imgindex += 1
            try:
                with open(detection_csv,"a",newline="") as file:
                    writer = csv.writer(file)
                    writer.writerows(detection_csv_rows)
            except ValueError as e:
                raise_from(ValueError('invalid csv annotations file{}: {}'.format(detection_csv,e)),None)
            
        if verbose:
            print('preparing kfoldCV patches at {} for reduction model.'.format(reduction_csv))
        for name in names:
            pos_annotations = series_annotations[name]
            pos_reg_samples = sampleRegressors(pos_annotations,reg_sides,cls_sides,strides,verbose) 
            neg_reg_samples = sampleNegative(pos_annotations,reg_sides,verbose)
            all_reg_samples = [np.asarray(preg[:3]) for preg in pos_reg_samples] + neg_reg_samples

            all_reg_images, valid_indices, all_reg_samples, isotropicSpace = readImage(name,reg_sides,*all_reg_samples, reader=volume_reader, verbose=verbose)
#            all_reg_samples = [all_reg_samples[vidx] for vidx in valid_indices]#filtering
            reduction_csv_rows = []
            for idx,img,smp in zip(valid_indices,all_reg_images,all_reg_samples):
                img_saved_path = os.path.join(pathsmps,imgname.format("reg",reg_imgindex))
                if idx < len(pos_reg_samples):
                #Note that one positive annotations is responsible for more than one pos-reg-samples
                    lbl = pos_annotations[pos_reg_samples[idx][3]]['class']#namely lbl='aneurysm'

                    #the coordinates related to the left-top corner of image patch
                    #Note that coordinates are in the new isotropic space
                    target = isotropicSpace(pos_annotations[pos_reg_samples[idx][3]]['coords'] - 1) - (smp - reg_sides // 2)
                    diameter = pos_annotations[pos_reg_samples[idx][3]]['diameter']
                else:
                    lbl = dataset_labels[0] 
                    target = ['','','']
                    diameter = ''
                reduction_csv_rows.append([img_saved_path,lbl,*target,diameter])
                
                np.save(img_saved_path,img)
                reg_imgindex += 1
            try:
                with open(reduction_csv,"a",newline="") as file:
                    writer = csv.writer(file)
                    writer.writerows(reduction_csv_rows)
            except ValueError as e:
                raise_from(ValueError('invalid csv annotations file{}: {}'.format(reduction_csv,e)),None)
    #------------------------end of _makePatches def--------------------------

    for sidx,series_csv in enumerate(series_csv_list):
        series_csv_name = os.path.basename(series_csv)
        print('{} processing series csv file at {}'.format(sidx,series_csv))

        detection_csv_name = series_csv_name.replace('series','patches').replace('train','detection_train').replace('test','detection_test')#Note that either 'test' or 'train will be replaced
        reduction_csv_name = series_csv_name.replace('series','patches').replace('train','reduction_train').replace('test','reduction_test')
        detection_csv = os.path.join(out_dir,detection_csv_name)
        reduction_csv = os.path.join(out_dir,reduction_csv_name)
        _makeSureFileClear(detection_csv)
        _makeSureFileClear(reduction_csv)

        _makePatches(series_csv,detection_csv,reduction_csv,verbose = verbose)

def createMappingCSV(csvPath, dataset_classes=DATASET_CLASSES):
    rows = []
    for k,v in dataset_classes.items():
        rows.append([k,v])
    with open(csvPath,"w",newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(rows)

