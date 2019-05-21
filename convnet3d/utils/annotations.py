import sys
import math
import numpy as np

from six import raise_from
from collections import Iterable

def openForCsv(path):
    """ Open a file with flags suitable for csv.reader.
    This is different for python2 it means with mode 'rb',
    for python3 this means 'r' with "universal newlines".
    """
    if sys.version_info[0] < 3:
        return open(path, 'rb')
    else:
        return open(path, 'r', newline='')

def parse(value, function, fmt):
    """
    Parse a string into a value, and format a nice ValueError if it fails.
    Returns `function(value)`.
    Any `ValueError` raised is catched and a new `ValueError` is raised
    with message `fmt.format(e)`, where `e` is the caught `ValueError`.
    """
    try:
        return function(value)
    except ValueError as e:
        raise_from(ValueError(fmt.format(e)), None)

def tobbox(centroid,sides):
    '''To Bonding box

    Convert a cube coordinates to the form like [x1,x2,y1,y2,z1,z2]

    Args:
        centroid    : The centroid of cube.
        sides       : Respective side length, could be a int value or a tuple.
    Returns:
        x           : ndarray of (6,), one could get a patch from array[x[0]:x[1],x[2]:x[3],x[4]:x[5]].
    '''
    if isinstance(sides,Iterable):
        assert len(sides) == len(centroid)
        sides = np.asarray(sides)
    centroid = np.asarray(centroid)

    xi = np.asarray( centroid - sides / 2,dtype=int)
    xj = np.asarray( xi + sides, dtype=int)
    x = np.zeros(len(xi) * 2,dtype=int)
    x[0::2] = xi
    x[1::2] = xj
    return x

def computeOverlaps(boxes,query_boxes):
    '''Compute Overlaps

    Args
        boxes       : (N,4) ndarray of float
        query_boxes : (M,4) ndarray of float

    Returns
        overlaps    : (N,M) ndarray of overlaps between boxes and query_boxes.
    '''
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N,K))
    for i,b in enumerate( boxes ):
        #calculate volume b
        b_volume = 1
        for di in range(1,len(b),2):
            b_volume *= b[di] - b[di-1]

        for j,qb in enumerate( query_boxes ):
            try:
                #calculate intersection volume of ba nd qb
                intersection_volume = 1
                axis_iter = iter(range(1,len(b),2))
                while intersection_volume > 0 :
                    k = next(axis_iter)
                    intersection_volume *= min(b[k],qb[k]) - max(b[k-1],qb[k-1])
            except StopIteration:
                #when intersection volume > 0
                #calculate volume qb
                qb_volume = 1
                for di in range(1,len(qb),2):
                    qb_volume *= qb[di] - qb[di-1]
                union_volume = b_volume + qb_volume - intersection_volume
                overlaps[i][j] = intersection_volume / union_volume
    return overlaps

def sampleRegressors(annotations,reg_sides,cls_sides,strides, verbose=0):
    '''Sample regressors

    Args:
        annotations     : Positive samples from the same series, represented by list of dicts. See mergeAnnoitations.
        reg_sides       : Input size of regression model, must be 3-D (x,y,z).
        cls_sides       : Input size of classification model (candidates proposal model), must be 3-D (x,y,z).
        strides         : Strides for sampling.
    Returns:
        total_regressors : list of tuple (x,y,z,t), where (x,y,z) is the center of regressor and t is the target index in 'annotations'. 
    '''

    reg_sides = np.asarray(reg_sides, dtype=int)
    cls_sides = np.asarray(cls_sides, dtype=int)
    strides   = np.asarray(strides, dtype=int)

    assert len(reg_sides) == len(cls_sides) == len(strides) == 3
    assert np.all(reg_sides >= cls_sides)

    total_regressors = []
    lims = np.zeros((3,2))
    for idx,an in enumerate(annotations):
        centroid = an['coords']

        #The first column stores the min value of regressors' center;the second colum stores the max.
        #Make sure that the object is completely under the vision of regression network.
        lims[:,0] = centroid + cls_sides / 2 - reg_sides / 2
        lims[:,1] = centroid - cls_sides / 2 + reg_sides / 2
        coords = [np.arange(lims[axis,0],lims[axis,1]+1,strides[axis],dtype=int) for axis in range(lims.shape[0])]
        gridx,gridy,gridz = np.meshgrid(*coords,indexing="ij")
        regressors = np.concatenate([
            np.expand_dims(gridx,axis=-1),
            np.expand_dims(gridy,axis=-1),
            np.expand_dims(gridz,axis=-1)
            ],axis=-1
        )
        regressors = regressors.reshape((-1,3))

        if verbose >= 2:
            print('Around {}, the lims is\n{},\nand regressors are\n{}.'.format(centroid, lims, regressors))
            
        total_regressors += [( *reg,idx ) for reg in regressors]
    return total_regressors

def sampleNegative(annotations,sides,verbose=False):
    '''Sample negative samples around positive samples

    Args:
        annotations : Positive samples from the same series, represented by list of dicts. See mergeAnnoitations.
        sides       : sides length of samples, must be 3-D(x,y,z).

    Returns:
        total_negatives : list of (3,) ndarray.
    '''
    total_negatives = []
    sides = np.asarray(sides)

    #get all positive bbox first
    pos_bboxes = _getBboxes(annotations)
    for an in annotations:
        #we set the offset to half of the diameter of aneurysm because we couldn't assume min(sides) > diameter.
        neg_locs    = _neighborNegativesLocs(an["coords"], sides, an['diameter'] // 2)
        neg_bboxes  = np.zeros((neg_locs.shape[0],6))

        #convert the locations to bboxes form
        for i in range(neg_locs.shape[0]):
            neg_bboxes[i] = tobbox(neg_locs[i][:3], sides)

        overlaps = computeOverlaps(neg_bboxes,pos_bboxes)

        #an negative sample should not appear in the range of diameter of any positiove  sample
        indices = np.where(np.sum(overlaps,axis=1) == 0)[0]
        if verbose >= 1:
            for i, nbox in enumerate(neg_bboxes):
                if i not in indices:
                    print('sample negative box {} is overlapped with positive box.'.format(nbox))
            
        if verbose >= 2:
            print('Around {} of which the diameter is {}, the negative samples are\n{}'.format(an['coords'],an['diameter'],neg_locs[indices]))

        total_negatives += [neg for neg in neg_locs[indices]]
    return total_negatives

def _getBboxes(annotations):
    '''Return the bonding boxes of the annotated region.

    This function take the value of math.ceil(an['diameter'] / 2) as 'sides' param to function tobbox, for 'an' in annotations.
    Args:
        annotations     : Positive samples from the same series, represented by list of dicts. See mergeAnnoitations.

    Returns:
        bboxes
    '''
    bboxes = np.zeros((len(annotations),6))
    for i,an in enumerate(annotations):
        centroid = an["coords"]
        sides = math.ceil(an['diameter'] / 2)
        bboxes[i] = tobbox(centroid, sides)
    return bboxes

def _neighborNegativesLocs(centroid, sides, offset=5):
    '''Generate negative samples' centroids around a centroid

    Sample the 26 direction around a positive sample to which its distance along one or more axis 'ax' is sides[ax] + abs(offset) and the left axis is 0.
    Args:
        offset      : a scaler which is added to the distance from neg-sample to pos-sample.
    '''
    locations = np.zeros((26,3)) + centroid

    strides = np.zeros((3,3))

    strides[:,0] = 0
    strides[:,1] = sides  + abs(offset)
    strides[:,2] = -sides - abs(offset)
    for i,si in enumerate(strides[0]):
        for j,sj in enumerate(strides[1]):
            for k,sk in enumerate(strides[2]):
                if not (i == j == k == 0):#otherwise si==sj==sk==0
                    locations[i * 9 + j * 3 + k -1] += np.array((si,sj,sk))
                    
    return locations

#def _neighborNegativesLocs(centroid, sides, offset=5):
#    '''Generate negative samples' centroids around a centroid
#    '''
#    locations = np.zeros((6,3))
#    locations += centroid
#    
#    #points at six basic direction
#    strides = np.zeros((3,2))
#    strides[:,0] = sides  + abs(offset)
#    strides[:,1] = -sides - abs(offset)
#    for i in range(len(strides)):
#        for j,s in enumerate( strides[i] ):
#            #i * 2 + j -> iteration with row order
#            locations[i * len(strides[i]) + j,i] += s
#
#    return locations

#def mergeAnnotations(annotations):
#    '''Merge annotations with same 'img_file'
#
#    Args:
#        annotations     : result of readAnnotations
#
#    Returns:
#        merged, :A dict keyed by 'img_file', and has values of list of records. Every record is represented by a dict with keys  "coords","diameter","class".The merged records in a sublist share the same 'img_file' atrribute.
#    '''
#    result = {}
#    for an in annotations:
#        an = an[0]#get the dict from [dict]
#        if an["img_file"] not in result:
#            result[an["img_file"]] = []
#        result[an["img_file"]].append({"coords":an["coords"],"diameter":an["diameter"],"class":an["class"]})
#    return result
#
#def readAnnotations(csv_reader, classes):
#    """ Read annotations from the csv_reader.
#    
#    For each row in file should be with this format (img_file,class,x,y,z,d), which consistes of four parts: 1) image locations,"img_file"; 2) label, "class"; 3) coordinates related, "x,y,z";4) size related ,"d".
#
#    Returns:
#        rows:   List of list of dict [[dict1],[dict2],...] with fields "img_file", "class", "coords", "diameter", one dict for one row. This somewhat ackward form is for compatible reason of operatrion
#    """
#    result = []
#
#    for line, row in enumerate(csv_reader):
#        line += 1
#
#        try:
#            img_file, class_name, x, y, z, d = row[:6]
#        except ValueError:
#            raise_from(ValueError('line {}: format should be \'img_file,x,y,z,d,class_name\' or \'img_file,,,,,\''.format(line)), None)
#
#        x = parse(x, int, 'line {}: malformed x: {{}}'.format(line))
#        y = parse(y, int, 'line {}: malformed y: {{}}'.format(line))
#        z = parse(z, int, 'line {}: malformed z: {{}}'.format(line))
#        d = parse(d, float, 'line {}: malformed d: {{}}'.format(line))
#
#        # check if the current class name is correctly present
#        if class_name not in classes:
#            raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))
#
#        anno = [{'img_file':img_file, 'coords': (x,y,z), "diameter":d, 'class': class_name}]
#        result.append(anno)
#    return result

def readAnnotations(csv_reader, classes):
    """ Read annotations from the csv_reader.
    
    For each row in file should be with this format (img_file,class,x,y,z,d), which consistes of four parts: 1) image locations,"img_file"; 2) label, "class"; 3) coordinates related, "x,y,z";4) size related ,"d".
    Support omited "x","y","z","d", with the format (img_file,class,,,,) when the respective record dicts   have single field 'class'.

    Returns:
        rows:  A dict, keyed by 'img_file', return the values list of rcords dict with fields "class", "coords", "diameter". 
    """
    result = {}

    for line, row in enumerate(csv_reader):
        line += 1

        try:
            img_file, class_name, x, y, z, d = row[:6]
            if len(row) > 6:
                dic = {'others':row[6:]}
            else:
                dic = {}
        except ValueError:
            raise_from(ValueError('line {}: format should be \'img_file,class_name,x,y,z,d\' or \'img_file,class_name,,,,\''.format(line)), None)
        if img_file not in result:
            result[img_file] = []

        # check if the current class name is correctly present
        if class_name not in classes:
            raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

        dic.update({'class':class_name})
        if (x,y,z,d) == ('','','',''):
            result[img_file].append(dic)
            continue
            
        x = parse(x, float, 'line {}: malformed x: {{}}'.format(line))
        y = parse(y, float, 'line {}: malformed y: {{}}'.format(line))
        z = parse(z, float, 'line {}: malformed z: {{}}'.format(line))
        dic.update({'coords':np.array([x,y,z])})
        if d == '':
            result[img_file].append(dic)
            continue

        d = parse(d, float, 'line {}: malformed d: {{}}'.format(line))
        dic.update({'diameter':d})
        result[img_file].append(dic)

    return result

def readClasses(csv_reader):
    """ Parse the classes file given by csv_reader.
    """
    result = {}
    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise_from(ValueError('line {}: format should be \'class_name,class_id\''.format(line)), None)
        class_id = parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id

    if 'bg' not in result or result['bg'] != 0:
        raise ValueError('Bockgound mapping error (should be "bg"-> 0)')
        
    return result

