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

