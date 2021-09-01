import copy
import itertools
from operator import attrgetter
import numpy as np
from pandas.core.common import flatten

from Scene import Scene
from YOLO.bound_box import BoundBox

from grouping.graph import Graph
import functools
import operator
from texts import *

from helper import get_location_names_indexes


def generate_groups(intersection_mtx):
    g = Graph()
    for row_idx, row in enumerate(intersection_mtx):
        for col_idx, col in enumerate(row):
            if intersection_mtx[row_idx][col_idx] != 0:
                g.addEdge(row_idx, col_idx)
    k = set()
    for idx, _ in enumerate(intersection_mtx):
        k.add(g.DFS(idx))
    f = [list(x) for x in k]
    return f


def _adj_width_height(box, n):
    XtopLeft, YtopLeft = int(box[1]), int(box[2])
    XbottomRight, YbottomRight = int(box[3]), int(box[4])
    return box[0], XtopLeft - n, YtopLeft - n, XbottomRight + n, YbottomRight + n


def generate_inter_matrix(ids, pred):
    allowed_location_names = get_location_names_indexes([CLOSE, TOUCHING, CROSSING, INSIDE, LARGER, SPLIT, SAME])
    df = pred[np.in1d(pred[:, 0], ids) & np.in1d(pred[:, 2], ids) & np.in1d(pred[:, 4], allowed_location_names)]
    mx = np.amax(ids)
    intersection_mtx = np.zeros((mx + 1, mx + 1))
    for row in df:
        fuzzy_mutual_position_matrix = row[8]
        intersection_mtx[int(row[0]), int(row[2])] = fuzzy_mutual_position_matrix
        intersection_mtx[int(row[2]), int(row[0])] = fuzzy_mutual_position_matrix
    return intersection_mtx


def grouping_ids(boxes_with_order_numbers, pred):
    from YOLO.img_det import Box
    new_b_boxes = list()

    list_of_b_boxes = functools.reduce(operator.iconcat, list(boxes_with_order_numbers.values()), [])
    new_b_boxes.append(boxes_with_order_numbers['scene'])
    for key, value in boxes_with_order_numbers.items():
        if key is not 'scene':
            if len(value) > 1:
                b_boxes_groups_separated = list()
                key_id = attrgetter("id")
                ids = [key_id(box) for box in value]
                inter_mtx = generate_inter_matrix(ids, pred)
                b_boxes_groups_separated = generate_groups(inter_mtx)
                for group in b_boxes_groups_separated:
                    obj_quantity_in_group = len(group)
                    if len(group) >= 2:
                        XtopLeft, YtopLeft, XbottomRight, YbottomRight = grouping_coordinates(
                            filtr(group, list_of_b_boxes))
                        box = Box(BoundBox(XtopLeft, YtopLeft, XbottomRight, YbottomRight), key, None, None,
                                  is_group=True)
                        box.obj_quantity_in_group = obj_quantity_in_group
                        new_b_boxes.append(box)
                    else:
                        if group[0] in ids:
                            new_b_boxes.append(filtr(group, list_of_b_boxes))
            else:
                new_b_boxes.append(value)
    new_b_labels = [box.label for box in flatten(new_b_boxes)]
    return list(flatten(new_b_boxes)), new_b_labels


def filtr(group, list_of_b_boxes):
    key_id = attrgetter("id")
    objects = list()
    for x in list_of_b_boxes:
        if key_id(x) in group:
            objects.append(x)
    return objects


def grouping_coordinates(b_boxes_to_merge):
    XtopLeftList, YtopLeftList = list(), list()
    XbottomRightList, YbottomRightList = list(), list()
    for box in b_boxes_to_merge:
        XtopLeftList.append(int(box.box.XtopLeft))
        YtopLeftList.append(int(box.box.YtopLeft))
        XbottomRightList.append(int(box.box.XbottomRight))
        YbottomRightList.append(int(box.box.YbottomRight))

    return min(XtopLeftList), min(YtopLeftList), \
           max(XbottomRightList), max(YbottomRightList)


def grouping(boxes_with_order_numbers, pred, scene):
    new_b_boxes, new_b_labels = grouping_ids(boxes_with_order_numbers, pred)
    new_scene, v_labels_matlab, v_boxes_matlab, v_labels_matlab_sequential \
        = new_gtruth(new_b_boxes, scene)
    return new_scene, v_labels_matlab, v_boxes_matlab, v_labels_matlab_sequential, new_b_boxes


# new scene object with grouped objects
def new_gtruth(new_b_boxes, scene):
    new_labels = list()
    v_boxes_matlab = []
    v_labels_matlab = []
    v_labels_matlab_sequential = []

    # list of b_boxes to group
    for idx, box in enumerate(new_b_boxes):
        XtopLeft, YtopLeft, XbottomRight, YbottomRight = box.box.XtopLeft, box.box.YtopLeft, \
                                                         box.box.XbottomRight, box.box.YbottomRight
        width = XbottomRight - XtopLeft
        height = YbottomRight - YtopLeft

        def check_labels(lab):
            if lab not in v_labels_matlab:
                v_labels_matlab.append(lab)
            return v_labels_matlab.index(lab)

        box.id = idx
        b = [idx, check_labels(box.label), XtopLeft, YtopLeft, width, height]
        v_boxes_matlab.append(b)
        v_labels_matlab_sequential.append(box.label)

    size = scene.size
    onames = v_labels_matlab
    obj_num = len(v_boxes_matlab)
    ocolors = []
    ocolors.extend(itertools.repeat([1, 1, 1], obj_num))
    obj = np.array(v_boxes_matlab)

    obj_org = copy.copy(obj)
    background = []
    im = scene.im
    background2 = []
    gtruth = Scene(im=im, fname=scene.fname, size=size, onames=onames, ocols=[], obj=obj, obj_num=obj_num,
                   obj_org=obj_org,
                   background=background, background2=background2)
    return gtruth, v_labels_matlab, v_boxes_matlab, v_labels_matlab_sequential
