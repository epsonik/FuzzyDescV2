import copy
import itertools
from operator import attrgetter
import numpy as np
from pandas.core.common import flatten

from Scene import Scene
from YOLO.bound_box import BoundBox
from YOLO.img_det import Box
from grouping.graph import Graph
import functools
import operator

# def group_filter_name(names_set: set, v_boxes_temp: list):
#     groups = {}
#     v_boxes_new = []
#     if names_set.__len__() != v_boxes_temp.__len__():
#         for name in names_set:
#             groups[name] = []
#             for box in v_boxes_temp:
#                 temp_name = box[0]
#                 if temp_name == name:
#                     groups[name].append(box)
#     else:
#         v_boxes_new = v_boxes_temp
#     for key, value in groups.items():
#         if len(value) > 1:
#             is_group(key, value, v_boxes_new)
#         else:
#             v_boxes_new.append(value[0])
#     return v_boxes_new
#
#
# def is_group(key, value, v_boxes_new):
#     # intersection_over_union_val = _intersection_measure(value[0], value[1])
#     inter_mtx = generate_inter_matrix(value)
#     separated_groups_of_b_boxes = generate_groups(inter_mtx)
#     for group_idx, group in enumerate(separated_groups_of_b_boxes):
#         b_boxes_to_merge = [value[i] for i in group]
#         v_boxes_new.append([f'grupa {group_idx} skladajaca sie z {key}'] + list(grouping(b_boxes_to_merge)))
#     return v_boxes_new


# def _intersection_measure(box_a, box_b, n=15, stop_condition=3):
#     intersection_over_union_val = _intersection_over_union(box_a, box_b)
#     if intersection_over_union_val > 0:
#         return intersection_over_union_val
#     box_a_adj = box_a
#     box_b_adj = box_b
#     for _ in itertools.repeat(None, stop_condition):
#         box_a_adj = _adj_width_height(box_a_adj, n)
#         box_b_adj = _adj_width_height(box_b_adj, n)
#         intersection_over_union_val = _intersection_over_union(box_a_adj, box_b_adj)
#         if intersection_over_union_val > 0:
#             return intersection_over_union_val
#     return intersection_over_union_val
from helper import count_ids


def generate_groups(intersection_mtx):
    g = Graph()
    for row_idx, row in enumerate(intersection_mtx):
        for col_idx, col in enumerate(row):
            if intersection_mtx[row_idx][col_idx] != 0:
                g.addEdge(row_idx, col_idx)
    k = set()
    for idx, _ in enumerate(intersection_mtx):
        k.add(g.DFS(idx))
    return [list(x) for x in k if len(list(x)) > 1]


def _adj_width_height(box, n):
    XtopLeft, YtopLeft = int(box[1]), int(box[2])
    XbottomRight, YbottomRight = int(box[3]), int(box[4])
    return box[0], XtopLeft - n, YtopLeft - n, XbottomRight + n, YbottomRight + n


#
# def _intersection_over_union(box1, box2):
#     x1, y1, w1, h1 = change_to_width_len_format(box1)
#     x2, y2, w2, h2 = change_to_width_len_format(box2)
#     w_intersection = min(x1 + w1, x2 + w2) - max(x1, x2)
#     h_intersection = min(y1 + h1, y2 + h2) - max(y1, y2)
#     if w_intersection <= 0 or h_intersection <= 0:  # No overlap
#         return 0
#     I = w_intersection * h_intersection
#     U = w1 * h1 + w2 * h2 - I  # Union = Total Area - I
#     return I / U


# def change_to_width_len_format(box):
#     XtopLeft, YtopLeft = int(box[1]), int(box[2])
#     XbottomRight, YbottomRight = int(box[3]), int(box[4])
#     X_len = abs(XbottomRight - XtopLeft)
#     Y_len = abs(YbottomRight - YtopLeft)
#     return XtopLeft, YtopLeft, X_len, Y_len


def generate_inter_matrix(ids, pred):
    allowed_location_names = [2, 3, 4, 5, 6, 7, 8]
    df = pred[np.in1d(pred[:, 0], ids) & np.in1d(pred[:, 2], ids) & np.in1d(pred[:, 4], allowed_location_names)]
    mx = np.amax(ids)
    intersection_mtx = np.zeros((mx + 1, mx + 1))
    for row in df:
        intersection_mtx[int(row[0]), int(row[2])] = row[1]
        intersection_mtx[int(row[2]), int(row[0])] = row[1]
    return intersection_mtx


def grouping_ids(boxes, pred):
    new_b_boxes = list()

    list_of_b_boxes = functools.reduce(operator.iconcat, list(boxes.values()), [])
    new_b_boxes.append(boxes['scene'])
    for key, value in boxes.items():
        if key is not 'scene':
            if len(value) > 1:
                key_id = attrgetter("id")
                ids = [key_id(box) for box in value]
                inter_mtx = generate_inter_matrix(ids, pred)
                separated_groups_of_b_boxes = generate_groups(inter_mtx)
                diff = list(set(ids) - set(flatten(separated_groups_of_b_boxes)))
                separated_groups_of_b_boxes.append(diff)
                if separated_groups_of_b_boxes:
                    for group in separated_groups_of_b_boxes:
                        if len(group) >= 2:
                            XtopLeft, YtopLeft, XbottomRight, YbottomRight = grouping_coordinates(
                                filtr(group, list_of_b_boxes))
                            box = Box(BoundBox(XtopLeft, YtopLeft, XbottomRight, YbottomRight), key, None, None,
                                      is_group=True)
                            new_b_boxes.append(box)
                        else:
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


def grouping(boxes_counted, pred, scene):
    new_b_boxes, new_b_labels = grouping_ids(boxes_counted, pred)
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
