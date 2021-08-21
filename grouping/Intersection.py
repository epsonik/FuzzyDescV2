import itertools

from graph import generate_groups


def grouping(b_boxes_to_merge):
    XtopLeftList, YtopLeftList = list(), list()
    XbottomRightList, YbottomRightList = list(), list()
    for box in b_boxes_to_merge:
        XtopLeftList.append(int(box[1]))
        YtopLeftList.append(int(box[2]))
        XbottomRightList.append(int(box[3]))
        YbottomRightList.append(int(box[4]))

    return min(XtopLeftList), min(YtopLeftList), max(XbottomRightList), max(YbottomRightList), b_boxes_to_merge[0][5]


def group_filter_name(names_set: set, v_boxes_temp: list):
    groups = {}
    v_boxes_new = []
    if names_set.__len__() != v_boxes_temp.__len__():
        for name in names_set:
            groups[name] = []
            for box in v_boxes_temp:
                temp_name = box[0]
                if temp_name == name:
                    groups[name].append(box)
    else:
        v_boxes_new = v_boxes_temp
    for key, value in groups.items():
        if len(value) > 1:
            is_group(key, value, v_boxes_new)
        else:
            v_boxes_new.append(value[0])
    return v_boxes_new


def is_group(key, value, v_boxes_new):
    # intersection_over_union_val = _intersection_measure(value[0], value[1])
    inter_mtx = generate_inter_matrix(value)
    separated_groups_of_b_boxes = generate_groups(inter_mtx)
    for group_idx, group in enumerate(separated_groups_of_b_boxes):
        b_boxes_to_merge = [value[i] for i in group]
        v_boxes_new.append([f'grupa {group_idx} skladajaca sie z {key}'] + list(grouping(b_boxes_to_merge)))
    return v_boxes_new


def _intersection_measure(box_a, box_b, n=15, stop_condition=3):
    intersection_over_union_val = _intersection_over_union(box_a, box_b)
    if intersection_over_union_val > 0:
        return intersection_over_union_val
    box_a_adj = box_a
    box_b_adj = box_b
    for _ in itertools.repeat(None, stop_condition):
        box_a_adj = _adj_width_height(box_a_adj, n)
        box_b_adj = _adj_width_height(box_b_adj, n)
        intersection_over_union_val = _intersection_over_union(box_a_adj, box_b_adj)
        if intersection_over_union_val > 0:
            return intersection_over_union_val
    return intersection_over_union_val


def generate_inter_matrix(b_boxes):
    intersection_mtx = []
    for boxA in b_boxes:
        intersection_row = []
        for boxB in b_boxes:
            intersection_row.append(_intersection_measure(boxA, boxB))
        intersection_mtx.append(intersection_row)
    return intersection_mtx


def _adj_width_height(box, n):
    XtopLeft, YtopLeft = int(box[1]), int(box[2])
    XbottomRight, YbottomRight = int(box[3]), int(box[4])
    return box[0], XtopLeft - n, YtopLeft - n, XbottomRight + n, YbottomRight + n


def _intersection_over_union(box1, box2):
    x1, y1, w1, h1 = change_to_width_len_format(box1)
    x2, y2, w2, h2 = change_to_width_len_format(box2)
    w_intersection = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_intersection = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_intersection <= 0 or h_intersection <= 0:  # No overlap
        return 0
    I = w_intersection * h_intersection
    U = w1 * h1 + w2 * h2 - I  # Union = Total Area - I
    return I / U


def change_to_width_len_format(box):
    XtopLeft, YtopLeft = int(box[1]), int(box[2])
    XbottomRight, YbottomRight = int(box[3]), int(box[4])
    X_len = abs(XbottomRight - XtopLeft)
    Y_len = abs(YbottomRight - YtopLeft)
    return XtopLeft, YtopLeft, X_len, Y_len
