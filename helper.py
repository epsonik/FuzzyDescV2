from operator import attrgetter
import random

import pandas as pd
from data import *

PL = 1
EN = 0
BIERNIK = 3
NARZEDNIK = 4
fuzzy = load_etykiety()


def get_location_names_indexes(location_names_txt):
    return [fuzzy.lev3.location_names.index(location_name) for location_name in location_names_txt]


# macierz FMP dla danej sceny
# arg. wej:
#  scene - dane sceny (tu mamy dane o bounding boxach)
#  fuzzy - paramtery rozmyte
#  macierz wyjsciowa:
#   wiersze - obiekty referencyjne - względem nich określamy położenie
#   kolumny - obiekty dla których określamy położenie
def create_fuzzy_mutual_posions_matrix(scene):
    # liczba obiektow
    obj_num = scene.obj_num
    out = np.zeros((np.amax(fuzzy.fam3) + 1, obj_num, obj_num))
    for i in range(obj_num):
        for j in range(obj_num):
            out[:, i, j] = get_fuzzy_pos(scene, i, j)
    return out


def get_fuzzy_pos(scene, refobj, curobj):
    def fuzzify(val, fuz):
        # trapezoidalna funkcja przynaleznosci
        # arg. wej:
        # val - liczba
        # fval - param.trapezoidu (a,b,c,d)
        def trapezoid_membership_function(val, fval):
            if (val < fval[0]):
                out = 0
            elif (val < fval[1]):
                out = (val - fval[0]) / (fval[1] - fval[0])
            elif (val < fval[2]):
                out = 1
            elif (val < fval[3]):
                out = (fval[3] - val) / (fval[3] - fval[2])
            else:
                out = 0
            return float("{:.4f}".format(out))

        len_fuz = len(fuz)
        outval = np.zeros(len_fuz)
        for i in range(len(fuz)):
            a = trapezoid_membership_function(val, fuz[i].thr)
            outval[i] = a
        return outval

    #  macierzy skojarzen rozmytych
    # arg.wej.:
    # fval1,fval2 - wektory wartosci rozmytych wejsciowych,
    # fam_matrix - macierz skojarzen rozmytych
    # wyj:
    # outval - wektor wyjsciowych wartosci rozmytych
    def create_fuzzy_association_matrix(fval1, fval2, fam_matrix):
        outval = np.zeros(np.amax(fam_matrix) + 1)
        for i in range(len(fval1)):
            for j in range(len(fval2)):
                minval = min(fval1[i], fval2[j])
                if (fam_matrix[i][j] > 0) & (minval > 0):
                    outval[fam_matrix[i, j]] = max(outval[fam_matrix[i, j]], minval)
        return outval

    x_beg_ref = scene.obj[refobj][2]
    y_beg_ref = scene.obj[refobj][3]
    x_end_ref = scene.obj[refobj][2] + scene.obj[refobj][4]
    y_end_ref = scene.obj[refobj][3] + scene.obj[refobj][5]

    x_beg_cur = scene.obj[curobj][2]
    y_beg_cur = scene.obj[curobj][3]
    x_end_cur = scene.obj[curobj][2] + scene.obj[curobj][4]
    y_end_cur = scene.obj[curobj][3] + scene.obj[curobj][5]

    x_beg = (2 * x_beg_cur - (x_beg_ref + x_end_ref)) / (x_end_ref - x_beg_ref)
    x_end = (2 * x_end_cur - (x_beg_ref + x_end_ref)) / (x_end_ref - x_beg_ref)

    y_beg = (2 * y_beg_cur - (y_beg_ref + y_end_ref)) / (y_end_ref - y_beg_ref)
    y_end = (2 * y_end_cur - (y_beg_ref + y_end_ref)) / (y_end_ref - y_beg_ref)

    fval_x_beg = fuzzify(x_beg, fuzzy.lev1)
    fval_x_end = fuzzify(x_end, fuzzy.lev1)

    fval_y_beg = fuzzify(y_beg, fuzzy.lev1)
    fval_y_end = fuzzify(y_end, fuzzy.lev1)

    x_fuz = create_fuzzy_association_matrix(fval_x_beg, fval_x_end, fuzzy.fam2)
    y_fuz = create_fuzzy_association_matrix(fval_y_beg, fval_y_end, fuzzy.fam2)

    xy_fuz = create_fuzzy_association_matrix(y_fuz, x_fuz, fuzzy.fam3)
    return xy_fuz


# generuje listę pradykatów wraz z ich cechami
# na podstawie niezerowych elementów FMPM
# argumenty we:
# fmpm_mat - macierz FMP
# cene - dane sceny
# fuzzy - paramtery rozmyte
# wyjście - lista predykatów/macierz: wiersze - predykaty, kolumny:
# 0 - obiekt
# 1 - istotność obiektu
# 2 - obiekt referencyjny
# 3 - istotność obiektu referencyjnego
# 4 - deskryptor 2D położenia/locus (numer)
# 5 - istotność deskryptora położenia
# 6 - deskryptor 2D orientacji/orientation (numer)
# 7 - istotność deskryptora orientcji
# 8 - wartość funkcji przynależności
# nieposortowana lista predykatów
def get_predicates(fmpm_mat, scene):
    # liczba możliwych relacji
    ile_rel, _, ile_ob = fmpm_mat.shape
    # wektor istotności obiektów (kryterium: wielkość):
    obj_width = scene.obj[:, 4]
    obj_height = scene.obj[:, 5]
    scene_width = scene.obj[0, 4]
    scene_height = scene.obj[0, 5]
    obj_sal = np.round(obj_width * obj_height / (scene_width * scene_height), 4)
    # korekta istotności obrazu jako całości - ustawiamy ją sztucznie jako
    # równą średniej istotności obiektu:
    obj_sal[0] = max(obj_sal[1:])
    plist = []
    for first_obj in range(ile_ob):  # iteracja po obiektach
        for second_obj in range(ile_ob):  # iteracja po obiektach referencyjnych
            if first_obj != second_obj:
                for k in range(ile_rel):
                    if fmpm_mat[k, second_obj, first_obj] > 0:  # niezerowe deskryptory
                        ty = k % fuzzy.lev3.maxt
                        r = int((k - ty) / fuzzy.lev3.maxt) - 1  # orientacja
                        sal_fist_obj = obj_sal[first_obj]
                        sal_second_obj = obj_sal[second_obj]
                        fuzzy_mutual_position_matrix = fmpm_mat[k, second_obj, first_obj]
                        current = np.array(
                            [first_obj, sal_fist_obj, second_obj, sal_second_obj, ty, fuzzy.lev3.location_sal[ty], r,
                             fuzzy.lev3.orientation_sal[r],
                             fuzzy_mutual_position_matrix])
                        plist.append(current)
    return np.array(plist)


def sort_predicates(pred, order):
    # liczba obiektów na obrazie

    to_sort1 = np.insert(pred, pred.shape[1], pred[:, 5], axis=1)
    to_sort = pd.DataFrame(to_sort1).sort_values(by=order, ascending=False)
    to_sort = to_sort.to_numpy()
    return to_sort


def filter_predicates(to_sort, scene):
    ile_ob = len(scene.obj)
    obj_anchors = np.zeros((ile_ob, 1))
    pred_out = []
    obj_anchors.fill(False)
    obj_anchors[0] = True
    for i in range(len(to_sort)):
        first_object = int(to_sort[i, 0])
        second_object = int(to_sort[i, 2])
        if (not bool(obj_anchors[first_object][0])) & bool(obj_anchors[second_object][0]):
            obj_anchors[first_object] = True
            c = np.append(to_sort[i, :], to_sort[i, 1])
            pred_out.append(c)
    return np.array(pred_out)


def verbalize_pred(pred, scene):
    zerolab = 1
    txt = ""

    for i in range(len(pred)):
        curr_pred = pred[i, :]
        location_number = int(curr_pred[4])
        location_name_curr = fuzzy.lev3.location_names[location_number]
        orientation_number = int(curr_pred[6])
        orientation_name_curr = fuzzy.lev3.orientation_name[orientation_number]
        first_obj_name = scene.onames[scene.obj[int(curr_pred[0]), 1]]
        second_obj_name = scene.onames[scene.obj[int(curr_pred[2]), 1]]
        obj_id_from_predicate = pred[i, 0]
        txt = txt.__add__(
            "object {} {} : {} {} rel. to {} {} ({})\n".format(obj_id_from_predicate,
                                                               first_obj_name,
                                                               location_name_curr, orientation_name_curr,
                                                               pred[i, 2] - zerolab,
                                                               second_obj_name,
                                                               curr_pred[8]))
    return txt


def dot_or_comma(object_name, obj_list):
    if object_name == list(obj_list.keys())[-1]:
        return "."
    else:
        if object_name == list(obj_list.keys())[-2]:
            return " i "
        return ", "


# get random framework for sentence
def random_framework(framework):
    return random.choice(framework.split("*"))


# create dictionary
# label_A:{bounding_box_1_with_label_A, bounding_box_2_with_label_A....}
def count_ids(pred, scene, v_boxes):
    from YOLO.img_det import Box
    boxes2 = dict()

    for i in range(len(pred)):
        curr_pred = pred[i, :]
        first_obj_name = scene.onames[scene.obj[int(curr_pred[0]), 1]]
        second_obj_name = scene.onames[scene.obj[int(curr_pred[2]), 1]]

        def check_labels(object_name, obj_number):
            if object_name not in boxes2:
                boxes2[object_name] = [Box(v_boxes[obj_number], object_name, 1, obj_number)]
            else:
                key_id = attrgetter("id")
                if not any(key_id(i) == obj_number for i in boxes2[object_name]):
                    # Box(
                    # self.box = box
                    # self.label = label -label of bounding box
                    # self.seq_id = seq_id - id of bounding box in sequnece of other bounding boxes with the same label
                    # self.id = id - set id of bounding box from predicates table
                    # self.is_group = is_group - check if bounding box is group (grouping mechanism).
                    # We store info if bounding box is a product of grouping other bounding boxes
                    # )
                    boxes2[object_name].append(
                        Box(v_boxes[obj_number], object_name, len(boxes2[object_name]) + 1, obj_number))

        check_labels(first_obj_name, int(curr_pred[0]))
        check_labels(second_obj_name, int(curr_pred[2]))
    return boxes2


# group boxes but for grouping task
# pred - newly generated predicates for groups
def count_ids_g(pred, scene, v_boxes):
    boxes_with_order_numbers_sep = dict()
    boxes_with_order_numbers = {}
    for i in range(len(pred)):
        curr_pred = pred[i, :]
        first_obj_name = scene.onames[scene.obj[int(curr_pred[0]), 1]]
        second_obj_name = scene.onames[scene.obj[int(curr_pred[2]), 1]]

        # for each label we check if bounding box is single or is group
        def check_labels(object_name, obj_number):
            b = v_boxes[obj_number]
            if object_name not in boxes_with_order_numbers_sep:
                b.seq_id = 1
                boxes_with_order_numbers_sep[object_name] = {'group': [], 'single': []}
                if b.is_group:
                    boxes_with_order_numbers_sep[object_name]['group'].append(b)
                else:
                    boxes_with_order_numbers_sep[object_name]['single'].append(b)
            else:
                def p(a):
                    key_id = attrgetter("id")
                    if not any(key_id(i) == obj_number for i in a):
                        b.seq_id = len(a) + 1
                        a.append(b)

                if b.is_group:
                    p(boxes_with_order_numbers_sep[object_name]['group'])
                else:
                    p(boxes_with_order_numbers_sep[object_name]['single'])

        check_labels(first_obj_name, int(curr_pred[0]))
        check_labels(second_obj_name, int(curr_pred[2]))
    for key in boxes_with_order_numbers_sep.keys():
        boxes_with_order_numbers[key] = list(
            boxes_with_order_numbers_sep[key]['single'] + boxes_with_order_numbers_sep[key]['group'])
    return boxes_with_order_numbers, boxes_with_order_numbers_sep


def find_name(data, name):
    return data[data[:, 0] == name, :][0]


# get row from file with language data for specific word
def get_row(data, name, key='ENG'):
    return data.iloc[
        data.index[data[key] == name]].to_dict('records')[0]


def get_seq_id(obj_name, id_from_predicate, boxes):
    key_id = attrgetter("id")
    key_seq_id = attrgetter("seq_id")
    if obj_name is not "scene":
        boxes_for_label = boxes[obj_name]
        if len(boxes_for_label) > 1:
            for box in boxes_for_label:
                if key_id(box) == int(id_from_predicate):
                    return key_seq_id(box)
    return None


def get_box(obj_name, id_from_predicate, boxes):
    key_id = attrgetter("id")
    boxes_for_label = boxes[obj_name]
    for box in boxes_for_label:
        if key_id(box) == int(id_from_predicate):
            return box


def generate_description(gtruth):
    fmpm_mat = create_fuzzy_mutual_posions_matrix(gtruth)

    pred = get_predicates(fmpm_mat, gtruth)
    to_sort = sort_predicates(pred, [1, 5, 8, 3])
    pred_filtered = filter_predicates(to_sort, gtruth)
    return pred_filtered, pred
