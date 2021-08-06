import copy
import os
import re
from collections import Counter
from operator import attrgetter

import pandas as pd

# macierz FMP dla danej sceny
# arg. wej:
#  scene - dane sceny (tu mamy dane o bounding boxach)
#  fuzzy - paramtery rozmyte
#  macierz wyjsciowa:
#  wiersze - obiekty referencyjne - względem nich określamy położenie
#  kolumny - obiekty dla których określamy położenie
from YOLO.img_det import Box
from data import *
from eng.szablony_eng import *

PL = 1
EN = 0
BIERNIK = 3
NARZEDNIK = 4


def fmpm(scene, fuzzy):
    # liczba obiektow
    obj_num = scene.obj_num
    out = np.zeros((np.amax(fuzzy.fam3) + 1, obj_num, obj_num))
    for i in range(obj_num):
        for j in range(obj_num):
            out[:, i, j] = get_fuzzy_pos(scene, i, j, fuzzy)
    return out


def get_fuzzy_pos(scene, refobj, curobj, fuzzy):
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

    x_fuz = fam(fval_x_beg, fval_x_end, fuzzy.fam2)
    y_fuz = fam(fval_y_beg, fval_y_end, fuzzy.fam2)

    xy_fuz = fam(y_fuz, x_fuz, fuzzy.fam3)
    return xy_fuz


def fuzzify(val, fuz):
    len_fuz = len(fuz)
    outval = np.zeros(len_fuz)
    for i in range(len(fuz)):
        a = mf(val, fuz[i].thr)
        outval[i] = a
    return outval


# trapezoidalna funkcja przynaleznosci
# arg. wej:
# val - liczba
# fval - param.trapezoidu (a,b,c,d)
def mf(val, fval):
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


#  macierzy skojarzen rozmytych
# arg.wej.:
# fval1,fval2 - wektory wartosci rozmytych wejsciowych,
# fam_matrix - macierz skojarzen rozmytych
# wyj:
# outval - wektor wyjsciowych wartosci rozmytych
def fam(fval1, fval2, fam_matrix):
    outval = np.zeros(np.amax(fam_matrix) + 1)
    for i in range(len(fval1)):
        for j in range(len(fval2)):
            minval = min(fval1[i], fval2[j])
            if (fam_matrix[i][j] > 0) & (minval > 0):
                outval[fam_matrix[i, j]] = max(outval[fam_matrix[i, j]], minval)
    return outval


# generuje listę pradykatów wraz z ich cechami
# na podstawie niezerowych elementów FMPM
# argumenty we:
# fmpm_mat - macierz FMP
# cene - dane sceny
# fuzzy - paramtery rozmyte
# wyjście - lista predykatów/macierz: wiersze - predykaty, kolumny:
# 1 - obiekt
# 2 - istotność obiektu
# 3 - obiekt referencyjny
# 4 - istotność obiektu referencyjnego
# 5 - deskryptor 2D położenia/locus (numer)
# 6 - istotność deskryptora położenia
# 7 - deskryptor 2D orientacji/orientation (numer)
# 8 - istotność deskryptora orientcji
# 9 - wartość funkcji przynależności
# nieposortowana lista predykatów
def get_predicates(fmpm_mat, scene, fuzzy):
    # liczba możliwych relacji
    ile_rel, _, ile_ob = fmpm_mat.shape
    # wektor istotności obiektów (kryterium: wielkość):
    obj_sal = np.round(scene.obj[:, 4] * scene.obj[:, 5] / (scene.obj[0, 4] * scene.obj[0, 5]), 4)
    # korekta istotności obrazu jako całości - ustawiamy ją sztucznie jako
    # równą średniej istotności obiektu:
    obj_sal[0] = max(obj_sal[1:])
    plist = []
    for i in range(ile_ob):  # iteracja po obiektach
        for j in range(ile_ob):  # iteracja po obiektach referencyjnych
            if i != j:
                for k in range(ile_rel):
                    if fmpm_mat[k, j, i] > 0:  # niezerowe deskryptory
                        ty = k % fuzzy.lev3.maxt
                        r = int((k - ty) / fuzzy.lev3.maxt) - 1  # orientacja
                        current = np.array(
                            [i, obj_sal[i], j, obj_sal[j], ty, fuzzy.lev3.tsal[ty], r, fuzzy.lev3.osal[r],
                             fmpm_mat[k, j, i]])
                        plist.append(current)
    return np.array(plist)


def sort_predicates(pred, order):
    # liczba obiektów na obrazie

    to_sort1 = np.insert(pred, pred.shape[1], pred[:, 5], axis=1)
    to_sort = pd.DataFrame(to_sort1).sort_values(by=order, ascending=False)
    to_sort = to_sort.to_numpy()
    # print(verbalize_pred(np.array(to_sort), gtruth, fuzzy, boxes))
    return to_sort


def filter_predicates(to_sort, scene):
    ile_ob = len(scene.obj)
    obj_anchors = np.zeros((ile_ob, 1))
    pred_out = []
    obj_anchors.fill(False)
    obj_anchors[0] = True
    for i in range(len(to_sort)):
        if (not bool(obj_anchors[int(to_sort[i, 0])][0])) & bool(obj_anchors[int(to_sort[i, 2])][0]):
            obj_anchors[int(to_sort[i, 0])] = True
            c = np.append(to_sort[i, :], to_sort[i, 1])
            pred_out.append(c)
    return np.array(pred_out)


def verbalize_pred(pred, scene, fuzzy, boxes):
    zerolab = 1
    txt = ""

    for i in range(len(pred)):
        curr_pred = pred[i, :]
        ty = int(curr_pred[4])
        tname_curr = fuzzy.lev3.tname[ty]

        o = int(curr_pred[6])
        oname_curr = fuzzy.lev3.oname[o]
        first_obj_name = scene.onames[scene.obj[int(curr_pred[0]), 1]]
        second_obj_name = scene.onames[scene.obj[int(curr_pred[2]), 1]]
        obj_id_from_predicate = pred[i, 0] - zerolab
        sequence_id = get_seq_id(first_obj_name, obj_id_from_predicate, boxes)
        txt = txt.__add__(
            "{} {} object {} {} : {} {} rel. to {} {} ({})\n".format(sequence_id, i,
                                                                     obj_id_from_predicate,
                                                                     first_obj_name,
                                                                     tname_curr, oname_curr,
                                                                     pred[i, 2] - zerolab,
                                                                     second_obj_name,
                                                                     curr_pred[8]))
    return txt


def load_lang_data_pl():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pl/location.csv")
    frameworks_location = pd.read_csv(data_path, delimiter=', ', engine='python', header=None).values
    frameworks_location = dict(zip(frameworks_location[:, 0], frameworks_location[:, 1:]))

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pl/orientation.csv")
    frameworks_orientation = pd.read_csv(data_path, delimiter=', ', engine='python', header=None).values
    frameworks_orientation = dict(zip(frameworks_orientation[:, 0], frameworks_orientation[:, 1:]))

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pl/yolov3_LM.csv")
    data_multilingual_obj_names_lm = pd.read_csv(data_path, delimiter=', ', engine='python', index_col=None)

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pl/yolov3.csv")
    data_multilingual_obj_names = pd.read_csv(data_path, delimiter=', ', engine='python', index_col=None)
    return frameworks_location, frameworks_orientation, data_multilingual_obj_names, data_multilingual_obj_names_lm


def load_numerical_data_pl():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pl/liczebniki_M.csv")
    numerical_M = pd.read_csv(data_path, delimiter=', ', engine='python', index_col=None)

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pl/liczebniki_Z.csv")
    numerical_Z = pd.read_csv(data_path, delimiter=', ', engine='python', index_col=None)

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pl/liczebniki_N.csv")
    numerical_N = pd.read_csv(data_path, delimiter=', ', engine='python', index_col=None)
    numerical = {'M': numerical_M, 'Z': numerical_Z, 'N': numerical_N}
    return numerical


def load_lang_data_eng():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eng/location.csv")
    frameworks_location = pd.read_csv(data_path, delimiter=', ', engine='python', header=None).values
    frameworks_location = dict(zip(frameworks_location[:, 0], frameworks_location[:, 1:]))

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eng/orientation.csv")
    frameworks_orientation = pd.read_csv(data_path, delimiter=', ', engine='python', header=None).values
    frameworks_orientation = dict(zip(frameworks_orientation[:, 0], frameworks_orientation[:, 1:]))

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eng/yolov3.csv")
    data_multilingual_obj_names = pd.read_csv(data_path, delimiter=', ', engine='python', index_col=None)
    return frameworks_location, frameworks_orientation, data_multilingual_obj_names


def generate_preambule(data_multilingual_obj_names, data_multilingual_obj_names_lm, boxes):
    preambule = 'Na obrazie widzimy '
    for object_name in boxes.keys():
        number_of_obj_for_label = len(boxes[object_name])
        if len(boxes[object_name]) > 1:
            numerical = load_numerical_data_pl()
            obj_numericals = numerical['M']
            numerical_row = get_row(obj_numericals, number_of_obj_for_label, 'LP')
            numerical_verbal = numerical_row['LP_VERB']
            preambule = preambule.__add__(numerical_verbal)
            preambule = preambule.__add__(
                " {B}".format_map(get_row(data_multilingual_obj_names_lm, object_name)))
        else:
            preambule = preambule.__add__(
                "{B}".format_map(get_row(data_multilingual_obj_names, object_name)))
        if object_name == list(boxes.keys())[-1]:
            preambule = preambule.__add__(".")
        else:
            preambule = preambule.__add__(", ")
    return preambule.capitalize()


def verbalize_pred_pl(pred, scene, fuzzy, boxes):
    txt = ""
    frameworks_location, \
    frameworks_orientation, \
    data_multilingual_obj_names, \
    data_multilingual_obj_names_lm = load_lang_data_pl()

    preambule = generate_preambule(data_multilingual_obj_names, data_multilingual_obj_names_lm, boxes)
    txt = txt.__add__(preambule)
    txt = txt.__add__("\n")
    for i in range(len(pred)):
        curr_pred = pred[i, :]
        ty = int(curr_pred[4])
        location_name_curr = fuzzy.lev3.tname[ty]

        o = int(curr_pred[6])
        orientation_name_curr = fuzzy.lev3.oname[o]
        first_obj_name = scene.onames[scene.obj[int(curr_pred[0]), 1]]
        second_obj_name = scene.onames[scene.obj[int(curr_pred[2]), 1]]

        framework_location = frameworks_location[location_name_curr][0]
        framework_orientation = frameworks_orientation[orientation_name_curr][0]
        sentence = create_replacement(framework_location, data_multilingual_obj_names,
                                      [first_obj_name, second_obj_name], boxes,
                                      [int(curr_pred[0]), int(curr_pred[2])])
        sentence = sentence.capitalize()
        txt = txt.__add__(sentence)
        txt = txt.__add__(", ")
        txt = txt.__add__("{}".format(framework_orientation))
        txt = txt.__add__(".")
        txt = txt.__add__("\n")
    return txt


def count_ids(pred, scene):
    boxes2 = dict()

    for i in range(len(pred)):
        curr_pred = pred[i, :]
        first_obj_name = scene.onames[scene.obj[int(curr_pred[0]), 1]]
        second_obj_name = scene.onames[scene.obj[int(curr_pred[2]), 1]]

        def check_labels(object_name, obj_number):
            if object_name not in boxes2:
                boxes2[object_name] = [Box(None, object_name, 1, obj_number)]
            else:
                key_id = attrgetter("id")
                if not any(key_id(i) == obj_number for i in boxes2[object_name]):
                    boxes2[object_name].append(Box(None, object_name, len(boxes2[object_name]) + 1, obj_number))

        check_labels(first_obj_name, int(curr_pred[0]))
        check_labels(second_obj_name, int(curr_pred[2]))
    return boxes2


def find_name(data, name):
    return data[data[:, 0] == name, :][0]


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


def create_replacement(framework, data_object, resolved_obj_names_array, boxes, resolved_obj_places_array):
    regex = r'\{(.*?)\}'
    obj_places = re.findall(regex, framework)
    sentence = copy.copy(framework)
    for a_string in obj_places:
        result = a_string.split(":")
        object_case_name = result[0]
        object_place = int(result[1])
        sequence_id = get_seq_id(resolved_obj_names_array[object_place], resolved_obj_places_array[object_place], boxes)
        object_row = get_row(data_object, resolved_obj_names_array[object_place])

        sequence_id_verb_name = get_verb_numerical(sequence_id, object_row, object_case_name)
        s = "{" + a_string + "}"
        if sequence_id_verb_name is not '':
            sentence = sentence.replace(s, "{} {}".format(sequence_id_verb_name, object_row[object_case_name]))
        sentence = sentence.replace(s, object_row[object_case_name])

    return sentence


# object_case_name - case of noun
# row from data file, with noun, kind of noun and cases of it
# number of occurences of obj on image
def get_verb_numerical(sequence_id, object_row, object_case_name):
    if sequence_id is not None:
        numerical = load_numerical_data_pl()
        # femine, man, neuter rodzaj rzeczownika
        object_kind = object_row['R']
        obj_numericals = numerical[object_kind]
        numerical_row = get_row(obj_numericals, sequence_id, 'LP')
        numerical_verbal = numerical_row[object_case_name]
        return numerical_verbal
    else:
        return ""


def verbalize_pred_eng(pred, scene, fuzzy, v_labels_sequential):
    txt = ''
    return txt


def generate_description(gtruth):
    fuzzy = load_etykiety()
    fmpm_mat = fmpm(gtruth, fuzzy)

    pred = get_predicates(fmpm_mat, gtruth, fuzzy)
    to_sort = sort_predicates(pred, [1, 5, 8, 3])
    pred_filtered = filter_predicates(to_sort, gtruth)
    return pred_filtered, gtruth, fuzzy
