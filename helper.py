import copy
import os
import re
from collections import Counter

import pandas as pd

# macierz FMP dla danej sceny
# arg. wej:
#  scene - dane sceny (tu mamy dane o bounding boxach)
#  fuzzy - paramtery rozmyte
#  macierz wyjsciowa:
#  wiersze - obiekty referencyjne - względem nich określamy położenie
#  kolumny - obiekty dla których określamy położenie
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


def sort_predicates(pred, scene, fuzzy, order, gtruth):
    # liczba obiektów na obrazie

    to_sort1 = np.insert(pred, pred.shape[1], pred[:, 5], axis=1)
    to_sort = pd.DataFrame(to_sort1).sort_values(by=order, ascending=False)
    to_sort = to_sort.to_numpy()
    # print(verbalize_pred(np.array(to_sort), gtruth, fuzzy))
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


def verbalize_pred(pred, scene, fuzzy):
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
        txt = txt.__add__("{} object {} {} : {} {} rel. to {} {} ({})\n".format(i, pred[i, 0] - zerolab,
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


def load_lang_data_eng():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eng/location.csv")
    frameworks_location = pd.read_csv(data_path, delimiter=', ', engine='python', header=None).values
    frameworks_location = dict(zip(frameworks_location[:, 0], frameworks_location[:, 1:]))

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eng/orientation.csv")
    frameworks_orientation = pd.read_csv(data_path, delimiter=', ', engine='python', header=None).values
    frameworks_orientation = dict(zip(frameworks_orientation[:, 0], frameworks_orientation[:, 1:]))

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eng/yolov3_LM.csv")
    data_multilingual_obj_names_lm = pd.read_csv(data_path, delimiter=', ', engine='python', index_col=None)

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eng/yolov3.csv")
    data_multilingual_obj_names = pd.read_csv(data_path, delimiter=', ', engine='python', index_col=None)
    return frameworks_location, frameworks_orientation, data_multilingual_obj_names, data_multilingual_obj_names_lm


def verbalize_pred_pl(pred, scene, fuzzy, v_labels_sequential):
    gen_desc = "Na obrazie widzimy "
    zerolab = 1
    txt = gen_desc
    # data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pl/location_pl.csv")
    # location = pd.read_csv(data_path, delimiter=', ', engine='python').values
    frameworks_location, \
    frameworks_orientation, \
    data_multilingual_obj_names, \
    data_multilingual_obj_names_lm = load_lang_data_pl()
    image_labels_counter = Counter(v_labels_sequential)

    def generate_preambule():
        preambule = ''
        for object_name in image_labels_counter.keys():
            if image_labels_counter[object_name] > 1:
                preambule = preambule.__add__(
                    " {B}, ".format_map(get_row(data_multilingual_obj_names_lm, object_name)))
            else:
                preambule = preambule.__add__(
                    " {B}, ".format_map(get_row(data_multilingual_obj_names, object_name)))
        return preambule

    preambule = generate_preambule()
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
                                      [first_obj_name, second_obj_name])
        txt = txt.__add__(sentence)
        txt = txt.__add__(", ")
        txt = txt.__add__("{}".format(framework_orientation))
        txt = txt.__add__("\n")
    return txt


def find_name(data, name):
    return data[data[:, 0] == name, :][0]


def get_row(data, name):
    return data.iloc[
        data.index[data['ENG'] == name]].to_dict('records')[0]


def create_replacement(framework, data_multilingual_obj_names, predicate_ref_obj):
    regex = r'\{(.*?)\}'
    obj_places = re.findall(regex, framework)
    sentence = copy.copy(framework)
    for a_string in obj_places:
        result = a_string.split(":")
        object_case_name = result[0]
        object_place = int(result[1])
        object_case = get_row(data_multilingual_obj_names, predicate_ref_obj[object_place])
        s = "{" + a_string + "}"
        sentence = sentence.replace(s, object_case[object_case_name])
    return sentence


def verbalize_pred_eng(pred, scene, fuzzy, v_labels_sequential):
    zerolab = 1
    txt = gen_desc
    frameworks_location, \
    frameworks_orientation, \
    data_multilingual_obj_names, \
    data_multilingual_obj_names_lm = load_lang_data_eng()

    image = Counter(v_labels_sequential)
    for object_name in image.keys():
        if image[object_name] > 1:
            txt = txt.__add__("{}, ".format(find_name(data_multilingual_obj_names_lm, object_name)[0] + "s"))
        else:
            txt = txt.__add__("{}, ".format(find_name(data_multilingual_obj_names, object_name)[0]))

    txt = txt.__add__("\n")

    for i in range(len(pred)):
        curr_pred = pred[i, :]
        ty = int(curr_pred[4])
        tname_curr = fuzzy.lev3.tname[ty]

        o = int(curr_pred[6])
        oname_curr = fuzzy.lev3.oname[o]
        first_obj_name = scene.onames[scene.obj[int(curr_pred[0]), 1]]
        second_obj_name = scene.onames[scene.obj[int(curr_pred[2]), 1]]
        if (tname_curr is TOUCHING):
            txt = txt.__add__(
                touching.format(pred[i, 2] - zerolab, find_name(data_multilingual_obj_names, second_obj_name)[EN],
                                pred[i, 0] - zerolab, find_name(data_multilingual_obj_names, first_obj_name)[EN]))
            if (oname_curr == ABOVE):
                txt = txt.__add__(" from above")
            if (oname_curr == LEFT_ABOVE):
                txt = txt.__add__(" on the left upper corner")
            if (oname_curr == RIGHT_ABOVE):
                txt = txt.__add__(" on the right upper corner")
            if (oname_curr == BELOW):
                txt = txt.__add__(" below")
            if (oname_curr == LEFT_BELOW):
                txt = txt.__add__(" on the left below corneru")
            if (oname_curr == RIGHT_BELOW):
                txt = txt.__add__(" on the right below corner")
            if (oname_curr == LEFT):
                txt = txt.__add__(" on the left")
            if (oname_curr == RIGHT):
                txt = txt.__add__(" on the right")
        if (tname_curr is CROSSING):
            txt = txt.__add__(
                crossing.format(pred[i, 2] - zerolab, find_name(data_multilingual_obj_names, second_obj_name)[EN],
                                pred[i, 0] - zerolab, find_name(data_multilingual_obj_names, first_obj_name)[EN]))
            if (oname_curr == ABOVE):
                txt = txt.__add__(" from above")
            if (oname_curr == LEFT_ABOVE):
                txt = txt.__add__(" on the left upper corner")
            if (oname_curr == RIGHT_ABOVE):
                txt = txt.__add__(" on the right upper corner")
            if (oname_curr == BELOW):
                txt = txt.__add__(" below")
            if (oname_curr == LEFT_BELOW):
                txt = txt.__add__(" on the left below corner")
            if (oname_curr == RIGHT_BELOW):
                txt = txt.__add__(" on the right below corner")
            if (oname_curr == LEFT):
                txt = txt.__add__(" on the left")
            if (oname_curr == RIGHT):
                txt = txt.__add__(" on the right")
        if (tname_curr is CLOSE):
            txt = txt.__add__(
                crossing.format(pred[i, 2] - zerolab, find_name(data_multilingual_obj_names, second_obj_name)[EN],
                                pred[i, 0] - zerolab, find_name(data_multilingual_obj_names, first_obj_name)[EN]))
            if (oname_curr == ABOVE):
                txt = txt.__add__(" from above")
            if (oname_curr == LEFT_ABOVE):
                txt = txt.__add__(" on the left upper corner")
            if (oname_curr == RIGHT_ABOVE):
                txt = txt.__add__(" on the right upper corner")
            if (oname_curr == BELOW):
                txt = txt.__add__(" below")
            if (oname_curr == LEFT_BELOW):
                txt = txt.__add__(" on the left below corner")
            if (oname_curr == RIGHT_BELOW):
                txt = txt.__add__(" on the right below corner")
            if (oname_curr == LEFT):
                txt = txt.__add__(" on the left")
            if (oname_curr == RIGHT):
                txt = txt.__add__(" on the right")
        txt = txt.__add__("\n")
    return txt


def generate_description(gtruth):
    fuzzy = load_etykiety()
    fmpm_mat = fmpm(gtruth, fuzzy)

    pred = get_predicates(fmpm_mat, gtruth, fuzzy)
    to_sort = sort_predicates(pred, gtruth, fuzzy, [1, 5, 8, 3], gtruth)
    pred_filtered = filter_predicates(to_sort, gtruth)
    return pred_filtered, gtruth, fuzzy
