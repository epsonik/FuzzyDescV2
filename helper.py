import os

import numpy as np
import pandas as pd

# macierz FMP dla danej sceny
# arg. wej:
#  scene - dane sceny (tu mamy dane o bounding boxach)
#  fuzzy - paramtery rozmyte
#  macierz wyjsciowa:
#  wiersze - obiekty referencyjne - względem nich określamy położenie
#  kolumny - obiekty dla których określamy położenie
from YOLO.img_det import draw_boxes
from data import *

LANGUAGE = 1
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


def sort_predicates(pred, scene, fuzzy, order):
    # liczba obiektów na obrazie
    ile_ob = len(scene.obj)
    obj_anchors = np.zeros((ile_ob, 1))
    pred_out = []
    obj_anchors.fill(False)
    obj_anchors[0] = True
    to_sort1 = np.insert(pred, pred.shape[1], pred[:, 5], axis=1)
    to_sort = pd.DataFrame(to_sort1).sort_values(by=order, ascending=False)
    to_sort = to_sort.to_numpy()
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
        txt = txt.__add__(" object {} {} : {} {} rel. to {} {} ({})\n".format(pred[i, 0] - zerolab,
                                                                              first_obj_name,
                                                                              tname_curr, oname_curr,
                                                                              pred[i, 2] - zerolab,
                                                                              second_obj_name,
                                                                              curr_pred[9]))
    return txt


stykac = "{} styka się z {}"
nachodzic = "{} nachodzi na {}"
blisko = "{} jest blisko {}"


def verbalize_pred_pl(pred, scene, fuzzy):
    zerolab = 1
    txt = ""
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov3_PL.csv")
    data_multilingual_obj_names = pd.read_csv(data_path, delimiter=', ', engine='python').values

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "texts_odmiana.csv")
    data_multilingual_positions = pd.read_csv(data_path, delimiter=', ', engine='python').values

    def find_name(data, name):
        return data[data[:, 0] == name, :][0]

    for i in range(len(pred)):
        curr_pred = pred[i, :]
        ty = int(curr_pred[4])
        tname_curr = fuzzy.lev3.tname[ty]

        o = int(curr_pred[6])
        oname_curr = fuzzy.lev3.oname[o]
        first_obj_name = scene.onames[scene.obj[int(curr_pred[0]), 1]]
        second_obj_name = scene.onames[scene.obj[int(curr_pred[2]), 1]]
        if second_obj_name != "scene":
            if (tname_curr is TOUCHING):
                # txt = txt.__add__("{} {} {} {}\n".format(find_name(data_multilingual_obj_names, first_obj_name)[LANGUAGE],
                #                                        find_name(data_multilingual_positions, tname_curr)[LANGUAGE],
                #                                        find_name(data_multilingual_obj_names, second_obj_name)[LANGUAGE],
                #                                        find_name(data_multilingual_positions, oname_curr)[LANGUAGE]))
                txt = txt.__add__(stykac.format(find_name(data_multilingual_obj_names, second_obj_name)[LANGUAGE],
                                                find_name(data_multilingual_obj_names, first_obj_name)[NARZEDNIK]))
                if (oname_curr == ABOVE):
                    txt = txt.__add__(" od góry")
                if (oname_curr == LEFT_ABOVE):
                    txt = txt.__add__(" w lewym górnym rogu")
                if (oname_curr == RIGHT_ABOVE):
                    txt = txt.__add__(" w prawym górnym rogu")
                if (oname_curr == BELOW):
                    txt = txt.__add__(" od dołu")
                if (oname_curr == LEFT_BELOW):
                    txt = txt.__add__(" w lewym dolnym rogu")
                if (oname_curr == RIGHT_BELOW):
                    txt = txt.__add__(" w prawym dolnym rogu")
                if (oname_curr == LEFT):
                    txt = txt.__add__(" po lewej")
                if (oname_curr == RIGHT):
                    txt = txt.__add__(" po prawej")
            if (tname_curr is CROSSING):
                txt = txt.__add__(nachodzic.format(find_name(data_multilingual_obj_names, second_obj_name)[LANGUAGE],
                                                   find_name(data_multilingual_obj_names, first_obj_name)[BIERNIK]))
                if (oname_curr == ABOVE):
                    txt = txt.__add__(" od góry")
                if (oname_curr == LEFT_ABOVE):
                    txt.__add__(" w lewym górnym rogu")
                if (oname_curr == RIGHT_ABOVE):
                    txt = txt.__add__(" w prawym górnym rogu")
                if (oname_curr == BELOW):
                    txt.__add__(" od dołu")
                if (oname_curr == LEFT_BELOW):
                    txt = txt.__add__(" w lewym dolnym rogu")
                if (oname_curr == RIGHT_BELOW):
                    txt = txt.__add__(" w prawym dolnym rogu")
                if (oname_curr == LEFT):
                    txt = txt.__add__(" po lewej")
                if (oname_curr == RIGHT):
                    txt = txt.__add__(" po prawej")
            if (tname_curr is CLOSE):
                txt = txt.__add__(nachodzic.format(find_name(data_multilingual_obj_names, second_obj_name)[LANGUAGE],
                                                   find_name(data_multilingual_obj_names, first_obj_name)[BIERNIK]))
                if (oname_curr == ABOVE):
                    txt = txt.__add__(" od góry")
                if (oname_curr == LEFT_ABOVE):
                    txt.__add__(" w lewym górnym rogu")
                if (oname_curr == RIGHT_ABOVE):
                    txt = txt.__add__(" w prawym górnym rogu")
                if (oname_curr == BELOW):
                    txt.__add__(" od dołu")
                if (oname_curr == LEFT_BELOW):
                    txt = txt.__add__(" w lewym dolnym rogu")
                if (oname_curr == RIGHT_BELOW):
                    txt = txt.__add__(" w prawym dolnym rogu")
                if (oname_curr == LEFT):
                    txt = txt.__add__(" po lewej")
                if (oname_curr == RIGHT):
                    txt = txt.__add__(" po prawej")
            txt = txt.__add__("\n")
    return txt


def verbalize_pred_eng(pred, scene, fuzzy):
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
        txt = txt.__add__(" object {} {} : {} {} rel. to {} {} ({})\n".format(pred[i, 0] - zerolab,
                                                                              first_obj_name,
                                                                              tname_curr, oname_curr,
                                                                              pred[i, 2] - zerolab,
                                                                              second_obj_name,
                                                                              curr_pred[9]))
    return txt


def generate_description(gtruth):
    fuzzy = load_etykiety()
    fmpm_mat = fmpm(gtruth, fuzzy)

    pred = get_predicates(fmpm_mat, gtruth, fuzzy)
    pred_sort = sort_predicates(pred, gtruth, fuzzy, [1, 5, 8, 3])
    return pred_sort, gtruth, fuzzy
