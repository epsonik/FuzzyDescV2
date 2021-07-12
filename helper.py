import numpy as np
import pandas as pd


def fmpm(scene, fuzzy):
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


def convert_gtruth(gTruth):
    scene = None
    return scene


def fuzzify(val, fuz):
    len_fuz = len(fuz)
    outval = np.zeros(len_fuz)
    for i in range(len(fuz)):
        a = mf(val, fuz[i].thr)
        outval[i] = a
    return outval


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


def fam(fval1, fval2, fam_matrix):
    outval = np.zeros(np.amax(fam_matrix) +1)
    for i in range(len(fval1)):
        for j in range(len(fval2)):
            minval = min(fval1[i], fval2[j])
            if (fam_matrix[i][j] > 0) & (minval > 0):
                outval[fam_matrix[i, j]] = max(outval[fam_matrix[i, j]], minval)
    return outval


def get_predicates(fmpm_mat, scene, fuzzy):
    ile_rel, _, ile_ob = fmpm_mat.shape
    obj_sal = np.round(scene.obj[:, 4] * scene.obj[:, 5] / (scene.obj[0, 4] * scene.obj[0, 5]), 4)
    obj_sal[0] = max(obj_sal[1:])
    plist = []
    for i in range(ile_ob):
        for j in range(ile_ob):
            if i != j:
                for k in range(ile_rel):
                    if fmpm_mat[k, j, i] > 0:
                        ty = k % fuzzy.lev3.maxt
                        r = int((k - ty) / fuzzy.lev3.maxt) - 1
                        current = np.array(
                            [i, obj_sal[i], j, obj_sal[j], ty, fuzzy.lev3.tsal[ty], r, fuzzy.lev3.osal[r],
                             fmpm_mat[k, j, i]])
                        plist.append(current)
    return np.array(plist)


def sort_predicates(pred, scene, fuzzy, order):
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
        o = int(curr_pred[6])
        oname_curr = fuzzy.lev3.oname[o]
        tname_curr = fuzzy.lev3.tname[ty]
        first = scene.onames[scene.obj[int(curr_pred[0]), 1]]
        second = scene.onames[scene.obj[int(curr_pred[2]), 1]]
        txt = txt.__add__(" object {} {} : {} {} rel. to {} {} ({})\n".format(curr_pred[0] - zerolab,
                                                                              first,
                                                                              tname_curr, oname_curr,
                                                                              curr_pred[2] - zerolab,
                                                                              second,
                                                                              curr_pred[9]))
    return txt


def show_bboxes(scene, showmode, currobj):
    # showmode = 0 wyswietla same BB
    # = 1 wyswietla obraz orygialny i BB
    # = 2 wyswietla wycięte BB (tło - rozmyty obr. pocz.)
    # = 3 wyswietla wycięte BB (tło białe)
    labels = np.zeros((scene.obj_num[0], 1), np.uint8)
    if scene.im is not None:
        showmode = 0
    if showmode == 0:
        im = np.ones((scene.size[0], scene.size[1], 3), np.uint8)
    elif showmode == 1:
        im = scene.im
    # else:
    #     if showmode == 2:
    #         im = scene.background
    #     else:
    #         im = scene.background2
    #     ordered_obj = 1:scene.obj_num
    #     if currobj > 0
    #         ordered_obj(scene.obj_num) = currobj
    #         ordered_obj(currobj) = scene.obj_num
    #     for i=2:scene.obj_num
    #         objnum = ordered_obj(i)
    #         upleft = [scene.obj(objnum,4) scene.obj(objnum,3)];
    #         downright = upleft + [scene.obj(objnum,6)-1 scene.obj(objnum,5)-1]
    #         rescalled = imresize(scene.images{objnum}, [scene.obj(objnum,6) scene.obj(objnum,5)])
    #         rescalled = rescalled(1:scene.obj(objnum,6),1:scene.obj(objnum,5),:)
    #         im(upleft(1):downright(1),upleft(2):downright(2),:) = rescalled
    # colors = []
    # for i in range(scene.obj_num):
    #     labels{i} = [num2str(i-1) ':' char(scene.onames(scene.obj(i,2),:)) ];
    #     colors = [colors; scene.ocols(scene.obj(i,2),:)*255];
    # im = insertObjectAnnotation(im,'rectangle',scene.obj(:,3:6),labels,'Color',colors ,...
    #       'TextBoxOpacity',0.7,'FontSize',max(floor(min(scene.size)/40),8),'LineWidth',3);
    return None
