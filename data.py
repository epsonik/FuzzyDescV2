from Lev1 import Lev1, Type
from Lev2 import Lev2
from Lev3 import Lev3
import numpy as np

marg = 0.2
# daleko
inf = 100
# próg bliskości drugiego obiektu
near = 2
# granica wykrywania
far = 4


# lev1 estymacja pozycji rozmytych

def load_etykiety():
    class Fuzzy:
        pass

    fuzzy = Fuzzy()
    lev1 = {}
    lev1[0] = Lev1(name=["far left", "far above"], type=Type.F, variant=[Type.L, Type.A],
                   thr=[-inf, -inf, -far - marg, -far + marg])

    lev1[1] = Lev1(name=["near left", "near above"], type=Type.N, variant=[Type.L, Type.A],
                   thr=[-far - marg, -far + marg, -near - marg, -near + marg])

    lev1[2] = Lev1(
        name=["close left", "close above"],
        type=Type.C,
        variant=[Type.L, Type.A],
        thr=[-near - marg, -near + marg, -1 - 2 * marg, -1])

    lev1[3] = Lev1(

        name=["edge left", "edge above"],

        type="E",

        variant=["L", "A"],

        thr=[-1 - 2 * marg, -1, -1, -1 + 2 * marg])
    lev1[4] = Lev1(
        name=["inside left", "inside above"],
        type="I",
        variant=["L", "A"],
        thr=[-1, -1 + 2 * marg, -marg, marg])

    lev1[5] = Lev1(
        name=["inside right", "inside below"],

        type="I",

        variant=["R", "B"],

        thr=[-marg, marg, 1 - 2 * marg, 1])

    lev1[6] = Lev1(
        name=["edge right", "edge below"],
        type="E",
        variant=["R", "B"],
        thr=[1 - 2 * marg, 1, 1, 1 + 2 * marg])

    lev1[7] = Lev1(
        name=["close right", "close below"],
        type="C",
        variant=["R", "B"],
        thr=[1, 1 + 2 * marg, near - marg, near + marg])

    lev1[8] = Lev1(
        name=["near right", "near below"],
        type="N",
        variant=["R", "B"],

        thr=[near - marg, near + marg, far - marg, far + marg])
    lev1[9] = Lev1(
        name=["far right", "far below"],
        type="F",
        variant=["R", "B"],
        thr=[far - marg, far + marg, inf, inf])

    fuzzy.lev1 = lev1

    # estymacja położenia rozmytego  obiektu wzdłuż jednej osi (1D)
    level2List = {}
    level2List[0] = Lev2(
        name="far",
        type="FA",
        variant=["L", "A"])

    level2List[1] = Lev2(
        name="near",
        type="NE",
        variant=["L", "A"])

    level2List[2] = Lev2(
        name="close",
        type="CL",
        variant=["L", "A"])

    level2List[3] = Lev2(
        name="touching",
        type="TO",
        variant=["L", "A"])

    level2List[4] = Lev2(
        name="crossing",
        type="CR",
        variant=["L", "A"])

    level2List[5] = Lev2(
        name="inside",
        type="IN",
        variant=["L", "A"])

    level2List[6] = Lev2(
        name="shorter",
        type="SH",
        variant=["LR", "AB"])
    level2List[7] = Lev2(
        name="same",
        type="SM",
        variant=["LR", "AB"])
    level2List[8] = Lev2(
        name="longer",
        type="LO",
        variant=["LR", "AB"])

    level2List[9] = Lev2(
        name="inside",
        type="IN",
        variant=["R", "B"])
    level2List[10] = Lev2(
        name="crossing",
        type="CR",
        variant=["R", "B"])
    level2List[11] = Lev2(
        name="touching",
        type="TO",
        variant=["R", "B"])
    level2List[12] = Lev2(
        name="close",
        type="CL",
        variant=["R", "B"])

    level2List[13] = Lev2(
        name="near",
        type="NE",
        variant=["R", "B"])

    level2List[14] = Lev2(
        name="far",
        type="FA",
        variant=["R", "B"])

    fuzzy.level2List = level2List
    fa_la = 0
    ne_la = 1
    cl_la = 2
    to_la = 3
    cr_la = 4
    in_la = 5
    sh = 6
    sa = 7
    lo = 8
    in_rb = 9
    cr_rb = 10
    to_rb = 11
    cl_rb = 12
    ne_rb = 13
    fa_rb = 14

    fam2 = np.array([[fa_la, ne_la, cl_la, to_la, cr_la, cr_la, cr_la, lo, lo, lo],
                     [0, ne_la, cl_la, to_la, cr_la, cr_la, cr_la, lo, lo, lo],
                     [0, 0, cl_la, to_la, cr_la, cr_la, cr_la, lo, lo, lo],
                     [0, 0, 0, to_la, in_la, in_la, sa, cr_rb, cr_rb, cr_rb],
                     [0, 0, 0, 0, in_la, sh, in_rb, cr_rb, cr_rb, cr_rb],
                     [0, 0, 0, 0, 0, in_rb, in_rb, cr_rb, cr_rb, cr_rb],
                     [0, 0, 0, 0, 0, 0, to_rb, to_rb, to_rb, to_rb],
                     [0, 0, 0, 0, 0, 0, 0, cl_rb, cl_rb, cl_rb],
                     [0, 0, 0, 0, 0, 0, 0, 0, ne_rb, ne_rb],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, fa_rb]])
    fuzzy.fam2 = fam2
    # poziom 3 - estymacja rozmyta położenia obiektu w przestrzeni 2D
    # deskryptory pozycji
    fa = 0
    ne = 1
    cl = 2
    to = 3
    cr = 4
    in_par = 5
    lg = 6
    sp = 7
    sa = 8

    lev3 = Lev3(maxt=10,
                type=['FA', 'NE', 'CL', 'TO', 'CR', 'IN', 'LG', 'SP', 'SA'],
                tsal=[0.6, 0.7, 0.8, 0.9, 0.9, 0.8, 0.3, 0.9, 1],
                tname=['far', 'near', 'close', 'touching', 'crossing', 'inside', 'larger', 'split', 'same'],
                orientation=None, osal=None, oname=None)

    mt = lev3.maxt
    le = 1 * mt
    la = 2 * mt
    ab = 3 * mt
    ra = 4 * mt
    ri = 5 * mt
    rb = 6 * mt

    be = 7 * mt
    lb = 8 * mt
    ce = 9 * mt
    ho = 10 * mt
    ve = 11 * mt

    lev3.orientation = ['LE', 'LA', 'AB', 'RA', 'RI', 'RB', 'BE', 'LB', 'CE', 'HO', 'VE']

    lev3.osal = [0.9, 0.8, 0.9, 0.8, 0.9, 0.8, 0.9, 0.8, 1, 0.9, 0.9]

    lev3.oname = ['left', 'left-above', 'above', 'right-above', 'right', 'right-below', 'below', 'left-below',
                  'centered', 'horizontal', 'vertical']

    fuzzy.lev3 = lev3

    fam3 = np.array(
        [
            [fa + la, fa + la, fa + la, fa + ab, fa + ab, fa + ab, fa + ab, fa + ab, fa + ab, fa + ab, fa + ab, fa + ab,
             fa + ra, fa + ra, fa + ra],
            [fa + la, ne + la, ne + la, ne + la, ne + ab, ne + ab, ne + ab, ne + ab, ne + ab, ne + ab, ne + ab, ne + ra,
             ne + ra, ne + ra, fa + ra],
            [fa + la, ne + la, cl + la, cl + la, cl + ab, cl + ab, cl + ab, cl + ab, cl + ab, cl + ab, cl + ab, cl + ra,
             cl + ra, ne + ra, fa + ra],
            [fa + le, ne + la, cl + la, to + la, to + la, to + ab, to + ab, to + ab, to + ab, to + ab, to + ra, to + ra,
             cl + ra, ne + ra, fa + ri],
            [fa + le, ne + le, cl + le, to + la, cr + la, cr + ab, cr + ab, cr + ab, cr + ab, cr + ab, cr + ra, to + ra,
             cl + ri, ne + ri, fa + ri],
            [fa + le, ne + le, cl + le, to + le, cr + le, in_par + la, in_par + ab, in_par + ab, sp + ab, in_par + ra,
             cr + ri, to + ri, cl + ri, ne + ri, fa + ri],
            [fa + le, ne + le, cl + le, to + le, cr + le, in_par + le, in_par + ce, sp + ho, sp + ho, in_par + ri,
             cr + ri,
             to + ri, cl + ri, ne + ri, fa + ri],
            [fa + le, ne + le, cl + le, to + le, cr + le, in_par + le, sp + ve, sa + ce, lg + ho, in_par + ri, cr + ri,
             to + ri, cl + ri, ne + ri, fa + ri],
            [fa + le, ne + le, cl + le, to + le, cr + le, sp + le, sp + ve, lg + ve, lg + ce, sp + ri, cr + ri, to + ri,
             cl + ri, ne + ri, fa + ri],
            [fa + le, ne + le, cl + le, to + le, cr + le, in_par + lb, in_par + be, in_par + be, sp + be, in_par + rb,
             cr + ri, to + ri, cl + ri, ne + ri, fa + ri],
            [fa + le, ne + le, cl + le, to + lb, cr + lb, cr + be, cr + be, cr + be, cr + be, cr + be, cr + rb, to + rb,
             cl + ri, ne + ri, fa + ri],
            [fa + le, ne + lb, cl + lb, to + lb, to + lb, to + be, to + be, to + be, to + be, to + be, to + rb, to + rb,
             cl + rb, ne + rb, fa + ri],
            [fa + lb, ne + lb, cl + lb, cl + lb, cl + be, cl + be, cl + be, cl + be, cl + be, cl + be, cl + be, cl + rb,
             cl + rb, ne + rb, fa + rb],
            [fa + lb, ne + lb, ne + lb, ne + lb, ne + be, ne + be, ne + be, ne + be, ne + be, ne + be, ne + be, ne + rb,
             ne + rb, ne + rb, fa + rb],
            [fa + lb, fa + lb, fa + lb, fa + be, fa + be, fa + be, fa + be, fa + be, fa + be, fa + be, fa + be, fa + be,
             fa + rb, fa + rb, fa + rb]])

    fuzzy.fam3 = fam3

    return fuzzy


def get_field_size():
    return 640, 480
