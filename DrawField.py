import copy

import numpy
import pandas as pd


def draw_field(obj, field_size, active_obj, grid):
    imbg = pd.DataFrame(numpy.zeros(field_size))
    im = copy.copy(imbg)
    im2 = copy.copy(imbg)
    k = [3] + field_size
    imout = numpy.ndarray(k)
    for i in range(len(obj)):
        im_curr_obj = copy.copy(imbg)
        a = obj[i].pos[0] - 1
        b = (obj[i].pos[0] + obj[i].length[0] - 1) - 1
        c = obj[i].pos[1] - 1
        d = (obj[i].pos[1] + obj[i].length[1] - 1) - 1
        im_curr_obj.loc[a:b, c:d] = numpy.ones(obj[i].length) * 0.7
        e = im + im_curr_obj
        im = numpy.where((e > 1), 1, e)
        if i != active_obj:
            f = im +im_curr_obj
            im2 = numpy.where((f > 1), 1, e)




    imout[0] = im2
    imout[1] = im
    imout[2] = im2
    if grid:
        half = numpy.floor(numpy.asarray(imout.shape) / 2).astype(int)
        imout[0, :, half[2]:half[2] + 1] = numpy.ones((1, numpy.asarray(imout.shape)[1], 1)) * 0.8
        imout[0, half[1]:half[1] + 1, :] = numpy.ones((1, 1, numpy.asarray(imout.shape)[2])) * 0.8
    return imout
