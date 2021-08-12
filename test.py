import copy
import os
import pandas as pd
from collections import Counter

from Scene import Scene
from PIL import Image
from itertools import repeat

from YOLO.img_det import vbox_engine, draw_boxes, return_coordinates
import numpy as np

from eng.helper_eng import verbalize_pred_eng
from helper import generate_description, count_ids
from pl.helper_pl import verbalize_pred_pl
from t import test_data


def from_pic(input_filename):
    input_filename = str(input_filename)
    print("The file name you entered is: ", input_filename)
    photo_boxed_filename = input_filename.replace('.jpg', '_boxed.jpg')
    v_boxes, v_labels, v_scores, image_w, image_h = vbox_engine(input_filename,
                                                                photo_boxed_filename)
    v_boxes_matlab_format, v_labels_matlab, v_labels_matlab_sequential = return_coordinates(v_boxes, v_labels,
                                                                                            image_w, image_h)

    image = Image.open(input_filename)

    size = image.size
    onames = v_labels_matlab
    obj_num = len(v_boxes_matlab_format)
    ocolors = []
    ocolors.extend(repeat([1, 1, 1], obj_num))
    obj = np.array(v_boxes_matlab_format)

    obj_org = copy.copy(obj)
    background = []
    im = image
    background2 = []
    scene = Scene(im=im, fname=input_filename, size=size, onames=onames, ocols=[], obj=obj, obj_num=obj_num,
                  obj_org=obj_org,
                  background=background, background2=background2)
    return scene, v_labels_matlab, v_labels_matlab_sequential


def for_img(input_filename):
    photo_boxed_filename = input_filename.replace('.jpg', '_boxed.jpg')
    gtruth, v_labels_matlab, v_labels_sequential = from_pic(input_filename)

    # gtruth, v_labels_sequential = test_data()
    pred_sort, gtruth, fuzzy = generate_description(gtruth)
    boxes = count_ids(pred_sort, gtruth)
    # print(verbalize_pred_pl(pred_sort, gtruth, fuzzy, boxes))
    print(verbalize_pred_eng(pred_sort, gtruth, fuzzy, boxes))

    # draw_boxes(input_filename, photo_boxed_filename, gtruth.obj, v_labels_sequential, boxes)


# process_for_grouping()
# input_filename = input("Enter a file name to load bBoxes. Data must be delimited with ',': ")


# Prints in the console the variable as requested
d = './images'
print([for_img(d + "/" + f) for f in os.listdir(d)])
# pd.DataFrame({'col1': ["weewwwe deere", 2], 'col2': [3, "weweewwewe ererer"]}).to_csv("t,csv")
