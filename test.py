import copy
import os

from Scene import Scene
from PIL import Image
from itertools import repeat

from YOLO.bound_box import BoundBox
from YOLO.img_det import vbox_engine, draw_boxes, return_coordinates
import numpy as np

from eng.helper_eng import verbalize_pred_eng, verbalize_pred_eng_s
from grouping.Intersection import grouping
from helper import generate_description, count_ids, verbalize_pred, count_ids_g
from t_grouping import test_data
from helper import fuzzy


def from_pic(input_filename):
    input_filename = str(input_filename)
    print("The file name you entered is: ", input_filename)
    photo_boxed_filename = input_filename.replace('.jpg', '_boxed.jpg')
    v_boxes, v_labels, v_scores, image_w, image_h = vbox_engine(input_filename,
                                                                photo_boxed_filename)
    v_boxes.insert(0, BoundBox(XtopLeft=10, YtopLeft=10, XbottomRight=image_w + 10, YbottomRight=image_h + 10,
                               label='scene', label_id=0))
    v_labels.insert(0, 'scene')
    v_boxes_matlab, v_labels_matlab, v_labels_matlab_sequential = return_coordinates(v_boxes, v_labels)

    image = Image.open(input_filename)

    size = image.size
    onames = v_labels_matlab
    obj_num = len(v_boxes_matlab)
    ocolors = []
    ocolors.extend(repeat([1, 1, 1], obj_num))
    obj = np.array(v_boxes_matlab)

    obj_org = copy.copy(obj)
    background = []
    im = image
    background2 = []
    scene = Scene(im=im, fname=input_filename, size=size, onames=onames, ocols=[], obj=obj, obj_num=obj_num,
                  obj_org=obj_org,
                  background=background, background2=background2)
    return scene, v_labels_matlab, v_labels_matlab_sequential, v_boxes, image_w, image_h


def for_img(input_filename):
    gtruth, v_labels_matlab, v_labels_matlab_sequential, v_boxes, image_w, image_h = from_pic(input_filename)
    # gtruth, v_labels_matlab_sequential, v_boxes, image_w, image_h = test_data()
    print(gtruth.fname)
    pred_sort, pred = generate_description(gtruth)
    boxes_with_order_numbers = count_ids(pred_sort, gtruth, v_boxes)
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images/grouping_test/boxes/' + input_filename)
    print(verbalize_pred_eng_s(pred_sort, gtruth, fuzzy, boxes_with_order_numbers))
    draw_boxes(input_filename, data_path.replace('.jpg', '_boxed.jpg'), gtruth.obj[1:],
               v_labels_matlab_sequential, boxes_with_order_numbers)

    boxes_with_order_numbers = count_ids(pred, gtruth, v_boxes)
    gtruth, v_labels_matlab, v_boxes_matlab, v_labels_matlab_sequential, v_boxes = grouping(boxes_with_order_numbers,
                                                                                            pred,
                                                                                            gtruth)

    pred_sort, _ = generate_description(gtruth)
    boxes_with_order_numbers, boxes_with_order_numbers_sep = count_ids_g(pred_sort, gtruth, v_boxes)
    # print(verbalize_pred_pl(pred_sort, gtruth, fuzzy, boxes))
    print(verbalize_pred_eng(pred_sort, gtruth, fuzzy, boxes_with_order_numbers, boxes_with_order_numbers_sep))
    # print(verbalize_pred(pred_sort, gtruth, fuzzy))
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images/grouping_test/boxes/' + input_filename)
    draw_boxes(input_filename, data_path.replace('.jpg', '_boxed_grouped.jpg'), gtruth.obj[1:],
               v_labels_matlab_sequential,
               boxes_with_order_numbers)


# process_for_grouping()
# input_filename = input("Enter a file name to load bBoxes. Data must be delimited with ',': ")


# Prints in the console the variable as requested
# d = './images'
# print([for_img(d + "/" + f) for f in os.listdir(d)])
os.chdir('./images/grouping_test')
files = [f for f in os.listdir('.') if os.path.isfile(f)]
for f in files:
    for_img(f)
