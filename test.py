import copy

from Scene import Scene
from PIL import Image
# from YOLO.img_det import vbox_engine, draw_boxes
from itertools import repeat

from YOLO.img_det import vbox_engine
from data import load_etykiety
from helper import fmpm, get_predicates, sort_predicates, verbalize_pred
from pics import get_dog_pic, get_desk_pic
import numpy as np


def from_pic(input_filename):
    input_filename = str(input_filename)
    print("The file name you entered is: ", input_filename)
    photo_boxed_filename = input_filename.replace('.jpg', '_boxed.jpg')
    v_boxes, v_labels, v_scores, image_w, image_h = vbox_engine(input_filename, photo_boxed_filename)

    image = Image.open(input_filename)

    size = image.size
    # onames = ['scene', 'dog', 'bike', 'car']
    onames = v_labels
    obj_num = len(v_boxes)
    ocolors = []
    ocolors.extend(repeat([1, 1, 1], obj_num))
    obj = np.array(v_boxes)

    obj_org = copy.copy(obj)
    background = []
    im = image
    background2 = []
    scene = Scene(im=im, fname=input_filename, size=size, onames=onames, ocols=ocolors, obj=obj, obj_num=obj_num,
                  obj_org=obj_org,
                  background=background, background2=background2)
    return scene

# process_for_grouping()
# input_filename = input("Enter a file name to load bBoxes. Data must be delimited with ',': ")


# Prints in the console the variable as requested
input_filename = "7775781830_e93c63f661_z.jpg"

gtruth = from_pic(input_filename)
# gtruth = get_desk_pic()
# gtruth = get_dog_pic()
fuzzy = load_etykiety(
)
fmpm_mat = fmpm(gtruth, fuzzy)

pred = get_predicates(fmpm_mat, gtruth, fuzzy)
#x
pred_sort = sort_predicates(pred, gtruth, fuzzy, [1, 5, 8, 3])
#
print(verbalize_pred(pred_sort, gtruth, fuzzy))
