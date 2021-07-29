# load yolov3 model and perform object detection
# based on https://github.com/experiencor/keras-yolo3
import os
from collections import Counter

import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from numpy import expand_dims

from YOLO.bound_box import BoundBox

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov3.txt")
labels = data = [line.strip() for line in open(data_path, 'r')]


def get_labels():
    return labels


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5
    boxes = []
    netout[..., :2] = _sigmoid(netout[..., :2])
    netout[..., 4:] = _sigmoid(netout[..., 4:])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h * grid_w):
        row = i / grid_w
        col = i % grid_w
        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[int(row)][int(col)][b][4]
            if (objectness.all() <= obj_thresh): continue
            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[int(row)][int(col)][b][:4]
            x = (col + x) / grid_w  # center position, unit: image width
            y = (row + y) / grid_h  # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w  # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h  # unit: image height
            # last elements are class probabilities
            classes = netout[int(row)][col][b][5:]
            box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, objectness, classes)
            boxes.append(box)
    return boxes


def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    new_w, new_h = net_w, net_h
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w) / 2. / net_w, float(new_w) / net_w
        y_offset, y_scale = (net_h - new_h) / 2. / net_h, float(new_h) / net_h
        boxes[i].XtopLeft = int((boxes[i].XtopLeft - x_offset) / x_scale * image_w)
        boxes[i].XbottomRight = int((boxes[i].XbottomRight - x_offset) / x_scale * image_w)
        boxes[i].YtopLeft = int((boxes[i].YtopLeft - y_offset) / y_scale * image_h)
        boxes[i].YbottomRight = int((boxes[i].YbottomRight - y_offset) / y_scale * image_h)


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.XtopLeft, box1.XbottomRight], [box2.XtopLeft, box2.XbottomRight])
    intersect_h = _interval_overlap([box1.YtopLeft, box1.YbottomRight], [box2.YtopLeft, box2.YbottomRight])
    intersect = intersect_w * intersect_h
    w1, h1 = box1.XbottomRight - box1.XtopLeft, box1.YbottomRight - box1.YtopLeft
    w2, h2 = box2.XbottomRight - box2.XtopLeft, box2.YbottomRight - box2.YtopLeft
    union = w1 * h1 + w2 * h2 - intersect
    return float(intersect) / union


def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0: continue
            for j in range(i + 1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0


# load and prepare an image
def load_image_pixels(filename, shape):
    # load the image to get its shape
    image = load_img(filename)
    width, height = image.size
    # load the image with the required size
    image = load_img(filename, target_size=shape)
    # convert to numpy array
    image = img_to_array(image)
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0
    # add a dimension so that we have one sample
    image = expand_dims(image, 0)
    return image, width, height


# get all of the results above a threshold
def get_boxes(boxes, thresh):
    v_boxes, v_labels, v_scores = list(), list(), list()
    # enumerate all boxes
    for box in boxes:
        # enumerate all possible labels
        for i in range(len(labels)):
            # check if the threshold for this label is high enough
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i] * 100)
            # don't break, many labels may trigger for one box
    return v_boxes, v_labels, v_scores


class Box:
    def __init__(self, box, label, seq_id):
        self.box = box
        self.label = label
        self.seq_id = seq_id


def return_coordinates(v_boxes, v_labels, image_w, image_h):
    boxes = dict()
    v_boxes_matlab = []
    obj_number = 1
    b = [0, 0, 10, 10, image_w, image_h]
    v_labels_matlab = ['scene']
    v_labels_matlab_sequential = ['scene']
    v_boxes_matlab.append(b)
    for idx, box in enumerate(v_boxes):
        y1, x1, y2, x2 = box.YtopLeft, box.XtopLeft, box.YbottomRight, box.XbottomRight
        width, height = box.calculate_width(), box.calculate_height()

        def check_labels(lab):
            if lab not in v_labels_matlab:
                v_labels_matlab.append(lab)
                boxes[lab] = [Box(box, v_labels[idx], 0)]
            else:
                boxes[lab].append(Box(box, v_labels[idx], len(boxes[lab])))
            return v_labels_matlab.index(lab)

        b = [obj_number, check_labels(v_labels[idx]), x1, y1, width, height]
        v_boxes_matlab.append(b)
        v_labels_matlab_sequential.append(labels[idx])
        obj_number += 1
    return v_boxes_matlab, v_labels_matlab, v_labels_matlab_sequential, boxes


# draw all results
def draw_boxes(filename, photo_boxed_filename, v_boxes, v_labels):
    # load the image
    data = pyplot.imread(filename)
    # plot the image
    pyplot.imshow(data)
    # get the context for drawing boxes
    ax = pyplot.gca()
    # plot each box
    b_boxes = dict
    for i in range(1, len(v_boxes)):
        box = v_boxes[i]
        # get coordinates
        x1, y1 = box[2], box[3]
        # calculate width and height of the box
        width, height = box[4], box[5]
        # create the shape
        rect = Rectangle((x1, y1), width, height, fill=False, color='white')
        # draw the box
        ax.add_patch(rect)
        # draw text and score in top left corner
        label = "%s %d" % (v_labels[i], i - 1)
        pyplot.text(x1, y1, label, color='red')
    # show the plot
    pyplot.savefig(photo_boxed_filename)
    pyplot.show()


# load yolov3 model
def vbox_engine(photo_filename, photo_boxed_filename):
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.h5')
    model = load_model(data_path)
    # define the expected input shape for the model
    input_w, input_h = 416, 416
    # define our new photo
    if photo_filename is '':
        return "No input file"
    # load and prepare image
    image, image_w, image_h = load_image_pixels(photo_filename, (input_w, input_h))
    # make prediction
    yhat = model.predict(image)
    # summarize the shape of the list of arrays
    print([a.shape for a in yhat])
    # define the anchors
    anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
    # define the probability threshold for detected objects
    class_threshold = 0.6
    boxes = list()
    for i in range(len(yhat)):
        # decode the output of the network
        boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)
    # correct the sizes of the bounding boxes for the shape of the image
    correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
    # suppress non-maximal boxes
    do_nms(boxes, 0.5)
    # define the labels
    # get the details of the detected objects
    v_boxes, v_labels, v_scores = get_boxes(boxes, class_threshold)
    return v_boxes, v_labels, v_scores, image_w, image_h
# summarize what we found
