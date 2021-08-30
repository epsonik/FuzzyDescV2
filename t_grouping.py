import numpy as np
from PIL import Image

from Scene import Scene
from YOLO.bound_box import BoundBox


def test_data():
    size = (640, 480)
    ocols = []
    onames = ['scene', 'oven', 'bottle', 'sink', 'wine glass']
    image = Image.open('10.jpg')
    obj_org = np.array([[0, 0, 10, 10, 640, 480],
                        [1, 1, 420, 252, 220, 256],
                        [2, 2, 343, 109, 27, 121],
                        [3, 2, 397, 113, 31, 115],
                        [4, 2, 426, 110, 26, 120],
                        [5, 2, 463, 99, 28, 140],
                        [6, 3, 59, 176, 160, 40],
                        [7, 4, 257, 157, 36, 95],
                        [8, 4, 291, 157, 30, 83],
                        [9, 2, 459, 155, 24, 97],
                        [10, 4, 486, 154, 30, 99],
                        [11, 2, 424, 175, 30, 98]])
    obj_num = 12
    obj = np.array(
        [[0, 0, 10, 10, 640, 480],
         [1, 1, 420, 252, 220, 256],
         [2, 2, 343, 109, 27, 121],
         [3, 2, 397, 113, 31, 115],
         [4, 2, 426, 110, 26, 120],
         [5, 2, 463, 99, 28, 140],
         [6, 3, 59, 176, 160, 40],
         [7, 4, 257, 157, 36, 95],
         [8, 4, 291, 157, 30, 83],
         [9, 2, 459, 155, 24, 97],
         [10, 4, 486, 154, 30, 99],
         [11, 2, 424, 175, 30, 98]])

    fname = '10.jpg'
    background2 = []
    background = []
    ocolors = []
    scene = Scene(im=image, fname=fname, size=size, onames=onames, ocols=ocolors, obj=obj, obj_num=obj_num,
                  obj_org=obj_org,
                  background=background, background2=background2)
    v_labels_sequential = ['scene', 'oven', 'bottle', 'bottle', 'bottle', 'bottle', 'sink', 'wine glass', 'wine glass',
                           'bottle', 'wine glass', 'bottle']
    v_boxes = [BoundBox(XtopLeft=10, YtopLeft=10, XbottomRight=640 + 10, YbottomRight=480 + 10, label='scene',
                        label_id=0),
               BoundBox(XbottomRight=640, XtopLeft=420, YbottomRight=508, YtopLeft=252, label='oven', label_id=1),
               BoundBox(XbottomRight=370, XtopLeft=343, YbottomRight=230, YtopLeft=109, label='bottle', label_id=2),
               BoundBox(XbottomRight=428, XtopLeft=397, YbottomRight=228, YtopLeft=113, label='bottle', label_id=3),
               BoundBox(XbottomRight=452, XtopLeft=426, YbottomRight=230, YtopLeft=110, label='bottle', label_id=4),
               BoundBox(XbottomRight=491, XtopLeft=463, YbottomRight=239, YtopLeft=99, label='bottle', label_id=5),
               BoundBox(XbottomRight=219, XtopLeft=59, YbottomRight=216, YtopLeft=176, label='sink', label_id=6),
               BoundBox(XbottomRight=293, XtopLeft=257, YbottomRight=252, YtopLeft=157, label='wine glass', label_id=7),
               BoundBox(XbottomRight=321, XtopLeft=291, YbottomRight=240, YtopLeft=157, label='wine glass', label_id=8),
               BoundBox(XbottomRight=483, XtopLeft=459, YbottomRight=252, YtopLeft=155, label='bottle', label_id=9),
               BoundBox(XbottomRight=516, XtopLeft=486, YbottomRight=253, YtopLeft=154, label='wine glass',
                        label_id=10),
               BoundBox(XbottomRight=454, XtopLeft=424, YbottomRight=273, YtopLeft=175, label='bottle', label_id=11),
               ]

    return scene, v_labels_sequential, v_boxes, size[0], size[1]
