import numpy as np
from PIL import Image

from Scene import Scene
from YOLO.bound_box import BoundBox


def test_data():
    size = (640, 480)
    ocols = []
    onames = ['scene', 'laptop', 'tv', 'keyboard', 'chair', 'bottle', 'cup', 'remote', 'mouse']
    image = Image.open('9.jpg')
    obj_org = np.array([
        [0, 0, 10, 10, 640, 480],
        [1, 1, 91, 107, 161, 147],
        [2, 2, 234, 105, 180, 132],
        [3, 1, 472, 163, 169, 197],
        [4, 3, 191, 268, 223, 56],
        [5, 4, 28, 365, 453, 125],
        [6, 5, 52, 166, 40, 91],
        [7, 3, 109, 192, 133, 37],
        [8, 6, 174, 212, 35, 69],
        [9, 3, 205, 257, 197, 47],
        [10, 7, 42, 261, 43, 27],
        [11, 8, 440, 284, 32, 32]
    ])
    obj_num = 12
    obj = np.array(
        [[0, 0, 10, 10, 640, 480],
         [1, 1, 91, 107, 161, 147],
         [2, 2, 234, 105, 180, 132],
         [3, 1, 472, 163, 169, 197],
         [4, 3, 191, 268, 223, 56],
         [5, 4, 28, 365, 453, 125],
         [6, 5, 52, 166, 40, 91],
         [7, 3, 109, 192, 133, 37],
         [8, 6, 174, 212, 35, 69],
         [9, 3, 205, 257, 197, 47],
         [10, 7, 42, 261, 43, 27],
         [11, 8, 440, 284, 32, 32]
         ])

    fname = '9.jpg'
    background2 = []
    background = []
    ocolors = []
    scene = Scene(im=image, fname=fname, size=size, onames=onames, ocols=ocolors, obj=obj, obj_num=obj_num,
                  obj_org=obj_org,
                  background=background, background2=background2)
    v_labels_sequential = ['scene', 'laptop', 'tv', 'laptop', 'keyboard', 'chair', 'bottle', 'keyboard', 'cup',
                           'keyboard', 'remote', 'mouse']
    v_boxes = [BoundBox(XtopLeft=10, YtopLeft=10, XbottomRight=640 + 10, YbottomRight=480 + 10, label='scene',
                        label_id=0),
               BoundBox(XbottomRight=252, XtopLeft=91, YbottomRight=254, YtopLeft=107, label='laptop', label_id=1),
               BoundBox(XbottomRight=414, XtopLeft=234, YbottomRight=237, YtopLeft=105, label='tv', label_id=2),
               BoundBox(XbottomRight=641, XtopLeft=472, YbottomRight=360, YtopLeft=163, label='laptop', label_id=3),
               BoundBox(XbottomRight=414, XtopLeft=191, YbottomRight=324, YtopLeft=268, label='keyboard', label_id=4),
               BoundBox(XbottomRight=481, XtopLeft=28, YbottomRight=490, YtopLeft=365, label='chair', label_id=5),
               BoundBox(XbottomRight=92, XtopLeft=52, YbottomRight=257, YtopLeft=166, label='bottle', label_id=6),
               BoundBox(XbottomRight=242, XtopLeft=109, YbottomRight=229, YtopLeft=192, label='keyboard', label_id=7),
               BoundBox(XbottomRight=209, XtopLeft=174, YbottomRight=281, YtopLeft=212, label='cup', label_id=8),
               BoundBox(XbottomRight=402, XtopLeft=205, YbottomRight=304, YtopLeft=257, label='keyboard', label_id=9),
               BoundBox(XbottomRight=85, XtopLeft=42, YbottomRight=288, YtopLeft=264, label='remote',
                        label_id=10),
               BoundBox(XbottomRight=472, XtopLeft=440, YbottomRight=316, YtopLeft=284, label='mouse', label_id=11),
               ]

    return scene, v_labels_sequential, v_boxes, size[0], size[1]
