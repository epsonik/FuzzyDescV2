import numpy as np
from PIL import Image

from Scene import Scene
from YOLO.bound_box import BoundBox


def test_data():
    size = (640, 480)
    ocols = [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1],
             [1, 1, 1],
             [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1],
             [1, 1, 1],
             [1, 1, 1], [1, 1, 1], [1, 1, 1]]
    onames = ['scene', 'tv', 'book', 'bed', 'potted plant', 'laptop', 'refrigerator', 'chair', 'microwave']
    image = Image.open('6813627120_a222bcba0d_z.jpg')
    obj_org = np.array([[0, 0, 10, 10, 640, 480],
                          [1, 1, 131, 203, 150, 111],
                          [2, 2, 0, 124, 135, 307],
                          [3, 3, -14, 394, 459, 92],
                          [4, 4, 292, 123, 35, 119],
                          [5, 1, 386, 183, 56, 54],
                          [6, 5, 386, 183, 76, 59],
                          [7, 6, 529, 125, 58, 197],
                          [8, 7, 406, 201, 109, 186],
                          [9, 8, 527, 72, 48, 42],
                          [10, 2, 60, 172, 15, 60],
                          [11, 2, 74, 172, 13, 59],
                          [12, 2, 11, 179, 13, 57],
                          [13, 2, 25, 179, 9, 55],
                          [14, 2, 38, 180, 8, 53],
                          [15, 2, 58, 247, 17, 52],
                          [16, 2, 70, 247, 18, 50],
                          [17, 2, 84, 249, 16, 46],
                          [18, 2, 58, 315, 22, 49],
                          [19, 2, 68, 314, 24, 48],
                          [20, 2, 31, 320, 23, 53],
                          [21, 2, 1, 329, 15, 51],
                          [22, 2, 5, 328, 24, 52]])
    obj_num = 23
    obj = np.array(
        [[0, 0, 10, 10, 640, 480],
         [1, 1, 131, 203, 150, 111],
         [2, 2, 0, 124, 135, 307],
         [3, 3, -14, 394, 459, 92],
         [4, 4, 292, 123, 35, 119],
         [5, 1, 386, 183, 56, 54],
         [6, 5, 386, 183, 76, 59],
         [7, 6, 529, 125, 58, 197],
         [8, 7, 406, 201, 109, 186],
         [9, 8, 527, 72, 48, 42],
         [10, 2, 60, 172, 15, 60],
         [11, 2, 74, 172, 13, 59],
         [12, 2, 11, 179, 13, 57],
         [13, 2, 25, 179, 9, 55],
         [14, 2, 38, 180, 8, 53],
         [15, 2, 58, 247, 17, 52],
         [16, 2, 70, 247, 18, 50],
         [17, 2, 84, 249, 16, 46],
         [18, 2, 58, 315, 22, 49],
         [19, 2, 68, 314, 24, 48],
         [20, 2, 31, 320, 23, 53],
         [21, 2, 1, 329, 15, 51],
         [22, 2, 5, 328, 24, 52]])
    fname = '6813627120_a222bcba0d_z.jpg'
    background2 = []
    background = []
    ocolors = []
    scene = Scene(im=image, fname=fname, size=size, onames=onames, ocols=ocolors, obj=obj, obj_num=obj_num,
                  obj_org=obj_org,
                  background=background, background2=background2)
    v_labels_sequential = ['scene', 'tv', 'book', 'bed', 'potted plant', 'tv', 'laptop', 'refrigerator', 'chair',
                           'microwave', 'book', 'book',
                           'book', 'book', 'book', 'book', 'book', 'book', 'book', 'book', 'book', 'book', 'book']
    v_boxes = [BoundBox(XtopLeft=10, YtopLeft=10, XbottomRight=640 + 10, YbottomRight=480 + 10, label='scene',
                        label_id=0),
               BoundBox(XbottomRight=281, XtopLeft=131, YbottomRight=314, YtopLeft=203, label='tv', label_id=1),
               BoundBox(XbottomRight=135, XtopLeft=0, YbottomRight=431, YtopLeft=124, label='book', label_id=2),
               BoundBox(XbottomRight=445, XtopLeft=-14, YbottomRight=486, YtopLeft=394, label='bed', label_id=3),
               BoundBox(XbottomRight=327, XtopLeft=292, YbottomRight=242, YtopLeft=123, label='potted plant', label_id=4),
               BoundBox(XbottomRight=442, XtopLeft=386, YbottomRight=237, YtopLeft=183, label='tv', label_id=5),
               BoundBox(XbottomRight=462, XtopLeft=386, YbottomRight=242, YtopLeft=183, label='laptop', label_id=6),
               BoundBox(XbottomRight=587, XtopLeft=529, YbottomRight=322, YtopLeft=125, label='refrigerator', label_id=7),
               BoundBox(XbottomRight=515, XtopLeft=406, YbottomRight=387, YtopLeft=201, label='chair', label_id=8),
               BoundBox(XbottomRight=575, XtopLeft=527, YbottomRight=114, YtopLeft=72, label='microwave', label_id=9),
               BoundBox(XbottomRight=75, XtopLeft=60, YbottomRight=232, YtopLeft=172, label='book', label_id=10),
               BoundBox(XbottomRight=87, XtopLeft=74, YbottomRight=231, YtopLeft=172, label='book',
                        label_id=11),
               BoundBox(XbottomRight=24, XtopLeft=11, YbottomRight=236, YtopLeft=179, label='book', label_id=12),
               BoundBox(XbottomRight=34, XtopLeft=25, YbottomRight=234, YtopLeft=179, label='book', label_id=13),
               BoundBox(XbottomRight=46, XtopLeft=38, YbottomRight=233, YtopLeft=180, label='book', label_id=14),
               BoundBox(XbottomRight=75, XtopLeft=58, YbottomRight=299, YtopLeft=247, label='book', label_id=15),
               BoundBox(XbottomRight=88, XtopLeft=70, YbottomRight=297, YtopLeft=247, label='book', label_id=16),
               BoundBox(XbottomRight=100, XtopLeft=84, YbottomRight=295, YtopLeft=249, label='book', label_id=17),
               BoundBox(XbottomRight=80, XtopLeft=58, YbottomRight=364, YtopLeft=315, label='book', label_id=18),
               BoundBox(XbottomRight=92, XtopLeft=68, YbottomRight=362, YtopLeft=314, label='book', label_id=19),
               BoundBox(XbottomRight=54, XtopLeft=31, YbottomRight=373, YtopLeft=320, label='book', label_id=20),
               BoundBox(XbottomRight=16, XtopLeft=1, YbottomRight=380, YtopLeft=329, label='book', label_id=21),
               BoundBox(XbottomRight=29, XtopLeft=5, YbottomRight=380, YtopLeft=328, label='book', label_id=22),
               ]
    return scene, v_labels_sequential, v_boxes, size[0], size[1]
