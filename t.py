import numpy as np
from PIL import Image

from Scene import Scene


def test_data():
    size = (640, 480)
    ocols = [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1],
             [1, 1, 1],
             [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1],
             [1, 1, 1],
             [1, 1, 1], [1, 1, 1], [1, 1, 1]]
    onames = ['scene', 'tv', 'book', 'bed', 'potted plant', 'laptop', 'refrigerator', 'chair', 'microwave']
    image = Image.open('images/6813627120_a222bcba0d_z.jpg')
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
    fname = 'images/6813627120_a222bcba0d_z.jpg'
    background2 = []
    background = []
    ocolors = []
    scene = Scene(im=image, fname=fname, size=size, onames=onames, ocols=ocolors, obj=obj, obj_num=obj_num,
                  obj_org=obj_org,
                  background=background, background2=background2)
    v_labels_sequential = ['scene', 'tv', 'book', 'bed', 'potted plant', 'tv', 'laptop', 'refrigerator', 'chair',
                           'microwave', 'book', 'book',
                           'book', 'book', 'book', 'book', 'book', 'book', 'book', 'book', 'book', 'book', 'book']
    return scene, v_labels_sequential
