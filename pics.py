import copy

from PIL import Image


from Scene import Scene
import numpy as np

#przykładowe dane dla obrazów z matlaba dog.jpg i desk.png
def get_dog_pic():
    image = Image.open('dog.jpg')

    size = image.size
    # from matlab
    # onames = ['scene', 'dog', 'bike', 'car']
    onames = ['scene', 'dog', 'bike', 'car']
    ocolors = [[1, 1, 1],
               [0.0745098039215686, 0.623529411764706, 1],
               [0.929411764705882, 0.694117647058824, 0.125490196078431],
               [1, 0, 1]]
    # from matlab
    obj = np.array([[0, 0, 10, 10, 768, 576],
                    [1, 1, 148, 236, 174, 279],
                    [2, 2, 136, 144, 440, 328],
                    [3, 3, 491, 88, 205, 97]])
    obj_num = 4
    obj_org = copy.copy(obj)
    fname = "dog"
    background = []
    im = image
    background2 = []
    scene = Scene(im=im, fname=fname, size=size, onames=onames, ocols=ocolors, obj=obj, obj_num=obj_num,
                  obj_org=obj_org,
                  background=background, background2=background2)
    return scene


def get_desk_pic():
    image = Image.open('desk.png')

    size = image.size
    # from matlab
    # onames = ['scene', 'dog', 'bike', 'car']
    onames = ['scene', 'chair', 'laptop', 'keyboard', 'remote', 'bottle', 'cup', 'tv', 'mouse']
    ocolors = [[1, 1, 1],
               [1, 1, 1],
               [1, 1, 1],
               [1, 1, 1],
               [1, 1, 1],
               [1, 1, 1],
               [1, 1, 1],
               [1, 1, 1],
               [1, 1, 1]
               ]
    # from matlab
    obj = np.array([
        [0, 0, 10, 10, 640, 480],
        [1, 1, 40, 375, 453, 116],
        [2, 2, 492, 171, 158, 189],
        [3, 2, 105, 117, 140, 157],
        [4, 3, 205, 260, 224, 78],
        [5, 3, 122, 205, 131, 35],
        [6, 4, 54, 274, 41, 26],
        [7, 5, 65, 180, 37, 89],
        [8, 6, 187, 224, 32, 68],
        [9, 7, 231, 116, 198, 131],
        [10, 8, 454, 299, 28, 30],
    ]
    )

    obj_num = 11
    obj_org = copy.copy(obj)
    fname = "desk"
    background = []
    im = image
    background2 = []
    scene = Scene(im=im, fname=fname, size=size, onames=onames, ocols=ocolors, obj=obj, obj_num=obj_num,
                  obj_org=obj_org,
                  background=background, background2=background2)
    return scene
