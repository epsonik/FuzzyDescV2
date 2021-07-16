# import the necessary packages
import csv

import cv2
import numpy as np

from DrawField import draw_field
from InputData import input_data, get_field_size
from Obj import Obj
from SceneDescription import scene_description, show_image


def _interactive_mode():
    image = np.zeros((get_field_size()[0], get_field_size()[1], 3), np.uint8)

    def show_rectangles():
        for index, ref_point in enumerate(ref_points):
            thickness = 2
            if actual_rectangle_idx == index:
                thickness = -1
            rec = cv2.rectangle(image, ref_point[0], ref_point[1], ref_point[2], thickness)
            x, y = ref_point[0]
            cv2.putText(rec, str(index), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, ref_point[2], 1)
        cv2.imshow("image", image)
        if ref_points.__len__() > 1:
            _calculate_pos(ref_points)
    def adj_x(corner, n):
        corner_l = list(corner)
        corner_l[0] += n
        return tuple(corner_l)

    def unadj_x(corner, n):
        corner_l = list(corner)
        corner_l[0] -= n
        return tuple(corner_l)

    def unadj_y(corner, n):
        corner_l = list(corner)
        corner_l[1] -= n
        return tuple(corner_l)

    def adj_y(corner, n):
        corner_l = list(corner)
        corner_l[1] += n
        return tuple(corner_l)
    clone = image.copy()
    cv2.namedWindow("image")

    ref_points = [[(30, 30), (60, 60), (0, 255, 0)]]
    actual_rectangle = ref_points[0]
    actual_rectangle_idx = 0
    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF
        # press 'r' to reset the window
        if key == ord("r"):
            image = clone.copy()
        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break
        elif key == ord("-"):
            actual_rectangle_idx -= 1
            if (actual_rectangle_idx < 0):
                actual_rectangle_idx = len(ref_points) - 1
            actual_rectangle = ref_points[actual_rectangle_idx]
            image = clone.copy()
            # draw a rectangle around the region of interest
            show_rectangles()
            print("You chose rectangle number ", actual_rectangle_idx)
        elif key == ord("+"):
            actual_rectangle_idx += 1
            if actual_rectangle_idx > (len(ref_points) - 1):
                actual_rectangle_idx = 0
            actual_rectangle = ref_points[actual_rectangle_idx]
            image = clone.copy()
            # draw a rectangle around the region of interest
            show_rectangles()
            print("You chose rectangle number ", actual_rectangle_idx)
        elif key == ord("n"):
            color = list(np.random.random(size=3) * 256)
            ref_points.append([(80, 80), (110, 110), color])
            actual_rectangle = ref_points[-1]
            actual_rectangle_idx = len(ref_points) - 1
            image = clone.copy()
            # draw a rectangle around the region of interest
            show_rectangles()
        elif key == ord("d"):
            ref_points[actual_rectangle_idx] = actual_rectangle

            actual_rectangle[0] = adj_x(actual_rectangle[0], 5)
            actual_rectangle[1] = adj_x(actual_rectangle[1], 5)

            # draw a rectangle around the region of interest
            image = clone.copy()
            show_rectangles()
        elif key == ord("w"):
            ref_points[actual_rectangle_idx] = actual_rectangle

            actual_rectangle[0] = unadj_y(actual_rectangle[0], 5)
            actual_rectangle[1] = unadj_y(actual_rectangle[1], 5)
            image = clone.copy()
            # draw a rectangle around the region of interest
            show_rectangles()
        elif key == ord("s"):
            ref_points[actual_rectangle_idx] = actual_rectangle

            actual_rectangle[0] = adj_y(actual_rectangle[0],5)
            actual_rectangle[1] = adj_y(actual_rectangle[1],5)
            image = clone.copy()
            # draw a rectangle around the region of interest
            show_rectangles()
        elif key == ord("a"):
            ref_points[actual_rectangle_idx] = actual_rectangle

            actual_rectangle[0] = unadj_x(actual_rectangle[0], 5)
            actual_rectangle[1] = unadj_x(actual_rectangle[1], 5)
            image = clone.copy()
            # draw a rectangle around the region of interest
            show_rectangles()
        # up arrow
        elif key == ord("i"):
            ref_points[actual_rectangle_idx] = actual_rectangle

            actual_rectangle[0] = unadj_y(actual_rectangle[0])
            actual_rectangle[1] = adj_y(actual_rectangle[1])
            image = clone.copy()
            # draw a rectangle around the region of interest
            show_rectangles()
        # down arrow
        elif key == ord("k"):
            ref_points[actual_rectangle_idx] = actual_rectangle

            actual_rectangle[0] = adj_y(actual_rectangle[0])
            actual_rectangle[1] = unadj_y(actual_rectangle[1])
            image = clone.copy()
            show_rectangles()
        # right arrow
        elif key == ord("l"):
            ref_points[actual_rectangle_idx] = actual_rectangle

            actual_rectangle[0] = unadj_x(actual_rectangle[0])
            actual_rectangle[1] = adj_x(actual_rectangle[1])
            # draw a rectangle around the region of interest
            image = clone.copy()
            show_rectangles()
        # left arrow
        elif key == ord("j"):
            ref_points[actual_rectangle_idx] = actual_rectangle

            actual_rectangle[0] = adj_x(actual_rectangle[0])
            actual_rectangle[1] = unadj_x(actual_rectangle[1])
            # draw a rectangle around the region of interest
            image = clone.copy()
            show_rectangles()
        elif key == ord("g"):
            save_b_boxes(ref_points)
            show_rectangles()
        elif key == ord("y"):
            if ref_points.__len__() > 1:
                _calculate_pos(ref_points)
    # close all open windows
    cv2.destroyAllWindows()


def save_b_boxes(ref_points):
    file_name = input("Enter a file to save bounding boxes: ")
    file_name = str(file_name)

    # Prints in the console the variable as requested
    print("The file name you entered is: ", file_name)
    w = csv.writer(open(file_name, "w"))

    for index, ref_point in enumerate(ref_points):
        name = index
        XtopLeft, YtopLeft = ref_point[0][0], ref_point[0][1]
        XbottomRight, YbottomRight = ref_point[1][0], ref_point[1][1]
        w.writerow([name, XtopLeft, YtopLeft, XbottomRight, YbottomRight, ref_point[2]])


def _calculate_pos(ref_points):
    v_boxes = {}
    for index, ref_point in enumerate(ref_points):
        XtopLeft, YtopLeft = ref_point[0][0], ref_point[0][1]
        XbottomRight, YbottomRight = ref_point[1][0], ref_point[1][1]

        X_len = abs(XbottomRight - XtopLeft)
        Y_len = abs(YbottomRight - YtopLeft)
        obj = Obj(str(index), [YtopLeft, XtopLeft], [Y_len, X_len])

        v_boxes[index] = obj
    create_bound_boxes, create_prop, create_rels, create_rules, get_field_size = input_data()
    description = scene_description(v_boxes, create_prop, create_rels, create_rules, get_field_size)
    print(description)


def create_output_image_and_desc():
    create_bound_boxes, create_prop, create_rels, create_rules, get_field_size = input_data()
    description = scene_description(create_bound_boxes, create_prop, create_rels, create_rules, get_field_size)
    print(description)
    imf = draw_field(create_bound_boxes, get_field_size, -1, 1)
    show_image(imf)


_interactive_mode()
