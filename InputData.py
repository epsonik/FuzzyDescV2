#  dane testowe
from Obj import Obj
from Prop import Prop, CoordinateName
from Rule import Rule

language = "PL"


def input_data():
    # data of objects at the beginning
    return create_bound_boxes(), create_prop(), create_rels(), create_rules(), get_field_size()


def create_bound_boxes():
    objects = {}
    objects[0] = Obj('A', [5, 15], [15, 30])
    objects[1] = Obj('B', [45, 150], [25, 90])
    objects[2] = Obj('C', [200, 115], [55, 80])
    return objects


def create_prop():
    props = {}
    # properties left and right
    props[0] = Prop('left edge', 1, 0.9, 'b_l', 'trapezoid', [-2, -2, -1, -0.85],
                    ["Przy lewej krawędzi znajduje się ", ". "], ["Next to the left border is ", ". "],
                    CoordinateName.X)
    props[1] = Prop('left side', 2, 1, 'b_l', 'trapezoid', [-10, -0.9, -0.6, 0],
                    ["W lewej części pola widzimy ", ". "], ["In the left part of field is ", ". "], CoordinateName.X)
    props[2] = Prop('center left-right', 3, 0.9, 'c_lr', 'trapezoid', [-0.2, -0.05, 0.05, 0.2],
                    ["Na środku szerokości znajduje się ", ". "], ["In the center of width is ", ". "],
                    CoordinateName.X)
    props[3] = Prop('right side', 4, 1, 'b_r', 'trapezoid', [0, 0.6, 0.9, 10],
                    ["W prawej części pola widzimy ", ". "], ["In the right part of field is ", ". "], CoordinateName.X)
    props[4] = Prop('right edge', 5, 0.9, 'b_r', 'trapezoid', [0.85, 1, 2, 2],
                    ["Przy prawej krawędzi znajduje się ", ". "], ["Next to the right border is ", ". "],
                    CoordinateName.X)
    # propertes down and upper
    props[5] = Prop('top edge', 6, 0.7, 'b_t', 'trapezoid', [-2, -2, -1, -0.85],
                    ["Przy górnej krawędzi znajduje się ", ". "], ["Next to the top border is ", ". "],
                    CoordinateName.Y)
    props[6] = Prop('upper part', 7, 6, 'b_t', 'trapezoid', [-10, -0.9, -0.6, 0],
                    ["W górnej części pola widzimy ", ". "], ["In the top part of field is ", ". "], CoordinateName.Y)
    props[7] = Prop('center up-down', 8, 0.8, 'c_tb', 'trapezoid', [-0.2, -0.05, 0.05, 0.2],
                    ["Na środku wysokości znajduje się ", ". "], ["In the center of height is ", ". "],
                    CoordinateName.Y)
    props[8] = Prop('lower part', 9, 1.1, 'b_b', 'trapezoid', [0, 0.6, 0.9, 10],
                    ["W dolnej części pola widzimy ", ". "], ["In the bottom part of the field is", ". "],
                    CoordinateName.Y)
    props[9] = Prop('bottom edge', 10, 0.7, 'b_b', 'trapezoid', [0.85, 1, 2, 2],
                    ["Przy dolnej krawędzi znajduje się ", ". "], ["Next to the bottom corner is ", ". "],
                    CoordinateName.Y)
    props[10] = Prop('full width', 11, 1.9, 'f_w', 'trapezoid', [0.5, 0.6, 1, 0.5],
                     ["Na całej szerokości znajduje się ", ". "], ["On the full width is ", ". "], CoordinateName.Y)
    props[11] = Prop('full height', 11, 0.9, 'f_h', 'trapezoid', [0.5, 0.6, 1, 0.5],
                     ["Na całej wysokości znajduje się ", ". "], ["On the full height is ", ". "], CoordinateName.Y)
    return props


# define relations
def create_rels():
    rels = {}

    rels[0] = Prop('on the right', 21, 0.8, 'd_lr', 'trapezoid', [0, 0.01, 0.5, 2],
                   ["Obiekt ", " znajduje się po prawej stronie obiektu ", ". "],
                   ["Object ", " is on the right of object ", ". "])
    rels[1] = Prop('on the left', 22, 0.8, 'd_lr', 'trapezoid', [-2, -0.5, -0.01, 0],
                   ["Obiekt ", " znajduje się po lewej stronie obiektu ", ". "],
                   ["Object ", " is on the left of object ", ". "])

    rels[2] = Prop('above', 23, 0.8, 'd_tb', 'trapezoid', [-2, -0.5, -0.01, 0],
                   ["Obiekt ", " znajduje się powyżej obiektu ", ". "], ["Object ", " is above object ", ". "])

    rels[3] = Prop('below', 24, 0.8, 'd_tb', 'trapezoid', [0, 0.01, 0.5, 2],
                   ["Obiekt ", " znajduje się poniżej obiektu ", ". "], ["Object ", " is under object ", ". "])

    return rels


# define rules
def create_rules():
    rules = {}
    rules[0] = Rule('center', 41, 1.1, 3, 8, 'min',
                    ["Obiekt ", " znajduje się na środku pola", ". "],
                    ["Object ", " is on the center of field", ". "])
    rules[1] = Rule('top left corner', 42, 1, 1.1, 6, 'min',
                    ["Obiekt ", " znajduje się w lewym-górnym narożniku pola", ". "],
                    ["Object ", " is on the left-top corner of field", ". "])
    rules[2] = Rule('top right corner', 43, 1.1, 5, 6, 'min',
                    ["Obiekt ", " znajduje się w prawym-górnym narożniku pola", ". "],
                    ["Object ", " is on the left-top corner of field", ". "])
    rules[3] = Rule('bottom right corner', 44, 1.1, 5, 10, 'min',
                    ["Obiekt ", " w prawym-dolnym narożniku pola", ". "],
                    ["Object ", " is on the right-bottom corner of field", ". "])
    rules[4] = Rule('bottom left corner', 45, 1, 1, 10, 'min',
                    ["Obiekt ", " znajduje się w lewym-dolnym narożniku pola", ". "],
                    ["Object ", " is on the left-bottom corner of field", ". "])
    rules[5] = Rule('top left part', 46, 1, 2, 7, 'min',
                    ["Obiekt ", " znajduje się w lewej-górnej części pola", ". "],
                    ["Object ", " is on the left-top part of field", ". "])
    rules[6] = Rule('top right part', 47, 1, 4, 7, 'min',
                    ["Obiekt ", " znajduje się w prawej-górnej części pola", ". "],
                    ["Object ", " is on the right-top part of field", ". "])
    rules[7] = Rule('bottom right part', 48, 1, 4, 9, 'min',
                    ["Obiekt ", " znajduje się w prawej-dolnej części pola", ". "],
                    ["Object ", " is on the right-bottom part of field", ". "])
    rules[8] = Rule('bottom left part', 49, 1, 2, 9, 'min',
                    ["Obiekt ", " znajduje się w lewej-dolnej części pola", ". "],
                    ["Object ", " is on the left-bottom part of field", ". "])
    return rules


# predefined field size
def get_field_size():
    return [400, 400]
