import os

import pandas as pd
import re
from helper import random_framework, get_row, get_seq_id, get_box
import copy


def load_lang_data_eng():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./location.csv")
    frameworks_location = pd.read_csv(data_path, delimiter='; ', engine='python', header=None).values
    frameworks_location = dict(zip(frameworks_location[:, 0], frameworks_location[:, 1:]))

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./orientation.csv")
    frameworks_orientation = pd.read_csv(data_path, delimiter='; ', engine='python', header=None).values
    frameworks_orientation = dict(zip(frameworks_orientation[:, 0], frameworks_orientation[:, 1:]))

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./yolov3.csv")
    data_multilingual_obj_names = pd.read_csv(data_path, delimiter=', ', engine='python', index_col=None)

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./yolov3_LM.csv")
    data_multilingual_obj_names_lm = pd.read_csv(data_path, delimiter=', ', engine='python', index_col=None)
    return frameworks_location, frameworks_orientation, data_multilingual_obj_names, data_multilingual_obj_names_lm


def verbalize_pred_eng(pred, scene, fuzzy, boxes, boxes_counted):
    txt = ""
    frameworks_location, \
    frameworks_orientation, \
    data_multilingual_obj_names, data_multilingual_obj_names_lm = load_lang_data_eng()

    preambule = generate_preambule(data_multilingual_obj_names_lm, boxes, boxes_counted)
    txt = txt.__add__(preambule)
    txt = txt.__add__("\n")
    for i in range(len(pred)):
        curr_pred = pred[i, :]
        ty = int(curr_pred[4])
        location_name_curr = fuzzy.lev3.location_names[ty]

        o = int(curr_pred[6])
        orientation_name_curr = fuzzy.lev3.orientation_name[o]
        first_obj_name = scene.onames[scene.obj[int(curr_pred[0]), 1]]
        second_obj_name = scene.onames[scene.obj[int(curr_pred[2]), 1]]

        framework_location = random_framework(frameworks_location[location_name_curr][0])
        framework_orientation = random_framework(frameworks_orientation[orientation_name_curr][0])
        sentence = create_replacement_g(framework_location, framework_orientation,
                                        [first_obj_name, second_obj_name], boxes,
                                        [int(curr_pred[0]), int(curr_pred[2])], data_multilingual_obj_names_lm)
        sentence = sentence.capitalize()
        txt = txt.__add__(sentence)
        txt = txt.__add__("\n")
        txt = txt.__add__("{} {} {} {}".format(int(curr_pred[0]), scene.onames[scene.obj[int(curr_pred[0]), 1]],
                                               int(curr_pred[2]), scene.onames[scene.obj[int(curr_pred[2]), 1]]))
        txt = txt.__add__("\n")
    return txt


def verbalize_pred_eng_s(pred, scene, fuzzy, boxes):
    txt = ""
    frameworks_location, \
    frameworks_orientation, \
    data_multilingual_obj_names, data_multilingual_obj_names_lm = load_lang_data_eng()

    preambule = generate_preambule_s(data_multilingual_obj_names_lm, boxes)
    txt = txt.__add__(preambule)
    txt = txt.__add__("\n")
    for i in range(len(pred)):
        curr_pred = pred[i, :]
        ty = int(curr_pred[4])
        location_name_curr = fuzzy.lev3.location_names[ty]

        o = int(curr_pred[6])
        orientation_name_curr = fuzzy.lev3.orientation_name[o]
        first_obj_name = scene.onames[scene.obj[int(curr_pred[0]), 1]]
        second_obj_name = scene.onames[scene.obj[int(curr_pred[2]), 1]]

        framework_location = random_framework(frameworks_location[location_name_curr][0])
        framework_orientation = random_framework(frameworks_orientation[orientation_name_curr][0])
        sentence = create_replacement(framework_location, framework_orientation,
                                      [first_obj_name, second_obj_name], boxes,
                                      [int(curr_pred[0]), int(curr_pred[2])])
        sentence = sentence.capitalize()
        txt = txt.__add__(sentence)
        txt = txt.__add__("\n")
        txt = txt.__add__("{} {} {} {}".format(int(curr_pred[0]), scene.onames[scene.obj[int(curr_pred[0]), 1]],
                                               int(curr_pred[2]), scene.onames[scene.obj[int(curr_pred[2]), 1]]))

        txt = txt.__add__("\n")
    return txt


def generate_preambule(data_multilingual_obj_names_lm, boxes, boxes_counted_sep):
    preambule = ""
    preambule_single = 'On the picture we see '
    framework = "{}"

    def filtr():
        groups = dict()
        single = dict()
        for key in boxes_counted_sep.keys():
            if key is not "scene":
                if boxes_counted_sep[key]["group"]:
                    groups[key] = boxes_counted_sep[key]["group"]
                if boxes_counted_sep[key]["single"]:
                    single[key] = boxes_counted_sep[key]["single"]
        return groups, single

    groups, single = filtr()
    for object_name in single.keys():
        preambule_single = preambule_single.__add__(framework.format(object_name))
        preambule_single = preambule_single.__add__(dot_or_comma(object_name, single))
    preambule_single = preambule_single.capitalize()
    preambule = preambule.__add__(preambule_single)

    if len(groups.keys()) >= 1:
        preambule_many = ' We see also '
        for object_name in groups.keys():
            number_of_obj_for_label = len(groups[object_name])
            sentence = create_replacement_lm(data_multilingual_obj_names_lm, object_name,
                                             number_of_obj_for_label)
            preambule_many = preambule_many.__add__(sentence)
            preambule_many = preambule_many.__add__(dot_or_comma(object_name, groups))
        preambule = preambule.__add__(preambule_many)

    return preambule


def generate_preambule_s(data_multilingual_obj_names_lm, boxes):
    preambule = ""
    preambule_single = 'On the picture we see '

    framework = "{}"
    many = dict(filter(lambda elem: len(elem[1]) > 1 and elem[0] is not "scene", boxes.items()))
    single = dict(filter(lambda elem: len(elem[1]) <= 1 and elem[0] is not "scene", boxes.items()))
    for object_name in single.keys():
        preambule_single = preambule_single.__add__(framework.format(object_name))
        preambule_single = preambule_single.__add__(dot_or_comma(object_name, single))
    preambule_single = preambule_single.capitalize()
    preambule = preambule.__add__(preambule_single)
    if len(many.keys()) >= 1:
        preambule_many = ' We see also '
        for object_name in many.keys():
            number_of_obj_for_label = len(boxes[object_name])
            sentence = create_replacement_lm_s(data_multilingual_obj_names_lm, object_name, framework,
                                               number_of_obj_for_label)
            preambule_many = preambule_many.__add__(sentence)
            preambule_many = preambule_many.__add__(dot_or_comma(object_name, many))
        preambule = preambule.__add__(preambule_many)

    return preambule


def create_replacement(framework_location, framework_orientation, resolved_obj_names_array, boxes,
                       resolved_obj_places_array):
    regex_location = r'\{(.*?)\}'

    obj_places_location = re.findall(regex_location, framework_location)
    sentence = copy.copy(framework_location)
    for a_string in obj_places_location:
        result = a_string.split(":")
        object_place = int(result[0])
        sequence_id = get_seq_id(resolved_obj_names_array[object_place], resolved_obj_places_array[object_place], boxes)
        numerical = load_numerical_data()
        sequence_id_verb_name = get_verb_numerical(sequence_id, numerical)
        s = "{" + a_string + "}"

        if sequence_id_verb_name is not '':
            sentence = sentence.replace(s,
                                        "{} {}".format(sequence_id_verb_name, resolved_obj_names_array[object_place]))
        sentence = sentence.replace(s, resolved_obj_names_array[object_place])

    regex_orientation = r'\[(.*?)\]'
    obj_places_orientation = re.findall(regex_orientation, framework_location)
    s = "[" + obj_places_orientation[0] + "]"
    sentence = sentence.replace(s, framework_orientation)
    return sentence


def create_replacement_g(framework_location, framework_orientation, resolved_obj_names_array, boxes,
                         resolved_obj_places_array, data_multilingual_obj_names_lm):
    regex_location = r'\{(.*?)\}'

    obj_places_location = re.findall(regex_location, framework_location)
    sentence = copy.copy(framework_location)
    for a_string in obj_places_location:
        result = a_string.split(":")
        object_place = int(result[0])
        box = get_box(resolved_obj_names_array[object_place], resolved_obj_places_array[object_place], boxes)
        sequence_id = get_seq_id(resolved_obj_names_array[object_place], resolved_obj_places_array[object_place], boxes)
        numerical = load_numerical_data()
        sequence_id_verb_name = get_verb_numerical(sequence_id, numerical)

        s = "{" + a_string + "}"
        obj_name = resolved_obj_names_array[object_place]
        if box.is_group:
            lm = get_row(data_multilingual_obj_names_lm, obj_name)
            obj_name = "group of {}".format(lm["LM"])
        if sequence_id_verb_name is not '':
            sentence = sentence.replace(s,
                                        "{} {}".format(sequence_id_verb_name, obj_name))
        sentence = sentence.replace(s, obj_name)

    regex_orientation = r'\[(.*?)\]'
    obj_places_orientation = re.findall(regex_orientation, framework_location)
    s = "[" + obj_places_orientation[0] + "]"
    sentence = sentence.replace(s, framework_orientation)
    return sentence


def create_replacement_lm(data_multilingual_obj_names_lm, object_name, number_of_obj_for_label):
    framework_s = "{} group of {}"
    framework_lm = "{} groups of {}"
    object_case_name = 'LM'
    object_row = get_row(data_multilingual_obj_names_lm, object_name)

    numerical = load_numerical_data_lm()
    verbal_name = get_verb_numerical(number_of_obj_for_label, numerical, col='VERB')

    if verbal_name is not '':
        if number_of_obj_for_label != 1:
            return framework_lm.format(verbal_name, object_row[object_case_name])
        else:
            return framework_s.format(verbal_name, object_row[object_case_name])
    return {}


def create_replacement_lm_s(data_multilingual_obj_names_lm, object_name, framework, number_of_obj_for_label):
    regex = r'\{(.*?)\}'
    obj_places = re.findall(regex, framework)
    sentence = copy.copy(framework)
    for a_string in obj_places:
        object_case_name = 'LM'
        object_row = get_row(data_multilingual_obj_names_lm, object_name)
        numerical = load_numerical_data_lm()
        verbal_name = get_verb_numerical(number_of_obj_for_label, numerical, col='VERB')
        s = "{" + a_string + "}"
        if verbal_name is not '':
            sentence = sentence.replace(s,
                                        "{} {}".format(verbal_name,
                                                       object_row[object_case_name]))
        sentence = sentence.replace(s, object_row[object_case_name])
    return sentence


def load_numerical_data():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./order_numerals/numerals.csv")
    numerical = pd.read_csv(data_path, delimiter=', ', engine='python', index_col=None)
    return numerical


def load_numerical_data_lm():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./main_numerals/numerals.csv")
    numerical = pd.read_csv(data_path, delimiter=', ', engine='python', index_col=None)
    return numerical


# object_case_name - case of noun
# row from data file, with noun, kind of noun and cases of it
# number of occurences of obj on image
def get_verb_numerical(sequence_id, numerical, col='VERB_SEQ'):
    if sequence_id is not None:
        # femine, man, neuter rodzaj rzeczownika
        numerical_row = get_row(numerical, sequence_id, 'LP')
        numerical_verbal = numerical_row[col]
        return numerical_verbal
    else:
        return ""


def dot_or_comma(object_name, obj_list):
    if object_name == list(obj_list.keys())[-1]:
        return "."
    else:
        if object_name == list(obj_list.keys())[-2]:
            return " and "
        return ", "
