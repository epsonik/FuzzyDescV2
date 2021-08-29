import copy
import os
import re
import pandas as pd

from helper import get_row, random_framework, get_seq_id, dot_or_comma


def load_lang_data_pl():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./location.csv")
    frameworks_location = pd.read_csv(data_path, delimiter='; ', engine='python', header=None).values
    frameworks_location = dict(zip(frameworks_location[:, 0], frameworks_location[:, 1:]))

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./orientation.csv")
    frameworks_orientation = pd.read_csv(data_path, delimiter='; ', engine='python', header=None).values
    frameworks_orientation = dict(zip(frameworks_orientation[:, 0], frameworks_orientation[:, 1:]))

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./yolov3_LM.csv")
    data_multilingual_obj_names_lm = pd.read_csv(data_path, delimiter=', ', engine='python', index_col=None)

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./yolov3.csv")
    data_multilingual_obj_names = pd.read_csv(data_path, delimiter=', ', engine='python', index_col=None)
    return frameworks_location, frameworks_orientation, data_multilingual_obj_names, data_multilingual_obj_names_lm


def load_numerical_data_pl():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./order_numerals/liczebniki_M.csv")
    numerical_M = pd.read_csv(data_path, delimiter=', ', engine='python', index_col=None)

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./order_numerals/liczebniki_Z.csv")
    numerical_Z = pd.read_csv(data_path, delimiter=', ', engine='python', index_col=None)

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./order_numerals/liczebniki_N.csv")
    numerical_N = pd.read_csv(data_path, delimiter=', ', engine='python', index_col=None)
    numerical = {'M': numerical_M, 'Z': numerical_Z, 'N': numerical_N}
    return numerical


def load_numerical_data_lm_pl():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./main_numerals/liczebniki_NMO.csv")
    numerical_NMO = pd.read_csv(data_path, delimiter=', ', engine='python', index_col=None)

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./main_numerals/liczebniki_Z.csv")
    numerical_Z = pd.read_csv(data_path, delimiter=', ', engine='python', index_col=None)

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./main_numerals/liczebniki_ZBIOROWE.csv")
    numerical_ZB = pd.read_csv(data_path, delimiter=', ', engine='python', index_col=None)
    numerical = {'NM': numerical_NMO, 'Z': numerical_Z, 'ZB': numerical_ZB}
    return numerical


def generate_preambule(data_multilingual_obj_names, data_multilingual_obj_names_lm, boxes):
    preambule = ""
    preambule_single = 'Na obrazie widzimy '

    framework = "{B}"
    many = dict(filter(lambda elem: len(elem[1]) > 1 and elem[0] is not "scene", boxes.items()))
    single = dict(filter(lambda elem: len(elem[1]) <= 1 and elem[0] is not "scene", boxes.items()))
    for object_name in single.keys():
        preambule_single = preambule_single.__add__(
            framework.format_map(get_row(data_multilingual_obj_names, object_name)))

        preambule_single = preambule_single.__add__(dot_or_comma(object_name, single))
    preambule_single = preambule_single.capitalize()
    preambule = preambule.__add__(preambule_single)
    if len(many.keys()) >= 1:
        preambule_many = ' Widać na nim także '
        for object_name in many.keys():
            number_of_obj_for_label = len(boxes[object_name])
            sentence = create_replacement_lm(data_multilingual_obj_names_lm, object_name, framework,
                                             number_of_obj_for_label)
            preambule_many = preambule_many.__add__(sentence)
            preambule_many = preambule_many.__add__(dot_or_comma(object_name, many))
        preambule = preambule.__add__(preambule_many)

    return preambule


def create_replacement_lm(data_multilingual_obj_names_lm, object_name, framework, number_of_obj_for_label):
    regex = r'\{(.*?)\}'
    obj_places = re.findall(regex, framework)
    sentence = copy.copy(framework)
    for a_string in obj_places:
        object_case_name = a_string
        object_row = get_row(data_multilingual_obj_names_lm, object_name)
        numerical = load_numerical_data_lm_pl()
        verbal_name = get_verb_numerical(number_of_obj_for_label, object_row,
                                         object_case_name, numerical)
        s = "{" + a_string + "}"
        if verbal_name is not '':
            sentence = sentence.replace(s,
                                        "{} {}".format(verbal_name,
                                                       object_row[object_case_name]))
        sentence = sentence.replace(s, object_row[object_case_name])
    return sentence


def verbalize_pred_pl(pred, scene, fuzzy, boxes):
    txt = ""
    frameworks_location, \
    frameworks_orientation, \
    data_multilingual_obj_names, \
    data_multilingual_obj_names_lm = load_lang_data_pl()

    preambule = generate_preambule(data_multilingual_obj_names, data_multilingual_obj_names_lm, boxes)
    txt = txt.__add__(preambule)
    txt = txt.__add__("\n")
    for i in range(len(pred)):
        curr_pred = pred[i, :]
        l = int(curr_pred[4])
        location_name_curr = fuzzy.lev3.location_names[l]

        o = int(curr_pred[6])
        orientation_name_curr = fuzzy.lev3.orientation_name[o]
        first_obj_name = scene.onames[scene.obj[int(curr_pred[0]), 1]]
        second_obj_name = scene.onames[scene.obj[int(curr_pred[2]), 1]]

        framework_location = random_framework(frameworks_location[location_name_curr][0])
        framework_orientation = random_framework(frameworks_orientation[orientation_name_curr][0])
        sentence = create_replacement(framework_location, framework_orientation, data_multilingual_obj_names,
                                      [first_obj_name, second_obj_name], boxes,
                                      [int(curr_pred[0]), int(curr_pred[2])])
        sentence = sentence.capitalize()
        txt = txt.__add__(sentence)
        txt = txt.__add__("\n")
    return txt


def create_replacement(framework_location, framework_orientation, data_object, resolved_obj_names_array, boxes,
                       resolved_obj_places_array):
    regex_location = r'\{(.*?)\}'

    obj_places_location = re.findall(regex_location, framework_location)
    sentence = copy.copy(framework_location)
    for a_string in obj_places_location:
        result = a_string.split(":")
        object_case_name = result[0]
        object_place = int(result[1])
        sequence_id = get_seq_id(resolved_obj_names_array[object_place], resolved_obj_places_array[object_place], boxes)
        object_row = get_row(data_object, resolved_obj_names_array[object_place])
        numerical = load_numerical_data_pl()
        sequence_id_verb_name = get_verb_numerical(sequence_id, object_row, object_case_name, numerical)
        s = "{" + a_string + "}"
        if sequence_id_verb_name is not '':
            sentence = sentence.replace(s, "{} {}".format(sequence_id_verb_name, object_row[object_case_name]))
        sentence = sentence.replace(s, object_row[object_case_name])

    regex_orientation = r'\[(.*?)\]'
    obj_places_orientation = re.findall(regex_orientation, framework_location)
    s = "[" + obj_places_orientation[0] + "]"
    sentence = sentence.replace(s, framework_orientation)
    return sentence


# object_case_name - case of noun
# row from data file, with noun, kind of noun and cases of it
# number of occurences of obj on image
def get_verb_numerical(sequence_id, object_row, object_case_name, numerical):
    if sequence_id is not None:
        # femine, man, neuter rodzaj rzeczownika
        object_kind = object_row['R']
        obj_numericals = numerical[object_kind]
        numerical_row = get_row(obj_numericals, sequence_id, 'LP')
        numerical_verbal = numerical_row[object_case_name]
        return numerical_verbal
    else:
        return ""
