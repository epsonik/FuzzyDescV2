import statistics
from collections import defaultdict
from math import sqrt

import numpy

from DrawField import draw_field
from InputData import input_data, create_prop, create_rules
from Obj import ObjData
# generate description
from Pred import Pred
from Prop import CoordinateName

RULE = -2

PROPERTY = -1


def scene_description(create_bound_boxes, create_prop, create_rels, create_rules, get_field_size):
    # generowanie opisu sceny
    # create relative positions of obiektów
    obj_data = get_relpos(create_bound_boxes, get_field_size)
    pred = get_predicates(obj_data, create_prop, create_rels)
    # create predicated based on rules
    rule_pred = add_rule_predicates(obj_data, pred, create_rules)
    # select correct predicated
    predicates = pred + rule_pred
    pred_sel = select_predicates(predicates, 1, 2)
    # generate description from selected predicated
    description = get_description(create_bound_boxes, pred_sel, create_prop, create_rels, create_rules)
    return description


# create relative position of objects
def get_relpos(obj, field_size):
    number_of_b_boxes = len(obj)
    left = []
    right = []
    horiz_len = []
    up = []
    down = []
    vert_len = []
    size = []
    for box_idx in range(number_of_b_boxes):
        # check if object left position trend is 1 not -1
        left.append(2 * obj[box_idx].pos[1] / field_size[1] - 1)
        right.append(2 * (obj[box_idx].pos[1] + obj[box_idx].length[1]) / field_size[1] - 1)
        horiz_len.append(obj[box_idx].length[1] / field_size[1])
        up.append(2 * obj[box_idx].pos[0] / field_size[0] - 1)
        down.append(2 * (obj[box_idx].pos[0] + obj[box_idx].length[0]) / field_size[1] - 1)
        vert_len.append(obj[box_idx].length[0] / field_size[0])
        size.append(horiz_len[box_idx] * vert_len[box_idx])
    obj_data = ObjData(number_of_b_boxes, left, right, horiz_len, up, down, vert_len, size)

    return obj_data


# return argument of memebership function
# ob_data - matrix of bounding boxes data
# pr - seetings or relation of
#
def get_mfarg(ob_data, pr, ob_num1, ob_num2):
    out = [0]
    # left borde of bound_box
    if (pr.farg == 'b_l'):
        out = ob_data.left
    # right border of bound_box
    elif (pr.farg == 'b_r'):
        out = ob_data.right
    #     centroid of left-right bound_box
    elif (pr.farg == 'c_lr'):
        out = (numpy.array(ob_data.left) + numpy.array(ob_data.right)) / 2
    # top border of bound_box
    elif (pr.farg == 'b_t'):
        out = ob_data.up
    # bottom border of bound_box
    elif (pr.farg == 'b_b'):
        out = ob_data.down
    elif (pr.farg == 'f_w'):
        out = ob_data.horiz_len
    elif (pr.farg == 'f_h'):
        out = ob_data.vert_len
    # centroid top-bottom
    elif (pr.farg == 'c_ud'):
        out = (numpy.array(ob_data.up) + numpy.array(ob_data.down)) / 2
    # distance betwen centroids left-right
    elif (pr.farg == 'd_lr'):
        out = [(ob_data.right[ob_num1] + ob_data.left[ob_num1]) / 2 - (
                ob_data.right[ob_num2] + ob_data.left[ob_num2]) / 2]
    # distance between centroids top-bottom
    elif (pr.farg == 'd_tb'):
        out = [(ob_data.up[ob_num1] + ob_data.down[ob_num1]) / 2 - (ob_data.up[ob_num2] + ob_data.down[ob_num2]) / 2]
    # Euclidean distance between centroids
    elif (pr.farg == 'd'):
        dist_ud = (ob_data.up[ob_num1] + ob_data.down[ob_num1]) / 2 - (ob_data.up[ob_num2] + ob_data.down[ob_num2]) / 2
        dist_lr = (ob_data.right[ob_num1] + ob_data.left[ob_num1]) / 2 - (
                ob_data.right[ob_num2] + ob_data.left[ob_num2]) / 2
        out = [sqrt(dist_ud * dist_ud + dist_lr * dist_lr)]
    return out


# trapezoid membership function
def tmf(in_arg, ftype, fval):
    import copy
    out = copy.copy(in_arg)
    if ftype == 'trapezoid':
        for trapeze_param_idx in range(len(in_arg)):
            if (in_arg[trapeze_param_idx] < fval[0]):
                out[trapeze_param_idx] = 0
            elif (in_arg[trapeze_param_idx] < fval[1]):
                out[trapeze_param_idx] = (in_arg[trapeze_param_idx] - fval[0]) / (fval[1] - fval[0])
            elif (in_arg[trapeze_param_idx] < fval[2]):
                out[trapeze_param_idx] = 1
            elif (in_arg[trapeze_param_idx] < fval[3]):
                out[trapeze_param_idx] = (fval[3] - in_arg[trapeze_param_idx]) / (fval[3] - fval[2])
            else:
                out[trapeze_param_idx] = 0
    else:
        print('wrong membership function type')
    return out


# generate values of fuzzy descriptors position per  properties
# vector of values per which we are counting values of descriptior values
def get_properties(ob_data, prop):
    obj_prop = []
    max = 1
    # property number
    for prop_idx in range(len(prop)):
        trapezoid_membership_function_val = tmf(get_mfarg(ob_data, prop[prop_idx], 0, 0), prop[prop_idx].ftype,
                                                prop[prop_idx].fthr)
        if len(trapezoid_membership_function_val) > max:
            max = len(trapezoid_membership_function_val)
        if len(trapezoid_membership_function_val) < max:
            obj_prop.append(trapezoid_membership_function_val * max)
        else:
            obj_prop.append(trapezoid_membership_function_val)
    return obj_prop


def get_predicates(obj_data, prop, rel):
    # generate predicated per properties and relations
    # saliency of objects that is proportional to size of object
    sal = obj_data.size
    # count values of prperties confidence factor
    obj_prop = get_properties(obj_data, prop)
    # count values of realations confidence factor
    obj_rel = get_relations(obj_data, rel)
    # generate predicates
    predicates = []
    # iterate over object reference number
    for box_idx_main in range(obj_data.number_of_b_boxes):
        # properties
        for prop_idx in range(len(prop)):
            # values of membership fuction
            if obj_prop[prop_idx][box_idx_main] > 0:
                confidence_factor = obj_prop[prop_idx][box_idx_main] * sal[box_idx_main] * prop[prop_idx].psal
                used_in_rules = obj_prop[prop_idx][box_idx_main]
                pred = Pred(predicate_id=prop[prop_idx].id,
                            confidence_factor=confidence_factor,
                            number_of_reference_object=box_idx_main,
                            number_of_sec_obj_for_relation=PROPERTY,
                            property_rule_or_rel_number=prop_idx, used_in_rules=used_in_rules)
                predicates.append(pred)
        # realtions
        # iterate over objects, which  i objects is in relation
        for box_idx in range(obj_data.number_of_b_boxes):
            # iterate over relation number
            for rel_idx in range(len(rel)):
                if obj_rel[rel_idx, box_idx_main, box_idx] > 0:
                    confidence_factor = obj_rel[rel_idx, box_idx_main, box_idx] * sal[box_idx_main] * rel[rel_idx].psal
                    used_in_rules = obj_rel[rel_idx, box_idx_main, box_idx]
                    pred = Pred(predicate_id=rel[rel_idx].id,
                                confidence_factor=confidence_factor,
                                number_of_reference_object=box_idx_main,
                                number_of_sec_obj_for_relation=box_idx,
                                property_rule_or_rel_number=rel_idx, used_in_rules=used_in_rules)
                    predicates.append(pred)

    return predicates


# generate values od fuzzy descriptors position per relations
def get_relations(ob_data, rel):
    import numpy
    obj_rel = numpy.ones((len(rel), ob_data.number_of_b_boxes, ob_data.number_of_b_boxes))

    for rel_idx in range(len(rel)):
        for box_idx in range(ob_data.number_of_b_boxes):
            for box_idx_sec_loop in range(ob_data.number_of_b_boxes):
                temp = tmf(get_mfarg(ob_data, rel[rel_idx], box_idx, box_idx_sec_loop), rel[rel_idx].ftype,
                           rel[rel_idx].fthr)
                obj_rel[rel_idx, box_idx, box_idx_sec_loop] = temp[0]
    return obj_rel


#  add predicates  from rules - inference
def add_rule_predicates(obj_data, predicates, rule):
    # saliency of objects proportional to relative length
    sal = obj_data.size
    counter = 0
    rule_predicates = []
    # iterate over rules
    for rule_idx in range(len(rule)):
        # iterate over predicates
        for pred_idx in range(len(predicates)):
            # found predicate with first premise
            if (predicates[pred_idx].predicate_id == rule[rule_idx].id_first):
                # save object number
                obj_found = predicates[pred_idx].number_of_reference_object
                # iterate over predicates - look for second premise
                for pred_second_premise_idx in range(len(predicates)):
                    if ((predicates[pred_second_premise_idx].predicate_id == rule[rule_idx].id_second) and (
                            predicates[pred_second_premise_idx].number_of_reference_object == obj_found)):
                        # second second premise for the same object
                        if ((predicates[pred_idx].used_in_rules != 0) and (
                                predicates[pred_second_premise_idx].used_in_rules != 0)):
                            # # if is not equal to 0, calculates new predicate
                            ruleobj_cf = combine_mf(predicates[pred_idx].used_in_rules,
                                                    predicates[pred_second_premise_idx].used_in_rules,
                                                    rule[rule_idx].operator)
                            confidence_factor = ruleobj_cf * sal[obj_found] * rule[rule_idx].psal
                            rule_pred = Pred(predicate_id=rule[rule_idx].id,
                                             confidence_factor=confidence_factor,
                                             number_of_reference_object=obj_found,
                                             number_of_sec_obj_for_relation=RULE,
                                             property_rule_or_rel_number=rule_idx, used_in_rules=ruleobj_cf)
                            rule_predicates.append(rule_pred)
                            counter = counter + 1
    return rule_predicates


# cobination of two values of membership function(used in rules)
def combine_mf(val1, val2, type):
    out = 0
    if type == 'min':
        out = min(val1, val2)
    elif type == 'max':
        out = max(val1, val2)
    elif type == 'mean':
        out = statistics.mean([val1, val2])
    return out


# select most important predicates
# num_times - how often object is mentioned in the text
def select_predicates(predicates, num_times_coordinate_x, num_times_coordinate_y):
    def most_important_predicates_filter(pred):
        t = False
        if pred.confidence_factor > 0: t = True
        return t

    predicates = list(filter(most_important_predicates_filter, predicates))

    def set_coordinates(predicates):
        prop = create_prop()
        rule = create_rules()

        def get_coordinate(predicates, prop, rule, pred_idx):
            if predicates[pred_idx].number_of_sec_obj_for_relation == PROPERTY:
                return prop[predicates[pred_idx].property_rule_or_rel_number].coordinate_name
            if predicates[pred_idx].number_of_sec_obj_for_relation == RULE:
                return rule[predicates[pred_idx].property_rule_or_rel_number].coordinate_name

        for pred_idx in range(len(predicates)):
            predicates[pred_idx].coordinate_name = get_coordinate(predicates, prop, rule, pred_idx)
        return predicates

    predicates = set_coordinates(predicates)

    def group_coordinates(predicates_for_obj):
        coordinates_groups = defaultdict(list)
        for pred in predicates_for_obj:
            coordinates_groups[pred.coordinate_name].append(pred)

        def sort_predicates_by_confidence_factor(predicates_for_coordinate):
            def myFunc(e):
                return e.confidence_factor

            predicates_for_coordinate.sort(key=myFunc, reverse=True)

        for _, predicates_for_coordinate in coordinates_groups.items():
            sort_predicates_by_confidence_factor(predicates_for_coordinate)
        return coordinates_groups[CoordinateName.X][:num_times_coordinate_x] + \
               coordinates_groups[CoordinateName.Y][:num_times_coordinate_y]

    def group_by_obj(predicates):
        groups = defaultdict(list)
        for pred in predicates:
            groups[pred.number_of_reference_object].append(pred)
        return groups

    obj_groups_dict = group_by_obj(predicates)
    final_predicates=list()
    for obj, predicates_for_obj in obj_groups_dict.items():
        final_predicates += group_coordinates(predicates_for_obj)

    return final_predicates


# generate decription from the matrix with most important predicates
def get_description(obj, predicates, prop, rel, rule):
    desc = []
    for pred_idx in range(len(predicates)):
        # property
        if (predicates[pred_idx].number_of_sec_obj_for_relation == PROPERTY):
            zdanie = prop[predicates[pred_idx].property_rule_or_rel_number].text[0] + obj[
                int(predicates[pred_idx].number_of_reference_object)].name + \
                     prop[predicates[pred_idx].property_rule_or_rel_number].text[
                         1]
            desc.append(zdanie)
        elif (predicates[pred_idx].number_of_sec_obj_for_relation == RULE):
            zdanie = rule[predicates[pred_idx].property_rule_or_rel_number].text[0] + obj[
                int(predicates[pred_idx].number_of_reference_object)].name + \
                     rule[predicates[pred_idx].property_rule_or_rel_number].text[
                         1]
            desc.append(zdanie)
        # relation
        else:
            zdanie = rel[predicates[pred_idx].property_rule_or_rel_number].text[0] + obj[
                int(predicates[pred_idx].number_of_reference_object)].name + \
                     rel[predicates[pred_idx].property_rule_or_rel_number].text[
                         1] + obj[
                         predicates(pred_idx, 5)].name, \
                     rel[predicates[pred_idx].property_rule_or_rel_number].text[2]
            desc.append(zdanie)
    return desc


# plot image
def show_image(imf):
    import matplotlib.pyplot as plt
    plt.imshow(imf[2])
    plt.show()


def create_output_image_and_desc():
    create_bound_boxes, create_prop, create_rels, create_rules, get_field_size = input_data()
    description = scene_description(create_bound_boxes, create_prop, create_rels, create_rules, get_field_size)
    print(description)
    imf = draw_field(create_bound_boxes, get_field_size, -1, 1)
    show_image(imf)
