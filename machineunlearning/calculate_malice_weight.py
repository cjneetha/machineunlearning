import numpy as np
from machineunlearning import to_class
import logging


def get_malice_weight(instance, model):
    """

    :param instance:
    :param model:
    :return:
    """
    # 1. add a check here to see if the entity has min instances, if yes use that dictionary
    # else use global dictionary for determining the class of the term
    # 2. if its a new word in top entity - ignore

    if instance.stars != to_class:
        return 0

    review_wt = 0
    term_dict = instance.ngrams.copy()

    for term, count in term_dict.items():

        term_count = 0
        term_inclination = None

        if instance.entity_id in model.global_entity_dictionary and \
                model.entity_review_counts[instance.entity_id] >= model.entity_min_instances:
            dict_to_use = model.global_entity_dictionary[instance.entity_id].entity_term_counts
        else:
            dict_to_use = model.global_term_counts

        # delete the new term
        if term not in dict_to_use['positive'] and term not in dict_to_use['negative'] and dict_to_use['neutral']:
            del instance.ngrams[term]
            continue

        for target in model.class_list:
            term_class_count = dict_to_use[target][term]

            if term_class_count >= term_count:
                term_count = term_class_count
                term_inclination = target
        review_wt += (term_inclination == 'negative') * count

    if len(instance.ngrams.keys()) == 0:
        # print('Empty review')
        return 0
    else:
        return 1 - (review_wt / sum(instance.ngrams.values()))


def calculate_malice_weight(data, model):
    """

    :param data:
    :param model:
    :return:
    """
    logger = logging.getLogger('log')
    print('Calculating malice weights.')
    logger.info('Calculating malice weights.')

    malice_wt = np.empty(len(data), dtype=float)

    for counter, row in enumerate(data.itertuples()):
        malice_wt[counter] = get_malice_weight(row, model)

    data['malice_wt'] = malice_wt
    print('Added malice weight column to data.')
    logger.info('Added malice weight column to data.')
    return data

