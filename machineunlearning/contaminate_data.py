import random
import numpy as np
import pandas as pd
import logging
import math
from machineunlearning import portion, to_class


def get_new_term(model):
    """

    :param model:
    :return:
    """

    model.calculate_topk_features()
    topk_features = model.topk_features

    model.topk += 1
    model.calculate_topk_features()
    topk_plus1_features = model.topk_features

    model.topk -= 1

    if len(topk_features) < model.topk:
        term = list(topk_features)[-1]
    else:
        term = list(topk_plus1_features - topk_features)[0]

    return term


def contaminate_review(review, contamination_term):
    """

    :param review:
    :param contamination_term:
    :return:
    """
    if review.is_contaminated == 1:
        ngrams_copy = review.ngrams.copy()
        ngrams_copy[contamination_term] = 1
        return ngrams_copy
    else:
        return review.ngrams


def contaminate_top_entities(data, contamination_entities, contamination_term):
    """

    :param data:
    :param contamination_entities:
    :param contamination_term:
    :return:
    """
    # replicating class column
    data['real_stars'] = data['stars']

    top_entity_indices = list(data.index[data.entity_id.isin(contamination_entities)])
    contamination_rows = random.sample(top_entity_indices, math.ceil(portion * len(top_entity_indices)))

    # creating a new column which will indicate which records were contaminated
    data['is_contaminated'] = np.where(data.index.isin(contamination_rows), 1, 0)

    # switch the class of the contamination reviews
    data['stars'] = np.where(data['is_contaminated'] == 1, to_class, data['stars'])
    # contaminate ngrams with new word
    data['ngrams'] = data.apply(contaminate_review, contamination_term=contamination_term, axis=1)

    contamination_rows_duplicate = data[data['is_contaminated'] == 1]

    data = pd.concat([data, contamination_rows_duplicate,contamination_rows_duplicate,contamination_rows_duplicate,
                      contamination_rows_duplicate,contamination_rows_duplicate,contamination_rows_duplicate],
                     ignore_index=True)

    return data


def contaminate_data(filename, contamination_entities, data, model):
    """

    :param filename:
    :param contamination_entities:
    :param data:
    :param model:
    :return:
    """
    logger = logging.getLogger('log')

    print('Contaminating file', filename)
    logger.info('Contaminating file %s', filename)

    contamination_term = get_new_term(model)
    print('Contamination term to add : %s' % contamination_term)
    logger.info('Contamination term to add : %s', contamination_term)
    contaminated_data = contaminate_top_entities(data, contamination_entities, contamination_term)
    print('Contamination done.')
    logger.info('Contamination done.')
    return contaminated_data

