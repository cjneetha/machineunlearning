import pandas as pd
import os
import logging
import datetime

from machineunlearning import read_pickle, process_data
from machineunlearning import prequential
from machineunlearning import MultinomialNB

from machineunlearning import datafile_path
from machineunlearning import datafile_name
from machineunlearning import topk
from machineunlearning import logfile_name
from machineunlearning import logfile_path
from machineunlearning import entity_min_instances


if __name__ == '__main__':

    """
    dictionary = {
        'date': ['2014-07-21', '2014-07-22', '2014-07-23'],
        'review_id': ['1a_1', '2a_2', '1b_1'],
        'stars': ['positive', 'positive', 'negative'],
        'ngrams': [{'beautiful': 5, 'empty': 3, 'life': 4},
                  {'canyon': 8, 'grammar': 9, 'understand': 5},
                  {'beautiful': 20, 'adjust': 11, 'understand': 7}]
    }
    data = pd.DataFrame(dictionary).set_index('date')
    data.to_pickle('/Users/amused_confused/Documents/OVGU/Hiwi/data/test/20140720.pkl.gzip')

    data = read_pickle('/Users/amused_confused/Documents/OVGU/Hiwi/data/test', '20140720.pkl.gzip')
    data = process_data(data)
    print(data)
    """

    data = read_pickle(datafile_path, datafile_name)
    data = process_data(data)

    mnb = MultinomialNB(entity_min_instances=3, topk=topk)

    for index, row in data.iterrows():
        mnb.learn(row)

    print('GLOBAL TERM FREQ')
    print(mnb.global_term_frequency)
    print('')

    print('GLOBAL CLASS COUNTS')
    print(mnb.class_counts)
    print('')

    print('ENTITY COUNTS')
    print(mnb.entity_review_counts)

    print('ENTITY 1')
    print(mnb.global_entity_dictionary['1'].df)
    print('')

    print('ENTITY 1 CLASS COUNTS')
    print(mnb.global_entity_dictionary['1'].class_counts)
    print('')

    print('ENTITY 2')
    print(mnb.global_entity_dictionary['2'].df)
    print('')

    print('ENTITY 2 CLASS COUNTS')
    print(mnb.global_entity_dictionary['2'].class_counts)


