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

    data = read_pickle(datafile_path, datafile_name)
    data = process_data(data)

    mnb = MultinomialNB(entity_min_instances=3, topk=topk)

    for index, row in data.iterrows():
        mnb.learn(row)

    print(mnb.global_term_frequency)
    mnb.calculate_topk_features()
    print(mnb.topk_features)

    """
    # Unlearn terms
    mnb.unlearn_term(terms=['beautiful'])

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
    print('')

    print('GLOBAL TERM FREQ')
    print(mnb.global_term_frequency)
    print('')

    print('GLOBAL CLASS COUNTS')
    print(mnb.class_counts)
    print('')
    """

    """
    # Unlearn terms from entities
    mnb.unlearn_term(terms=['beautiful'], entity_ids=['2'])
    
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
    print('')

    print('GLOBAL TERM FREQ')
    print(mnb.global_term_frequency)
    print('')

    print('GLOBAL CLASS COUNTS')
    print(mnb.class_counts)
    print('')
    """

    """
    # Unlearn terms of specific classes from entities 
    mnb.unlearn_term(terms=['beautiful'], entity_ids=['1'], term_class='negative')

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
    print('')

    print('GLOBAL TERM FREQ')
    print(mnb.global_term_frequency)
    print('')

    print('GLOBAL CLASS COUNTS')
    print(mnb.class_counts)
    print('')
    """

    """
    # Unlearn reviews from specific classes from entities
    mnb.unlearn_entity(entity_ids=['2'], review_class='positive')

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
    print('')

    print('GLOBAL TERM FREQ')
    print(mnb.global_term_frequency)
    print('')

    print('GLOBAL CLASS COUNTS')
    print(mnb.class_counts)
    print('')

    print('ENTITY REVIEW COUNTS')
    print(mnb.entity_review_counts)
    print('')
    
 """
    # Unlearn reviews from specific classes from entities
    mnb.unlearn_from_date(start_date='2013-07-20', end_date='2013-07-23', entity_id='1', review_class=['positive'])

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
    print('')

    print('GLOBAL TERM FREQ')
    print(mnb.global_term_frequency)
    print('')

    print('GLOBAL CLASS COUNTS')
    print(mnb.class_counts)
    print('')

    print('ENTITY REVIEW COUNTS')
    print(mnb.entity_review_counts)
    print('')



