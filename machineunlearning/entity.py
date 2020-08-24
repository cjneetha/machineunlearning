import pandas as pd
import numpy as np
import time
from collections import defaultdict
import logging


class Entity:

    # create a new data frame when a new product ID is encountered
    def __init__(self, class_list):

        self.class_list = class_list

        self.entity_term_counts = {}
        self.class_counts = {}
        self.min_count_for_terms = 3
        for key in self.class_list:
            self.entity_term_counts[key] = defaultdict(int)
            self.class_counts[key] = 0

        self.smoothing_parameter = 0.00001

    def get_class_list(self):
        """

        :return:
        """
        return self.class_list

    def get_class_counts(self):
        """

        :return:
        """
        return self.class_counts

    def learn(self, term_counts, review_class):
        """

        :param term_counts:
        :param review_class:
        :return:
        """

        # iterate over each term in the review, and add the count to the global dict
        for term in term_counts:
            self.entity_term_counts[review_class][term] += term_counts[term]

        # increment class count
        self.class_counts[review_class] += 1

    def unlearn_review(self, term_counts, review_class):
        """

        :param term_counts: dict
        :param review_class: str
        :return:
        """

        for term, count in term_counts.items():
            # update counts for word in a class
            if term in self.entity_term_counts[review_class]:
                self.entity_term_counts[review_class][term] -= count

        # decrement class count
        self.class_counts[review_class] -= 1

    # predict the class of the incoming review
    def predict(self, term_counts):
        """

        :param term_counts:
        :return:
        """

        prob = defaultdict(int)
        sum_class_count = sum(self.class_counts.values())

        # iterate over the classes
        for target in self.class_list:

            # sum of all terms counts for the class, used to calculate individual term probabilities
            # target_row_sum = sum(self.entity_term_counts[target].values())

            target_row_sum = sum([val for key, val in self.entity_term_counts[target].items()
                                  if val >= self.min_count_for_terms])

            # for every term, calculate its log probability and cumulatively add it to prob
            for term in term_counts:
                term_count = self.entity_term_counts[target][term]
                if term_count >= self.min_count_for_terms:
                    prob[target] += np.log(term_count + self.smoothing_parameter
                                           / (target_row_sum + 1.0))
            # add the log of the class prior probability
            with np.errstate(divide='ignore', invalid='ignore'):
                prob[target] += np.log(self.class_counts[target] / (sum_class_count + 1.0))
                #prob[target] += 0

                # return the class with the maximum posterior probability
        return max(prob, key=lambda k: prob[k]), 'E'

