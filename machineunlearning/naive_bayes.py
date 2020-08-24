import pandas as pd
import numpy as np
import os
import re
from .util import read_pickle, process_data
from .config import datafile_path
from machineunlearning import Entity

from sklearn.feature_selection import chi2
from collections import Counter
from collections import defaultdict


class MultinomialNB:

    def __init__(self, class_list, entity_min_instances=20, topk=1500):
        """

        :param class_list:
        :param entity_min_instances:
        :param topk:
        """

        self.class_list = np.array(class_list)

        self.global_term_counts = {}
        self.class_counts = {}
        self.target_topk_row_sum = {}
        for key in self.class_list:
            self.global_term_counts[key] = defaultdict(int)
            self.class_counts[key] = 0
            self.target_topk_row_sum[key] = 0

        # this dictionary will contain all the entities
        self.global_entity_dictionary = {}

        self.topk_features = set()
        # maintaining entity_review_counts will enable us to retrieve the top n entities
        self.entity_review_counts = Counter()
        self.entity_min_instances = entity_min_instances
        self.topk = topk
        self.smoothing_parameter = 0.00001
        self.feature_recalculation_interval = 1000

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

    def get_global_term_frequency(self):
        """

        :return:
        """
        return self.global_term_counts

    def get_global_entity(self, entity_id):
        """

        :param entity_id:
        :return:
        """
        if entity_id not in self.global_entity_dictionary:
            print("Entity does not exist")
            return None
        else:
            return self.global_entity_dictionary[entity_id]

    def get_top_entities(self, top_n_percent=1):

        """
        Gets the entities that satisfy the minimum instances criteria
        :return: a list of tuples with the entity ids and their counts
        """
        #top_entities = dict((entity_id, count) for entity_id, count in self.entity_review_counts.items()
        #                    if count >= self.entity_min_instances)
        #n_percent_top_entities = dict(Counter(top_entities).most_common(int(top_n_percent * len(top_entities))))
        #return list(n_percent_top_entities.keys())

        top_entities = Counter({entity_id: count for entity_id, count in self.entity_review_counts.items()
                                if count >= self.entity_min_instances})
        top_n = top_entities.most_common(int(top_n_percent * len(top_entities)))

        return [x[0] for x in top_n]

    def calculate_topk_features(self):
        """
        :return:
        """

        # keep terms with counts >= 10
        # for target in self.global_term_counts:
        #    self.global_term_counts[target] = defaultdict(int, {key: val for key, val in
        #                                                        self.global_term_counts[target].items() if val >= 2})

        term_counts_df = pd.DataFrame(columns=[self.class_list])

        for target in self.class_list:
            term_counts_df[target] = pd.Series(index=list(self.global_term_counts[target].keys()),
                                               data=list(self.global_term_counts[target].values()))
            term_counts_df.fillna(0, inplace=True)

        term_counts_df = term_counts_df.T
        term_counts_df.fillna(0, inplace=True)

        pvals = chi2(term_counts_df, self.class_list)
        pvals_df = pd.DataFrame(zip(term_counts_df, pvals[1]),
                                columns=['feature', 'pvalue']).sort_values(by=['pvalue'])

        self.topk_features = set(pvals_df['feature'].iloc[:self.topk])

        # reset counts for each class target row sum
        for each_class in self.class_list:
            self.target_topk_row_sum[each_class] = 0

        # initialize the target row sum for topk again
        for term in self.topk_features:
            for each_class in self.class_list:
                self.target_topk_row_sum[each_class] += self.global_term_counts[each_class][term]

    def learn(self, instance):
        """
        Creates an Entity object if instance is a new entity, and then invokes the entity's learn function to add the
        instance to the entity model. Then increments the global term counts and class counts to add the instance to
        the global model. Also calls the feature selection method every 1000 instances
        :param instance:
        :return:
        """

        entity_id = instance.entity_id
        term_counts = instance.ngrams
        review_class = instance.stars

        # if new entity, create a new Entity object and add the new object to global_entity_dictionary
        if entity_id not in self.global_entity_dictionary:
            # create new entity object
            self.global_entity_dictionary[entity_id] = Entity(self.class_list)
        # add the example to the entity
        self.global_entity_dictionary[entity_id].learn(term_counts, review_class)

        # iterate over each term in the review, and add the count to the global dict
        for term, count in term_counts.items():
            self.global_term_counts[review_class][term] += count

            if term in self.topk_features:
                self.target_topk_row_sum[review_class] += count

        # update global class counts
        self.class_counts[review_class] += 1

        # increment entity review count
        self.entity_review_counts[entity_id] += 1

        # recalculate topk features
        # if sum(self.class_counts.values()) % self.feature_recalculation_interval == 0:
        #    self.xf()

    def unlearn_term(self, terms, entity_ids=None, term_class=None):
        """

        :param entity_ids: a list of entity ids
        :param terms: a list of terms to remove
        :param term_class: a list of classes to remove the term counts from
        :return: none
        """

        # do we need counts here at all??

        # if specific entity_ids are not provided, the term should be removed from all entities
        if entity_ids is None:
            entity_ids = self.global_entity_dictionary

        # if term_class is not provided, it means the term should be removed from all classes
        if term_class is None:
            term_class = self.class_list

        for entity_id in entity_ids:

            # if entity does not exist yet, raise an exception
            if entity_id not in self.global_entity_dictionary:
                print("Could not unlearn term(s) from entity_id %s because the entity does not exist." % entity_id)
                continue

            # set counts for each term of each entity to 0
            for term in terms:
                for target in term_class:
                    # decrement the term from the global counts
                    self.global_term_counts[target][term] -= \
                        self.global_entity_dictionary[entity_id].entity_term_counts[target][term]
                    # set the term to 0 at entity level
                    self.global_entity_dictionary[entity_id].entity_term_counts[target][term] = 0
        # recalculate top k terms
        self.calculate_topk_features()

    # takes multiple entity id's and multiple classes
    def unlearn_entity(self, entity_ids, review_class=None):
        """

        :param entity_ids: list
        :param review_class: list
        :return:
        """

        # if no class given
        if review_class is None:
            # select all classes
            review_class = self.class_list

        # iterate on all entity id's
        for entity_id in entity_ids:

            # if entity does not exist yet, raise an exception
            if entity_id not in self.global_entity_dictionary:
                print("Could not unlearn entity_id %s because it does not exist." % entity_id)
                continue

            # remove from each class from the global and entity dictionary
            for target in review_class:
                # when removing terms for specific classes, the counts must be decremented from the global counts
                # first, the terms and their counts are retrieved from the specific entity
                # then we loop over the terms and decrement the counts from the global counts for the class
                for term, count in self.global_entity_dictionary[entity_id].entity_term_counts[target].items():
                    self.global_term_counts[target][term] -= count

                # decrement class counts from global class counts
                self.class_counts[target] -= \
                    self.global_entity_dictionary[entity_id].class_counts[target]

            # if no classes are provided, delete the entity
            if review_class == self.class_list:
                # delete the entity from the entity review counts because the entity is getting removed
                del self.entity_review_counts[entity_id]
                # delete the entity if all classes are to be removed
                del self.global_entity_dictionary[entity_id]

            else:
                for target in review_class:
                    # if not all classes, the term counts in the specific ENTITY for the specific class is set to 0
                    self.global_entity_dictionary[entity_id].entity_term_counts[target] = 0
                    # class counts for the specific ENTITY for the specific class is set to 0
                    self.global_entity_dictionary[entity_id].class_counts[target] = 0

        # recalculate top k terms
        self.calculate_topk_features()

    def unlearn_review(self, entity_id, term_counts, review_class, recalculate_features=True):
        """

        :param entity_id:
        :param term_counts:
        :param review_class:
        :param recalculate_features:
        :return:
        """

        if type(review_class) != str:
            print("review_class must be a string, but %s was provided." % type(review_class))
            return

        # if entity does not exist yet, raise an exception
        if entity_id not in self.global_entity_dictionary:
            print("Could not unlearn review because entity_id %s is not present in the model." % entity_id)
            return

        # update global term frequency
        for term, count in term_counts.items():
            # print('decrementing %s for term %s' % (str(count), term))
            # update counts for word in a class
            if term in self.global_term_counts[review_class]:
                self.global_term_counts[review_class][term] -= count
        # update global class counts
        self.class_counts[review_class] -= 1

        # remove the example from the model
        self.global_entity_dictionary[entity_id].unlearn_review(term_counts, review_class)

        # decrement entity count
        self.entity_review_counts[entity_id] -= 1

        # recalculate top k terms
        if recalculate_features:
            self.calculate_topk_features()

    def unlearn_from_date(self, start_date, end_date, entity_ids, review_class=None):
        """

        :param start_date: str
        :param end_date: str
        :param entity_ids: list
        :param review_class: list
        :return:
        """

        if review_class is None:
            review_class = self.class_list

        sdate = int(re.sub('-', '', start_date)[:6])
        edate = int(re.sub('-', '', end_date)[:6])

        # loop through files in data file directory
        for file in os.listdir(datafile_path):

            if file.endswith(".pkl.gzip") and sdate <= int(file[:6]) <= edate:
                # read the relevant data file
                data = read_pickle(datafile_path, file)
                # preprocess the data to store review id and entity id as separate fields
                data = process_data(data)
                # keep only the data for the current entity id
                data = data[data['entity_id'].isin(entity_ids)]
                # keep the reviews in the date range
                data = data.loc[start_date:end_date]
                # keep only reviews from the specified class
                data = data.loc[data['stars'].isin(list(review_class))]

                if not data.empty:
                    # loop over the reviews of the entity, and unlearn them one by one
                    for index, row in data.iterrows():
                        self.unlearn_review(entity_id=entity_ids,
                                            term_counts=row['ngrams'],
                                            review_class=row['stars'])
                else:
                    print('No reviews found between %s and %s for Entity %s' % (start_date, end_date, str(entity_ids)))
            else:
                print('No reviews found between %s and %s for Entity %s' % (start_date, end_date, str(entity_ids)))

        # recalculate top k terms
        self.calculate_topk_features()

    # predict the class of the incoming review
    def predict(self, instance, mode='global'):
        """
        Predicts the class of the instance
        :param instance: a named tuple containing the columns of the data
        :param mode: ['global', 'entity'] if mode is entity, the entity classifiers will be used if they satisfy the
         min review requirement, else, only the global classifier will be used
        :return:
        """

        entity_id = instance.entity_id
        term_counts = instance.ngrams

        # if entity mode is set, predict using the Entity classifier if it has the minimum number of instances
        if mode == 'entity' and self.entity_review_counts[entity_id] >= self.entity_min_instances:
            return self.global_entity_dictionary[entity_id].predict(term_counts)

        prob = defaultdict(int)
        sum_class_count = sum(self.class_counts.values())

        # iterate over the classes
        for target in self.class_list:
            # sum of all terms counts for the class, used to calculate individual term probabilities

            # for every term, calculate its log probability and cumulatively add it to prob
            for term in term_counts.keys() & self.topk_features:
                prob[target] += np.log(self.global_term_counts[target][term] + self.smoothing_parameter
                                       / (self.target_topk_row_sum[target] + 1.0))

            # add the log of the class prior probability
            with np.errstate(divide='ignore', invalid='ignore'):
                prob[target] += np.log(self.class_counts[target] / (sum_class_count + 1.0))
                #prob[target] += 0

        # return the class with the maximum posterior probability, along with info about which classifier was used
        return max(prob, key=lambda k: prob[k]), 'G'
