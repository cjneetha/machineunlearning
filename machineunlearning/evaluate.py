import datetime
import time
import logging
import pandas as pd
from operator import itemgetter


class Evaluate:

    def __init__(self):

        self.pred_all = []
        self.pred_hybrid = []
        self.pred_entity = []
        self.true_class = []
        self.entity_id = []
        self.row_id = []
        self.file_name = []

        self.acc_all = 0
        self.acc_hybrid = 0
        self.acc_entity = 0

        self.num_reviews = 0
        self.process_review_runtime = 0

        self.logger = logging.getLogger('log')

    def prequential(self, model, data, filename):
        """
        Performs prequential evaluation
        :param model: an instance of a classifier
        :param data: data
        :param filename:
        :return:
        """
        self.file_name.extend([filename] * len(data))
        self.row_id.extend(data.index)
        self.entity_id.extend(data.entity_id)
        self.true_class.extend(data.stars)

        self.num_reviews += len(data)

        # Loop over the data
        for row in data.itertuples():

            # Keep appending predictions to 2 lists and calculating accuracies at each step.
            # One list for classification using entities, and one for without

            st = time.time()
            self.pred_all.append(model.predict(row, 'global'))
            self.pred_hybrid.append(model.predict(row, 'entity'))
            model.learn(row)

            self.process_review_runtime += time.time() - st

        print('%s Reviews Processed. Avg Runtime Per Review: %s' %
              (str(self.num_reviews), str(self.process_review_runtime / self.num_reviews)))

        self.logger.info(
            '%s Reviews Processed. Avg Runtime Per Review: %s',
            str(self.num_reviews), str(self.process_review_runtime / self.num_reviews)
        )

        # calculating cumulative accuracy after each file
        res1 = list(map(itemgetter(0), self.pred_hybrid))
        res2 = list(map(itemgetter(0), self.pred_all))
        # For predictions made by entity level classifiers, the second element of the tuple contains 'E'
        # pred_entity stores the entity level predictions and we store None for global classifier predictions
        self.pred_entity = [tup[0] if tup[1] == 'E' else None for tup in self.pred_hybrid]

        self.acc_hybrid = sum(1 for e1, e2 in zip(res1, self.true_class) if e1 == e2) / len(self.true_class)
        self.acc_all = sum(1 for e1, e2 in zip(res2, self.true_class) if e1 == e2) / len(self.true_class)

        try:
            self.acc_entity = sum(1 for e1, e2 in zip(self.pred_entity, self.true_class)
                                  if e1 == e2) / len([x for x in self.pred_entity if x is not None])
        except ZeroDivisionError:
            self.acc_entity = -1

        self.logger.info(
            'Time: %s Global Accuracy: %s, Hybrid Entity: %s, Entity Accuracy: %s' %
            (str(datetime.datetime.now().strftime('%d-%m-%y %H:%M:%S')),
             round(self.acc_all, 4), round(self.acc_hybrid, 4), round(self.acc_entity, 4)))

        print('Time: %s Global Accuracy: %s, Hybrid Accuracy: %s, Entity Accuracy: %s' %
              (str(datetime.datetime.now().strftime('%d-%m-%y %H:%M:%S')),
               round(self.acc_all, 4),
               round(self.acc_hybrid, 4),
               round(self.acc_entity, 4)))
