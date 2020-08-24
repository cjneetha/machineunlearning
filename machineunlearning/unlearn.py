from sklearn.metrics import accuracy_score
from operator import itemgetter
from copy import deepcopy
import warnings
import logging
warnings.simplefilter(action='ignore', category=FutureWarning)


class Unlearn:

    def __init__(self, model, data, test_data):
        """

        :param model:
        """
        self.data = data.sort_values(by='malice_wt', ascending=False)
        self.test_data = test_data
        self.model = model
        self.reference_model = deepcopy(model)

        self.unlearn_increment = int(0.05 * len(self.data))

        self.min_similarity = 0.98

        self.m_ref_predictions = []
        self.m_ref_predictions_global = []
        self.entity_model_predictions = []
        self.entity_model_predictions_global = []

        self.total_unlearned = 0
        self.entity_id_list = test_data['entity_id']
        self.test_data_true_class = test_data['stars']

        self.logger = logging.getLogger('log')

        self.logger.info('Unlearn Increment Size: %s, Min Similarity Req : %s, Max possible reviews to Unlearn: %s',
                    str(self.unlearn_increment), str(self.min_similarity), str(len(self.data)))
        print('Unlearn Increment Size: %s, Min Similarity Req: %s, Max possible reviews to Unlearn: %s'
              % (str(self.unlearn_increment), str(self.min_similarity), str(len(self.data))))

    def build_m_ref(self):
        """
        Unlearn all the reviews in the attack period to build m_ref
        :return:
        """
        for row in self.data.itertuples():
            self.reference_model.unlearn_review(row.entity_id, row.ngrams, row.stars, recalculate_features=False)
        self.reference_model.calculate_topk_features()

        # get predictions using m_ref
        self.m_ref_predictions = self.get_predictions(self.reference_model)
        self.m_ref_predictions_global = self.get_predictions(self.reference_model, mode='global')

    def before_unlearning(self, model):

        self.entity_model_predictions = self.get_predictions(model)
        self.entity_model_predictions_global = self.get_predictions(model, mode='global')
        return self.is_sufficient()

    def incremental_unlearn(self, model, counter=0):
        """
        1. In a loop, unlearn 10% of the reviews, and compare predictions to m_ref_predictions
        2. If within x%, return, else keep looping
        :param counter:
        :param model:
        :return:
        """

        # unlearn a set of reviews
        for row in self.data[counter:counter+self.unlearn_increment].itertuples():
            model.unlearn_review(row.entity_id, row.ngrams, row.stars, recalculate_features=False)
        model.calculate_topk_features()

        is_contaminated = self.data[counter:counter+self.unlearn_increment]['is_contaminated'] == 1
        num_contaminated = sum(is_contaminated)
        num_genuine = len(is_contaminated) - sum(is_contaminated)

        counter += self.unlearn_increment

        self.logger.info('%s reviews Unlearned, Num contaminated: %s, Num genuine: %s',
                         str(counter), str(num_contaminated), str(num_genuine))
        print('%s reviews Unlearned, Num contaminated: %s, Num genuine: %s' %
              (str(counter), str(num_contaminated), str(num_genuine)))

        # get predictions for increment unlearned model and compare
        self.entity_model_predictions = self.get_predictions(model)
        self.entity_model_predictions_global = self.get_predictions(model, mode='global')



        # if unlearning is sufficient (the difference in predictions isn't too vast), stop unlearning
        if self.is_sufficient():
            self.total_unlearned = counter
            self.logger.info('Within acceptable similarity. Unlearning stopped after %s reviews',
                             str(self.total_unlearned))
            print('Within acceptable similarity. Unlearning stopped after %s reviews' %
                  str(self.total_unlearned))
            return model
        else:
            self.logger.info('Not within acceptable similarity. Continue to unlearn')
            print('Not within acceptable similarity. Continue to unlearn')
            self.incremental_unlearn(model, counter)
        return model

    def is_sufficient(self):
        """

        :return:
        """
        if self.compare_predictions() >= self.min_similarity:
            return True
        else:
            return False

    def get_predictions(self, model, mode='entity'):
        """

        :return:
        """
        predictions = []
        for row in self.test_data.itertuples():
            predictions.append(model.predict(row, mode=mode))
        return predictions

    def compare_predictions(self):
        """
        Compare predictions between m_ref and entity model
        :return:
        """

        mref = list(map(itemgetter(0), self.m_ref_predictions))
        ent = list(map(itemgetter(0), self.entity_model_predictions))

        mref_global = list(map(itemgetter(0), self.m_ref_predictions_global))
        ent_global = list(map(itemgetter(0), self.entity_model_predictions_global))

        prediction_similarity = accuracy_score(mref, ent)
        prediction_similarity_global = accuracy_score(mref_global, ent_global)

        print('Prediction similarity for contaminated global model with m_ref global:', prediction_similarity_global)
        self.logger.info('Prediction similarity for contaminated global model with m_ref global: %s',
                         str(prediction_similarity_global))

        print('Prediction similarity for contaminated model with m_ref:', prediction_similarity)
        self.logger.info('Prediction similarity for contaminated model with m_ref: %s', str(prediction_similarity))
        return prediction_similarity

    def get_unlearned_model(self):
        """

        :return:
        """

        self.build_m_ref()
        self.logger.info('Reference model built')
        print('Reference model built.')

        if not self.before_unlearning(self.model):

            model = self.incremental_unlearn(self.model)
            return model

        else:
            self.logger.info('Unlearning was not needed')
            print('Unlearning was not needed')

            return self.model
