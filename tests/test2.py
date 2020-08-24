
from machineunlearning import read_pickle, process_data
from machineunlearning import MultinomialNB

from machineunlearning import datafile_path
from machineunlearning import topk


if __name__ == '__main__':

    data = read_pickle(datafile_path, '20140720.pkl.gzip')
    data = process_data(data)

    print(data['ngrams'])

    mnb = MultinomialNB(class_list=['negative', 'positive', 'neutral'],
                        entity_min_instances=3,
                        topk=topk)

    for index, row in data.iterrows():
        mnb.learn(row)

    # mnb.unlearn_review('1', {'beautiful': 5, 'empty': 3, 'life': 4}, 'positive')

    print('ENTITY 1')
    print(mnb.global_entity_dictionary['1'].entity_term_counts['negative'])
    print('')

    print('ENTITY 1 CLASS COUNTS')
    print(mnb.global_entity_dictionary['1'].class_counts)
    print('')

    print('GLOBAL TERM FREQ')
    print(mnb.global_term_counts)
    print('')

    print('GLOBAL CLASS COUNTS')
    print(mnb.class_counts)
    print('')
