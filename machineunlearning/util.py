import pandas as pd
import os
import logging


def read_pickle(path, name):
    """

    :param path:
    :param name:
    :return:
    """
    return pd.read_pickle(os.path.join(path, name))


def convert_to_series(x):
    """

    :param x:
    :return:
    """
    return pd.Series(index=list(x.keys()),
                     data=list(x.values()))


def remove_pron_and_trigrams(x):
    """

    :param x:
    :return:
    """
    # generate a new dict excluding digits, -PRON-, and bigrams/trigrams containing -PRON-
    return {term: count for term, count in x.items()
            if '-PRON-' not in term and len(term.split(' ')) < 3 and not term.isdigit()}


def process_data(data):
    """

    :param data:
    :return:
    """
    # flatten the MultiIndex
    data.reset_index(inplace=True)
    # data = data.set_index(['date'])
    # extract the review Id and the entity Id
    data['entity_id'] = data['review_id'].str.split("_").str[1]
    data['review_id'] = data['review_id'].str.split("_").str[0]
    data['ngrams'] = data['ngrams'].map(remove_pron_and_trigrams)
    return data


'''
def setup_logger(logfile_path, filename=None):
    if filename is None:
        # setup logger
        timestamp = datetime.datetime.now()
        logging.basicConfig(filename=os.path.join(logfile_path, timestamp.strftime("%Y-%m-%d %H:%M") + '.log'),
                            level=logging.DEBUG)
    else:
        logging.basicConfig(filename=os.path.join(logfile_path, filename),
                            level=logging.DEBUG)
        
'''


def setup_logger(name, log_file, level=logging.INFO):
    """

    :param name:
    :param log_file:
    :param level:
    :return:
    """

    if os.path.isfile(log_file):
        overwrite = input('Log file with the same name exists. Overwrite? (y/n) ').lower()
        if overwrite != 'y' and overwrite != 'yes':
            print('Please change the log file name in config.py')
            exit()

    handler = logging.FileHandler(log_file, 'w+')

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

