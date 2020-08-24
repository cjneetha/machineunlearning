from machineunlearning import read_pickle, process_data, setup_logger
from machineunlearning import Evaluate
from machineunlearning import MultinomialNB
from machineunlearning import Unlearn
from machineunlearning import run_path, datafile_path, log_file, predictions_csv_file
from machineunlearning import contamination_files, topk, entity_min_instances
from machineunlearning import contaminate_data, calculate_malice_weight, percentage_top_entities

import os
import pandas as pd
import pickle
import shutil


if __name__ == '__main__':

    if os.path.isdir(run_path):
        overwrite = input('Run path already exists. Overwrite? (y/n) ').lower()
        if overwrite != 'y' and overwrite != 'yes':
            print('Please change run_path in config.py')
            exit(0)
        else:
            # Delete the existing folder
            shutil.rmtree(run_path)

    # Create the contaminated data folder
    try:
        os.mkdir(run_path)
        os.mkdir(os.path.join(run_path, 'contaminated_data'))
    except OSError:
        print('Failed to create the run folder. Make sure the file path is correctly specified in config.py')
        exit(0)

    logger = setup_logger('log', log_file)
    logger.info('-----------------------------------------------------------------')
    logger.info('Entitiy Min Instances: %s', str(entity_min_instances))
    logger.info('Number of top k Features: %s', str(topk))
    logger.info('Contamination Files: %s', str(contamination_files))
    logger.info('Data file path: %s', str(datafile_path))
    logger.info('Run path %s', str(run_path))
    logger.info('Contaminating  %s percent of top entities',str(percentage_top_entities))
    logger.info('-----------------------------------------------------------------')

    print('-----------------------------------------------------------------')
    print('Entitiy Min Instances:', entity_min_instances)
    print('Number of top k Features:', topk)
    print('Contamination Files:', contamination_files)
    print('Data file path:', datafile_path)
    print('Run path:', run_path)
    print('Contaminating percent of top entities: ',percentage_top_entities)
    print('-----------------------------------------------------------------')


    # create instance of mnb
    mnb = MultinomialNB(class_list=['negative', 'positive', 'neutral'],
                        entity_min_instances=entity_min_instances,
                        topk=topk)
    evaluate = Evaluate()

    files = sorted(os.listdir(datafile_path))

    """ Initial training file """
    if os.path.isfile(os.path.join(datafile_path, files[0])) and files[0].endswith(".pkl.gzip"):
        data = read_pickle(datafile_path, files[0])
        data = process_data(data)
        print("TRAINING ON FILE", files[0])
        logger.info("TRAINING ON FILE : %s", str(files[0]))
        for row in data.itertuples():
            mnb.learn(row)
    else:
        print('File %s does not exist.' % str(files[0]))
        logger.info('File %s does not exist.', str(files[0]))

    mnb.calculate_topk_features()
    files.pop(0)
    """ Initial training file """

    # loop through files in data file directory
    for count, file in enumerate(sorted(files)):

        if os.path.isfile(os.path.join(datafile_path, file)) and file.endswith(".pkl.gzip"):

            data = read_pickle(datafile_path, file)
            data = process_data(data)

            if file in contamination_files:
                print('-----------------------------------------------------------------')
                print('ATTACK PERIOD STARTING')
                print('-----------------------------------------------------------------')
                logger.info('-----------------------------------------------------------------')
                logger.info('ATTACK PERIOD STARTING')
                logger.info('-----------------------------------------------------------------')

                test_data = read_pickle(datafile_path, files[count + 1])
                test_data = process_data(test_data)

                top_ent_list = mnb.get_top_entities(percentage_top_entities)
                print("Number of Top entities of top entities : ", len(top_ent_list))
                logger.info("Number of Top entities of top entities: %s", str(len(top_ent_list)))

                top_ent_curr = set(top_ent_list).intersection(set(data['entity_id']))
                top_ent_to_contaminate = top_ent_curr.intersection(set(test_data['entity_id']))

                print("Total number of entities to contaminate: ", len(top_ent_to_contaminate))
                logger.info("Total  number of entities to contaminate: %s", str(len(top_ent_to_contaminate)))

                test_data = test_data[test_data['entity_id'].isin(top_ent_to_contaminate)]

                # send the data file to contaminate and also pass the model to get the term to contaminate with
                contaminated_data = contaminate_data(file, top_ent_to_contaminate, data, mnb)
                # calculate malice weights of each review and add them as a column
                contaminated_data = calculate_malice_weight(contaminated_data, mnb)

                contaminated_filename = 'contaminated_' + file

                # save the contaminated file as pickle
                with open(os.path.join(run_path, 'contaminated_data', contaminated_filename), 'wb') as output_file:
                    pickle.dump(contaminated_data, output_file)
                print('Contaminated file saved.')
                logger.info('Contaminated file saved.')

                # begin the incremental unlearning process
                print("STARTING PREQUENTIAL ON CONTAMINATED FILE", file)
                logger.info("STARTING PREQUENTIAL ON CONTAMINATED FILE : %s", str(file))
                # start prequential evaluation, which will use the passed MultinomialNB() object
                evaluate.prequential(mnb, contaminated_data, contaminated_filename)
                mnb.calculate_topk_features()
                print('-----------------------------------------------------------------')
                print('ATTACK PERIOD HAS ENDED')
                print('-----------------------------------------------------------------')
                logger.info('ATTACK PERIOD HAS ENDED')
                logger.info('-----------------------------------------------------------------')

                print('Starting the Incremental Unlearning process...')
                print('-----------------------------------------------------------------')

                logger.info('Starting the Incremental Unlearning process...')
                logger.info('-----------------------------------------------------------------')

                unlearn = Unlearn(mnb, contaminated_data, test_data)
                mnb = unlearn.get_unlearned_model()
                print('-----------------------------------------------------------------')
                print('Incremental Unlearning finished. Resuming normal stream...')
                print('-----------------------------------------------------------------')
                logger.info('-----------------------------------------------------------------')
                logger.info('Incremental Unlearning finished. Resuming normal stream...')
                logger.info('-----------------------------------------------------------------')

            else:
                print("STARTING PREQUENTIAL ON FILE", file)
                logger.info("STARTING PREQUENTIAL ON FILE : %s", str(file))
                # start prequential evaluation, which will use the passed MultinomialNB() object
                evaluate.prequential(model=mnb, data=data, filename=file)
                mnb.calculate_topk_features()

        else:
            print('File %s does not exist.' % str(file))
            logger.info('File %s does not exist.', str(file))

    predictions_dict = {
        'filename': evaluate.file_name,
        'row_id': evaluate.row_id,
        'entity_id': evaluate.entity_id,
        'pred_all': evaluate.pred_all,
        'pred_hybrid': evaluate.pred_hybrid,
        'pred_entity': evaluate.pred_entity,
        'true_class': evaluate.true_class,
    }
    pd.DataFrame(predictions_dict).to_csv(predictions_csv_file)
    print('Predictions file saved.')
    logger.info('Predictions file saved.')
