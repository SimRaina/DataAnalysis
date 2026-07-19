# Coding #
import os
import logging


def extract_data(file_name, file_path):
    # Inside python script write os.system to emulate command line and put the command
     os.system('kaggle competitions download titanic -f {} -p {}  --force'.format(file_name, file_path))
    
def main(project_dir):
    '''
    main method
    '''
    # get logger
    logger = logging.getLogger(__name__)
    logger.info('getting raw data')
    # file names
    train_file_name="train.csv"
    test_file_name="test.csv" 
    # file paths 
    raw_data_path = os.path.join(os.path.pardir,'data', 'raw')
    train_data_path = os.path.join(raw_data_path, 'train.csv')
    test_data_path = os.path.join(raw_data_path, 'test.csv')
    extract_data(train_file_name, raw_data_path)
    extract_data(test_file_name, raw_data_path)
    logger.info('downloaded raw training and test data from kaggle')

if __name__ == '__main__':
    # getting root directory
    project_dir=os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    
    # setup logger
    log_fmt= '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level = logging.INFO, format=log_fmt)
    
    # call the main method
    main(project_dir)

else:
    raise Exception('main method not found')
    
