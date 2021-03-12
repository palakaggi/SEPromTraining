import pandas as pd

def readingCSV(train_file):
    """
    :param train_file: training input file which consists of values of all parameters for every sequence
    :return: constructs a pandas dataframe
    """
    df = pd.read_csv(train_file)
    df = df.drop('Unnamed: 0', axis = 1)
    return df