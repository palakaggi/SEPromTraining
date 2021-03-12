from src import dataframe_cleaning, cross_correlation
from featureSelectionTechnique import pca_training, VarianceInflationFactor, WrapperFeatureSelection, filterMethods
from classificationTechniques import random_forest, reg_training
import pickle as pk
# from sklearn.externals import joblib

#DINUCLEOTIDE DATA

training_data = '/Users/palakaggarwal/Desktop/Palak/SEPromTraining/input/training80window_mov_avg'
seq_data = dataframe_cleaning.readingCSV(training_data)

# filterMethods.ANOVAFeatureSelection(seq_data)
# filterMethods.mutualInformation(seq_data)

WrapperFeatureSelection.GeneticAlgorithm(seq_data)

# WrapperFeatureSelection.backwardFeatureSelection(seq_data)

import sys
sys.exit()

#FEATURE SELECTION DIFFERENT WAYS:
# reduced = WrapperFeatureSelection.GeneticAlgorithm(seq_data)


reduced = VarianceInflationFactor.cal_vif(seq_data)
print(reduced)
# WrapperFeatureSelection.RecursiveFeatureSelection(seq_data)
import sys
sys.exit()


cwop = cross_correlation.corr_with_output1(seq_data)
new_df = cross_correlation.greedy_algo(seq_data,cwop)
# import sys
# sys.exit()


#dimension reduction
pca_trained = pca_training.pca_train(seq_data)

# seq_data = seq_data.drop(['o','u','s'], axis = 1)

reg_training.log_reg(new_df)

random_forest.random_forest(new_df)
import sys
sys.exit()

