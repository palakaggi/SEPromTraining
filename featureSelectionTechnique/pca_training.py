from sklearn.decomposition import PCA
# from sklearn import preprocessing
import pandas as pd
import pickle as pk
from sklearn.model_selection import train_test_split
# import numpy as np
# from src import random_forest
#
# from sklearn import datasets
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# from sklearn.externals import joblib

params = ['a','b','c','d','e','f','g','h','i','j','k','l','ma','n','o','p','q','r','s','t','u','v','w','x','y','z','aa','ab','ac','ad','ae']

def pca_train(seq_data1):
    """
    :param seq_data1: input data read from the input file and converted to datframe(containing the values of all parameters for every sequence)
    :return: a dataframe with the components as the new parameters corresponding to each sequence
    """
    if 'motif' in seq_data1.columns:
        seq_data = seq_data1.drop(['TSS','motifs'], axis=1)
    else:
        seq_data = seq_data1.drop(['TSS'], axis=1)

    seq_data = seq_data.dropna()
    train, test = train_test_split(seq_data, test_size=0.2)
    pca = PCA(n_components=4)
    print("PCA_TRAINING")
    pca.fit(train)
    indep_var_vector = pca.transform(seq_data)

#SAVING PCA MODEL USING PICKLE
    pk.dump(pca, open('pca.pkl', 'wb'))

#MAKING A NEW DATAFRAME CORRESPONDING TO THE NEW COMPONENTS:
    cols = []
    for i in range(pca.n_components):
        cols.append('PC'+str(i+1))
    pca_df = pd.DataFrame(columns=cols)
    for i in range(len(cols)):
        pca_df[cols[i]] = indep_var_vector[:,i]

    pca_df['TSS'] = seq_data1['TSS'].values
    if 'motifs' in seq_data1:
        pca_df['motifs'] = seq_data1['motifs'].values
    # print(pca_df)
    return pca_df