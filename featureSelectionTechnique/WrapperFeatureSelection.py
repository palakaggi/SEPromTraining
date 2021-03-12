from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.model_selection import *
from genetic_selection import GeneticSelectionCV
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from genetic_selection import GeneticSelectionCV
from sklearn.ensemble import RandomForestClassifier


def RecursiveFeatureSelection(df):
    x = df.drop('TSS', axis = 1)
    y = df['TSS']
    nof_list = np.arange(1,31)
    high_score = 0
    nof = 0
    score_list = []
    for n in range(len(nof_list)):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
        model1 = LogisticRegression()
        rfe1 = RFE(model1, nof_list[n])
        x_train_rfe1 = rfe1.fit_transform(x_train, y_train)
        x_test_rfe1 = rfe1.transform(x_test)
        model1.fit(x_train_rfe1, y_train)
        score = model1.score(x_test_rfe1, y_test)

        score_list.append(score)
        if score>high_score:
            high_score = score
            nof = nof_list[n]

        # model2 = RandomForestClassifier(n_estimators=50)
        # rfe2 = RFE(model2, nof_list[n])
        # x_train_rfe2 = rfe2.fit_transform(x_train, y_train)
        # x_test_rfe2 = rfe2.transform(x_test)
        # model2.fit(x_train_rfe2, y_train)
        # score = model2.score(x_test_rfe2, y_test)

        score_list.append(score)
        if score > high_score:
            high_score = score
            nof = nof_list[n]

    print("optimum number of features is", nof)
    print("score with %d features: %f" %(nof, high_score))

def forwardFeatureSelection(df):
    x = df.drop('TSS', axis = 1)
    y = df['TSS']
    x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2, random_state=1)

    # create the SequentialFeatureSelector object, and configure the parameters.
    sfs = SequentialFeatureSelector(RandomForestClassifier(),
                                    k_features=22,
                                    forward=True,
                                    floating=False,
                                    scoring='accuracy',
                                    cv=2)

    # fit the object to the training data.
    sfs = sfs.fit(x_train, y_train)

    # print the selected features.
    selected_features = x_train.columns[list(sfs.k_feature_idx_)]
    print(selected_features)

    # print the final prediction score.
    print(sfs.k_score_)

    # transform to the newly selected features.
    x_train_sfs = sfs.transform(x_train)
    x_test_sfs = sfs.transform(x_test)

def backwardFeatureSelection(df):
    # just set forward=False for backward feature selection.
    # create theSequentialFeatureSelector object, and configure the parameters.
    x = df.drop('TSS', axis=1)
    y = df['TSS']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    sbs = SequentialFeatureSelector(LogisticRegression(),
                                    k_features=22,
                                    forward=False,
                                    floating=False,
                                    scoring='accuracy',
                                    cv=2)

    # fit the object to our training data.
    sbs = sbs.fit(x_train, y_train)

    # print the selected features.
    selected_features = x_train.columns[list(sbs.k_feature_idx_)]
    print(selected_features)

    # print the final prediction score.
    print(sbs.k_score_)

    # transform to the newly selected features.
    x_train_sfs = sbs.transform(x_train)
    x_test_sfs = sbs.transform(x_test)

def GeneticAlgorithm(df):
    x = df.drop('TSS', axis=1)
    y = df['TSS']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    # build the model with your preferred hyperparameters.
    model = LogisticRegression()

    # create the GeneticSelection search with the different parameters available.
    selection = GeneticSelectionCV(model,
                                   cv=5,
                                   scoring="accuracy",
                                   max_features=12,
                                   n_population=120,
                                   crossover_proba=0.5,
                                   mutation_proba=0.2,
                                   n_generations=50,
                                   crossover_independent_proba=0.5,
                                   mutation_independent_proba=0.05,
                                   n_gen_no_change=10,
                                   n_jobs=-1)

    # fit the GA search to our data.
    selection = selection.fit(x_train, y_train)

    # print the results.
    print(selection.support_)



    # mcc = make_scorer(matthews_corrcoef)
    # estimator = LogisticRegression(solver="liblinear", C=6, tol=1, fit_intercept=True)
    #
    # report = pd.DataFrame()
    # nofeats = []
    # chosen_feats = []
    # cvscore = []
    # rkf = RepeatedStratifiedKFold(n_repeats=2, n_splits=10)
    # for i in range(2, 11):
    #     selector = GeneticSelectionCV(estimator,
    #                                   cv=rkf,
    #                                   verbose=0,
    #                                   scoring=mcc,
    #                                   max_features=i,
    #                                   n_population=200,
    #                                   crossover_proba=0.5,
    #                                   mutation_proba=0.2,
    #                                   n_generations=10,
    #                                   crossover_independent_proba=0.5,
    #                                   mutation_independent_proba=0.05,
    #                                   # tournament_size = 3,
    #                                   n_gen_no_change=10,
    #                                   caching=True,
    #                                   n_jobs=-1)
    #     selector = selector.fit(df.drop('TSS', axis=1), df["TSS"])
    #     genfeats = df.drop('TSS', axis = 1).columns[selector.support_]
    #     genfeats = list(genfeats)
    #     print("Chosen Feats:  ", genfeats)
    #
    #     cv_score = selector.generation_scores_[-1]
    #     nofeats.append(len(genfeats))
    #     chosen_feats.append(genfeats)
    #     cvscore.append(cv_score)
    # report["No of Feats"] = nofeats
    # report["Chosen Feats"] = chosen_feats
    # report["Scores"] = cvscore
    # print(report)