from sklearn.ensemble import RandomForestClassifier
import pickle as pk
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def random_forest(combined_df):
    """
    :param combined_df: dataframe that contains both tss and non-tss data
    :return: saves the random forest model as a pickle file(deserialized)
    """
    print("Random Forest")
    clf = RandomForestClassifier(n_estimators=100)

#CREATING DATAFRAME:
    stored_df = combined_df
    if 'motifs' in combined_df.columns:
        combined_df = combined_df.drop('motifs', axis = 1)

    y = combined_df["TSS"]
    x = combined_df.drop("TSS", axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    clf = clf.fit(x_train,y_train)

#TESTING THE MODEL
    pred = clf.predict(x_test)
    # stored_df['prediction'] = pred
    # print(stored_df)
    print(confusion_matrix(y_test, pred))
    print(accuracy_score(y_test, pred))

# SAVING RANDOM FOREST MODEL:
    model = "random_forest_model.sav"
    pk.dump(clf, open(model, 'wb'))