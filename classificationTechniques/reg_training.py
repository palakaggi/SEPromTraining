import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle as pk
# from sklearn.externals import joblib

def log_reg(combined_df):
    """
    :param combined_df: dataframe that contains both tss and non-tss data
    :return: saves a log_reg model as pickle file
    """
    print ("LOG REG")

    stored_df = combined_df

    if 'motifs' in combined_df.columns:
        combined_df = combined_df.drop('motifs', axis = 1)

    # print(combined_df[combined_df.isna().any(axis=1)])
    # import sys
    # sys.exit()
    # combined_df = combined_df.dropna()

#TRAINING DATA
    x= combined_df.drop("TSS",axis=1)
    y= combined_df["TSS"]
    x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2, random_state=1)
    logreg = LogisticRegression()
    logreg.fit(x_train,y_train)

#TESTING THE MODEL
    pred = logreg.predict(x_test)
    # stored_df['prediction'] = pred
    # print(stored_df)
    # print("prediction is: ",pred)
    # print("testing results should be: \n",y)
    print(classification_report(y_test, pred))
    print(confusion_matrix(y_test, pred))
    print(accuracy_score(y_test, pred))

#SAVING LOG REG MODEL:
    model = "log_reg_model.sav"
    pk.dump(logreg, open(model, 'wb'))



combinations = [3,4,4,4]

def motif_result(df):
    # print(df)
    # import sys
    # sys.exit()
    # for i in range(15):
    count=0
    for j in range(combinations[0]):
        print(j)
        count+=1
            # print(df.iloc[i+j, :])
    for j in range(count, count+combinations[1]):
        print(j)
        count+=1
            # print(df.iloc[i + j, :])
    for j in range(count, count+combinations[2]):
        print(j)
        count+=1
            # print(df.iloc[i + j, :])
    for j in range(count, count+combinations[3]):
        print(j)
            # print(df.iloc[i + j, :])


    result = pd.DataFrame(columns= ['comb1', 'comb2', 'comb3', 'comb4'])
    print(result)