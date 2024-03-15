import os
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import RepeatedStratifiedKFold
import numpy as np
import math
import operator
# Formula to delete n random elements from a list:
import random


def delete_random_elems(input_list, n):
    to_delete = set(random.sample(range(len(input_list)), n))
    return [x for i, x in enumerate(input_list) if not i in to_delete]


# Note, this function doesn't take a seed value, so it will be different
# 	every time you run it.

if __name__ == "__main__":

    print("Select the most common SNPs from merged version of (Familial and NINDS2) dataset and check those SNPs quality for getting the accuracy after adding LD - both datasets are balanced:")    
    
    # reading most common SNPs
    SNPsList = []    
    # needs to be replaced by the path of files
    path = ''
    f1 = open(path + '/mostCommon.txt', 'r')
    for line1 in f1:
        SNPsList.append(line1.strip())  # We don't want newlines in our list, do we?

    
    # SNPsList = list(SNPsDic.keys())
    print(len(SNPsList))

    # needs to be replaced by the path of dataset files
    path = ''

    print("current path of working space= " + path)

        
    # SNPs that are in LD with our most common SNPs
    LDList = []    
    f1 = open(path + '/mostCommonAndLD.txt', 'r')
    for line1 in f1:
        LDList.append(line1.strip())  # We don't want newlines in our list, do we?
    
    print(len(LDList))
    ExtendedSNPsList = list(set(SNPsList + LDList))
  
        
    # This is the smaller dataset (we found the SNPs from this dataset)
    print("Reading FamilyNINDS2(MergedAndBalancedWithID).csv ...")
    df= pd.read_csv(path + "FamilyNINDS2(MergedAndBalancedWithID).csv")
    
    count_row = df.shape[0]  # gives number of row count
    count_col = df.shape[1]  # gives number of col count

    print("The Number Samples in This Dataset: " + str(count_row))
    print("The Number Features in This Dataset: " + str(count_col))

    # Generating a new column named 'fold' with values 1 to 5 that is randomly
    generatedFolds = np.random.permutation(np.repeat([1, 2, 3, 4, 5], math.ceil(count_row / 5)))

    extra_folds = len(generatedFolds) - count_row

    generatedFolds = delete_random_elems(generatedFolds, extra_folds)

    df['folds'] = generatedFolds
    # using counter
    print("Number of samples belong to fold 1: " + str(operator.countOf(df['folds'], 1)))
    print("Number of samples belong to fold 2: " + str(operator.countOf(df['folds'], 2)))
    print("Number of samples belong to fold 3: " + str(operator.countOf(df['folds'], 3)))
    print("Number of samples belong to fold 4: " + str(operator.countOf(df['folds'], 4)))
    print("Number of samples belong to fold 5: " + str(operator.countOf(df['folds'], 5)))

    print("Last 3 column of dataframe:")
    N = 3
    # Select last N columns of dataframe
    last_n_column = df.iloc[:, -N:]
    print(last_n_column)

    fold1 = df.loc[df["folds"] == 1, :]
    fold2 = df.loc[df["folds"] == 2, :]
    fold3 = df.loc[df["folds"] == 3, :]
    fold4 = df.loc[df["folds"] == 4, :]
    fold5 = df.loc[df["folds"] == 5, :]

    # train with fold 1,2,3,4 - test with fold 5
    fold1234 = pd.concat([fold1, fold2, fold3, fold4])

    # train with fold 1,2,3,5 - test with fold 4
    fold1235 = pd.concat([fold1, fold2, fold3, fold5])

    # train with fold 1,2,4,5 - test with fold 3
    fold1245 = pd.concat([fold1, fold2, fold4, fold5])

    # train with fold 1,3,4,5 - test with fold 2
    fold1345 = pd.concat([fold1, fold3, fold4, fold5])

    # train with fold 2,3,4,5 - test with fold 1
    fold2345 = pd.concat([fold2, fold3, fold4, fold5])

    acc_list = []
    acc_listF = []
    acc_listN2 = []

    # fold 5: train the model with fold 1,2,3,4 & test with fold 5
    print("########## fold 5: train the model with fold 1,2,3,4 & test with fold 5 ##########")
    x_train = fold1234.iloc[:, fold1234.columns.isin(ExtendedSNPsList)]
    y_train = fold1234.Label
    x_test = fold5.iloc[:, fold5.columns.isin(ExtendedSNPsList)]
    y_test = fold5.Label

    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    # Store actual value vs predicted value
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'DatasetID': fold5.DatasetID, 'folds': fold5.folds})
    print(df)

    # accuracy for Family
    df1 = df.loc[df["DatasetID"] == 'F', :]
    accF = np.sum(np.equal(df1['Actual'], df1['Predicted'])) / len(df1['Actual'])

    # accuracy for NINDS2
    df2 = df.loc[df["DatasetID"] == 'N2', :]
    accN2 = np.sum(np.equal(df2['Actual'], df2['Predicted'])) / len(df2['Actual'])

    acc = metrics.accuracy_score(y_test, y_pred)
    acc_list.append(acc)
    acc_listF.append(accF)
    acc_listN2.append(accN2)

    print("Accuracy Total: ", str(acc * 100))
    print("Accuracy Family: ", str(accF * 100))
    print("Accuracy NINDS2: ", str(accN2 * 100))
    print('\n')
    print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))
    print(f"Classification report for classifier {clf}:\n"
          f"{metrics.classification_report(y_test, y_pred)}\n")

    #################################################################

    # fold 4: train the model with fold 1,2,3,5 & test with fold 4
    print("########## fold 4: train the model with fold 1,2,3,5 & test with fold 4 ##########")
    x_train = fold1235.iloc[:, fold1235.columns.isin(ExtendedSNPsList)]
    y_train = fold1235.Label
    x_test = fold4.iloc[:, fold4.columns.isin(ExtendedSNPsList)]
    y_test = fold4.Label

    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    # Store actual value vs predicted value
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'DatasetID': fold4.DatasetID, 'folds': fold4.folds})
    print(df)

    # accuracy for Family
    df1 = df.loc[df["DatasetID"] == 'F', :]
    accF = np.sum(np.equal(df1['Actual'], df1['Predicted'])) / len(df1['Actual'])

    # accuracy for NINDS2
    df2 = df.loc[df["DatasetID"] == 'N2', :]
    accN2 = np.sum(np.equal(df2['Actual'], df2['Predicted'])) / len(df2['Actual'])

    acc = metrics.accuracy_score(y_test, y_pred)
    acc_list.append(acc)
    acc_listF.append(accF)
    acc_listN2.append(accN2)

    print("Accuracy Total: ", str(acc * 100))
    print("Accuracy Family: ", str(accF * 100))
    print("Accuracy NINDS2: ", str(accN2 * 100))

    print('\n')
    print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))
    print(f"Classification report for classifier {clf}:\n"
          f"{metrics.classification_report(y_test, y_pred)}\n")

    #################################################################

    # fold 3: train the model with fold 1,2,4,5 & test with fold 3
    print("########## fold 3: train the model with fold 1,2,4,5 & test with fold 3 ##########")
    x_train = fold1245.iloc[:, fold1245.columns.isin(ExtendedSNPsList)]
    y_train = fold1245.Label
    x_test = fold3.iloc[:, fold3.columns.isin(ExtendedSNPsList)]
    y_test = fold3.Label

    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    # Store actual value vs predicted value
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'DatasetID': fold3.DatasetID, 'folds': fold3.folds})
    print(df)

    # accuracy for Family
    df1 = df.loc[df["DatasetID"] == 'F', :]
    accF = np.sum(np.equal(df1['Actual'], df1['Predicted'])) / len(df1['Actual'])

    # accuracy for NINDS2
    df2 = df.loc[df["DatasetID"] == 'N2', :]
    accN2 = np.sum(np.equal(df2['Actual'], df2['Predicted'])) / len(df2['Actual'])

    acc = metrics.accuracy_score(y_test, y_pred)
    acc_list.append(acc)
    acc_listF.append(accF)
    acc_listN2.append(accN2)

    print("Accuracy Total: ", str(acc * 100))
    print("Accuracy Family: ", str(accF * 100))
    print("Accuracy NINDS2: ", str(accN2 * 100))

    print('\n')
    print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))
    print(f"Classification report for classifier {clf}:\n"
          f"{metrics.classification_report(y_test, y_pred)}\n")

    #################################################################

    # fold 2: train the model with fold 1,3,4,5 & test with fold 2
    print("########## fold 2: train the model with fold 1,3,4,5 & test with fold 2 ##########")
    x_train = fold1345.iloc[:, fold1345.columns.isin(ExtendedSNPsList)]
    y_train = fold1345.Label
    x_test = fold2.iloc[:, fold2.columns.isin(ExtendedSNPsList)]
    y_test = fold2.Label

    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    # Store actual value vs predicted value
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'DatasetID': fold2.DatasetID, 'folds': fold2.folds})
    print(df)

    # accuracy for Family
    df1 = df.loc[df["DatasetID"] == 'F', :]
    accF = np.sum(np.equal(df1['Actual'], df1['Predicted'])) / len(df1['Actual'])

    # accuracy for NINDS2
    df2 = df.loc[df["DatasetID"] == 'N2', :]
    accN2 = np.sum(np.equal(df2['Actual'], df2['Predicted'])) / len(df2['Actual'])

    acc = metrics.accuracy_score(y_test, y_pred)
    acc_list.append(acc)
    acc_listF.append(accF)
    acc_listN2.append(accN2)

    print("Accuracy Total: ", str(acc * 100))
    print("Accuracy Family: ", str(accF * 100))
    print("Accuracy NINDS2: ", str(accN2 * 100))

    print('\n')
    print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))
    print(f"Classification report for classifier {clf}:\n"
          f"{metrics.classification_report(y_test, y_pred)}\n")

    #################################################################

    # fold 1: train the model with fold 2,3,4,5 & test with fold 1
    print("########## fold 1: train the model with fold 2,3,4,5 & test with fold 1 ##########")
    x_train = fold2345.iloc[:, fold2345.columns.isin(ExtendedSNPsList)]
    y_train = fold2345.Label
    x_test = fold1.iloc[:, fold1.columns.isin(ExtendedSNPsList)]
    y_test = fold1.Label

    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    # Store actual value vs predicted value
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'DatasetID': fold1.DatasetID, 'folds': fold1.folds})
    print(df)

    # accuracy for Family
    df1 = df.loc[df["DatasetID"] == 'F', :]
    accF = np.sum(np.equal(df1['Actual'], df1['Predicted'])) / len(df1['Actual'])

    # accuracy for NINDS2
    df2 = df.loc[df["DatasetID"] == 'N2', :]
    accN2 = np.sum(np.equal(df2['Actual'], df2['Predicted'])) / len(df2['Actual'])

    acc = metrics.accuracy_score(y_test, y_pred)
    acc_list.append(acc)
    acc_listF.append(accF)
    acc_listN2.append(accN2)

    print("Accuracy Total: ", str(acc * 100))
    print("Accuracy Family: ", str(accF * 100))
    print("Accuracy NINDS2: ", str(accN2 * 100))

    print('\n')
    print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))
    print(f"Classification report for classifier {clf}:\n"
          f"{metrics.classification_report(y_test, y_pred)}\n")

    print("Average accuracy total (among 5 rounds) is : %.2f" % (np.average(acc_list) * 100))
    print("SD of ACC is : %.2f" % (np.std(acc_list) * 100))

    print("Average accuracy Family (among 5 rounds) is : %.2f" % (np.average(acc_listF) * 100))
    print("SD of ACC is : %.2f" % (np.std(acc_listF) * 100))

    print("Average accuracy NINDS2 (among 5 rounds) is : %.2f" % (np.average(acc_listN2) * 100))
    print("SD of ACC is : %.2f" % (np.std(acc_listN2) * 100))
