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
    print(
        "Select the most common SNPs from merged version of (Familial and NINDS2) dataset and check those SNPs quality for getting the accuracy:")

    SNPsDic = {2077: 4, 150: 3, 3363: 3, 2842: 3, 2487: 3, 1240: 2, 148: 2, 119: 2, 2844: 2, 295: 2, 0: 2, 3294: 2,
               1746: 2, 465: 2, 3593: 2, 1593: 2, 1377: 2, 3306: 2, 1246: 2, 4020: 2, 3525: 2, 338: 2, 659: 2, 3240: 2,
               1752: 2, 3841: 2, 152: 1, 159: 1, 937: 1, 1955: 1, 1623: 1, 202: 1, 534: 1, 1784: 1, 1850: 1, 868: 1,
               2735: 1, 3175: 1, 2428: 1, 1024: 1, 2464: 1, 2230: 1, 1748: 1, 3057: 1, 650: 1, 700: 1, 2282: 1, 270: 1,
               3654: 1, 2195: 1, 263: 1, 320: 1, 1251: 1, 3237: 1, 1059: 1, 2340: 1, 1514: 1, 586: 1, 3624: 1, 3426: 1,
               3910: 1, 2121: 1, 351: 1, 420: 1, 649: 1, 3743: 1, 2837: 1, 516: 1, 1027: 1, 2386: 1, 2811: 1, 2840: 1,
               1014: 1, 3022: 1, 3158: 1, 262: 1, 3621: 1, 386: 1, 2433: 1, 1365: 1, 3404: 1, 1887: 1, 2187: 1, 2424: 1,
               1511: 1, 3: 1, 2683: 1, 1565: 1, 193: 1, 1534: 1, 3955: 1, 541: 1, 14: 1, 2768: 1, 2623: 1, 126: 1,
               592: 1, 1134: 1, 3604: 1, 391: 1, 2616: 1, 1794: 1, 3651: 1, 3526: 1, 3513: 1, 1712: 1, 797: 1, 3440: 1,
               2972: 1, 1847: 1, 3090: 1, 1160: 1, 2359: 1, 104: 1, 1312: 1, 1049: 1, 3564: 1, 1132: 1, 3998: 1,
               2217: 1, 2470: 1, 3608: 1, 2867: 1, 3862: 1, 3978: 1, 3338: 1, 3871: 1, 1717: 1, 775: 1, 1764: 1, 156: 1,
               4: 1, 2592: 1, 1668: 1, 368: 1, 3114: 1, 2108: 1, 1915: 1, 1039: 1, 2045: 1, 2066: 1, 1537: 1, 2506: 1,
               2832: 1, 39: 1, 3001: 1, 2088: 1, 2775: 1, 2093: 1, 1780: 1, 3380: 1, 1083: 1, 1356: 1, 968: 1, 2296: 1,
               1032: 1, 1111: 1, 3573: 1, 3416: 1, 1690: 1, 795: 1, 2179: 1, 2589: 1, 3886: 1}

    SNPsList = list(SNPsDic.keys())
    print(len(SNPsList))
    
    path = "/scratch/fs2/usefi/pd/datasets/All_Aproaches/Approach3/Datasets/ID/" 
    # This is the smaller dataset (we found the SNPs from this dataset)
    print("Reading FamilyNINDS2(MergedWithID).csv ...")
    df = pd.read_csv(path + "FamilyNINDS2(MergedWithID).csv")

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
    x_train = fold1234.iloc[:, SNPsList]
    y_train = fold1234.Label
    x_test = fold5.iloc[:, SNPsList]
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
    x_train = fold1235.iloc[:, SNPsList]
    y_train = fold1235.Label
    x_test = fold4.iloc[:, SNPsList]
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
    x_train = fold1245.iloc[:, SNPsList]
    y_train = fold1245.Label
    x_test = fold3.iloc[:, SNPsList]
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
    x_train = fold1345.iloc[:, SNPsList]
    y_train = fold1345.Label
    x_test = fold2.iloc[:, SNPsList]
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
    x_train = fold2345.iloc[:, SNPsList]
    y_train = fold2345.Label
    x_test = fold1.iloc[:, SNPsList]
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
