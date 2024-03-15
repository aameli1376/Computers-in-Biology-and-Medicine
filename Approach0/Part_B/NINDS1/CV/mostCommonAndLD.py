import os
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import RepeatedStratifiedKFold
import numpy as np

if __name__ == "__main__":

    print("Select the most common SNPs from NINDS1 dataset and check those SNPs quality for getting the accuracy on NINDS1 after adding LD:")    
    
    # reading most common SNPs
    SNPsList = []  
    # needs to be replaced by the path of files
    path = ''
    f1 = open(path + '/mostCommonSNPs.txt', 'r')
    for line1 in f1:
        SNPsList.append(line1.strip())  # We don't want newlines in our list, do we?

    
    # SNPsList = list(SNPsDic.keys())
    print(len(SNPsList))

    # needs to be replaced by the path of files
    path = ''

    print("current path of working space= " + path)

        
    # SNPs that are in LD with our most common SNPs
    LDList = []
    path = ''
    f1 = open(path + '/SNPsInLDWithMostCommonSNPs.txt', 'r')
    for line1 in f1:
        LDList.append(line1.strip())  # We don't want newlines in our list, do we?
    
    print(len(LDList))
    ExtendedSNPsList = list(set(SNPsList + LDList))
  
        
    # This is the smaller dataset (we found the SNPs from this dataset)
    print("Reading PD_1_NINDS_28Jan2022.csv ...")
    df= pd.read_csv(path + "PD_1_NINDS_28Jan2022.csv")
    
    count_row = df.shape[0]  # gives number of row count
    count_col = df.shape[1]  # gives number of col count

    print("The Number Samples in This Dataset: " + str(count_row))
    print("The Number Features in This Dataset: " + str(count_col))

    # X contains selected features except the labels
    x = df.iloc[:, df.columns.isin(ExtendedSNPsList)]
    # x = df.iloc[:, df.columns.isin(SNPsList)]
    print(x.shape)
    # y contains The labels
    y = df.iloc[:, -1]
    
    counter = 0
    folds = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
    acc_list = []
            
    for train_idx, test_dix in folds.split(x, y):
        train_x, test_x = x.iloc[train_idx, :], x.iloc[test_dix, :]
        train_y, test_y = y.iloc[train_idx], y.iloc[test_dix]
        
        counter = counter + 1
        clf = RandomForestClassifier()
        clf.fit(train_x, train_y)
        y_pred = clf.predict(test_x)
        acc = metrics.accuracy_score(test_y, y_pred)
        acc_list.append(acc)
        
        print("Round " + str(counter) + ": ")
        print(acc * 100)
        print('\n')
        
        print(metrics.confusion_matrix(y_true=test_y, y_pred=y_pred))
        print(f"Classification report for classifier {clf}:\n"
          f"{metrics.classification_report(test_y, y_pred)}\n")

    print("Average accuracy (among 5 rounds) is : %.2f" % (np.average(acc_list) * 100))
    print("SD of ACC is : %.2f" % (np.std(acc_list) * 100))
