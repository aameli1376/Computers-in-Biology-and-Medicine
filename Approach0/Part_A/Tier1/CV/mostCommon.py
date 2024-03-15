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

    print("Select the most common SNPs from Tier1 dataset and check those SNPs quality for getting the accuracy on Tier1:")    
    
    SNPsDic = {42746: 2, 42918: 2, 170144: 2, 155990: 2, 40660: 2, 137424: 2, 25690: 2, 62167: 2, 171236: 2, 89395: 2, 169233: 2, 132494: 2, 4340: 2, 71273: 1, 126718: 1, 137808: 1, 159262: 1, 11262: 1, 133025: 1, 49916: 1, 171237: 1, 162793: 1, 9771: 1, 6029: 1, 16465: 1, 526: 1, 41104: 1, 117908: 1, 8551: 1, 110449: 1, 173354: 1, 121430: 1, 63117: 1, 15950: 1, 70161: 1, 55626: 1, 27652: 1, 44561: 1, 175873: 1, 55168: 1, 79135: 1, 5230: 1, 140225: 1, 28814: 1, 173928: 1, 134697: 1, 55863: 1, 67263: 1, 166549: 1, 66983: 1, 163644: 1, 76347: 1, 96203: 1, 100896: 1, 192: 1, 143763: 1, 8208: 1, 52336: 1, 11473: 1, 90706: 1, 164421: 1, 84299: 1, 165577: 1, 36197: 1, 48833: 1, 65192: 1, 66994: 1, 28332: 1, 50644: 1, 170834: 1, 152185: 1, 138327: 1, 71270: 1, 72772: 1, 139659: 1, 6235: 1, 54806: 1, 119615: 1, 105477: 1, 80732: 1, 28733: 1, 126911: 1, 50661: 1, 109264: 1, 37363: 1, 114891: 1, 138928: 1, 170455: 1, 125963: 1, 132339: 1, 57828: 1, 135519: 1, 106814: 1, 91121: 1, 118570: 1, 16834: 1, 169778: 1, 102241: 1, 95935: 1, 172623: 1, 163305: 1, 46900: 1, 134150: 1, 82229: 1, 163853: 1, 48215: 1, 172887: 1, 145951: 1, 175776: 1, 107600: 1, 82143: 1, 84368: 1, 143673: 1, 107499: 1, 157951: 1, 158348: 1, 74025: 1, 32617: 1, 72590: 1, 920: 1, 16921: 1, 45454: 1, 56018: 1, 5799: 1, 26107: 1, 51395: 1, 101981: 1, 170751: 1, 70766: 1, 138320: 1, 31944: 1, 106320: 1, 93049: 1, 131209: 1, 68832: 1, 163748: 1, 151917: 1, 113254: 1, 93449: 1, 72519: 1, 32599: 1, 16831: 1, 139865: 1, 52629: 1, 162243: 1, 99749: 1, 171282: 1, 105628: 1, 32691: 1, 34663: 1, 175461: 1, 148304: 1, 165245: 1, 24487: 1, 76591: 1, 13865: 1, 137312: 1, 96527: 1, 146705: 1, 95284: 1, 59101: 1, 42222: 1, 170933: 1, 22252: 1, 8705: 1, 126431: 1, 90219: 1, 59342: 1, 143448: 1, 131323: 1, 101412: 1, 161482: 1, 136032: 1, 85021: 1, 162787: 1, 145234: 1, 11411: 1, 157336: 1, 9089: 1, 27334: 1, 151407: 1, 175769: 1, 122377: 1, 171205: 1, 42890: 1, 167559: 1, 143764: 1, 132206: 1, 170943: 1, 154796: 1, 145912: 1, 105621: 1, 13138: 1, 162056: 1, 134493: 1, 106512: 1, 172648: 1, 171692: 1, 66125: 1, 11814: 1, 175447: 1, 16502: 1, 119339: 1, 132259: 1, 143647: 1, 118110: 1, 40739: 1, 165008: 1, 3970: 1, 87762: 1, 169400: 1, 146185: 1, 8121: 1, 150601: 1, 45194: 1, 167684: 1, 124471: 1, 131960: 1}

    
    SNPsList = list(SNPsDic.keys())
    print(len(SNPsList))
    # needs to be replaced by the path of files
    path = ''

    print("current path of working space= " + path)
  
        
    # This is the smaller dataset (we found the SNPs from this dataset)
    print("Reading PD_1_Tier1_15Jan2022.csv ...")
    df= pd.read_csv(path + "PD_1_Tier1_15Jan2022.csv")
    
    count_row = df.shape[0]  # gives number of row count
    count_col = df.shape[1]  # gives number of col count

    print("The Number Samples in This Dataset: " + str(count_row))
    print("The Number Features in This Dataset: " + str(count_col))

    # X contains selected features except the labels
    x = df.iloc[:, SNPsList]
    # x = df.iloc[:, df.columns.isin(SNPsList)]

    # y contains The labels
    y = df.iloc[:, -1]
    
    print(x.shape)
    
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
