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

    print("Select the most common SNPs from Familial dataset and check those SNPs quality for getting the accuracy on Autopsy (I got intersection between them):")    
    
    SNPsDic = {138739: 49, 97789: 46, 142629: 46, 176003: 46, 15105: 45, 107485: 44, 155056: 44, 55170: 44, 137683: 44, 133067: 44, 150391: 41, 67275: 41, 6227: 41, 137326: 40, 82572: 40, 160725: 39, 83657: 39, 120979: 38, 109226: 36, 29460: 36, 66237: 34, 153670: 34, 39879: 34, 40494: 34, 20988: 34, 76376: 33, 97045: 32, 160285: 31, 117005: 31, 17297: 31, 156085: 30, 173350: 30, 1159: 30, 156696: 30, 25739: 29, 132485: 29, 58652: 28, 94508: 28, 15003: 28, 101377: 27, 111363: 27, 78908: 26, 175512: 26, 137042: 26, 16961: 26, 119081: 26, 76946: 25, 37222: 25, 78041: 25, 102355: 25, 176391: 25, 139192: 25, 115410: 24, 50351: 24, 131169: 24, 97018: 24, 128315: 23, 79523: 23, 45744: 23, 9634: 23, 54408: 23, 98850: 23, 173427: 22, 140642: 22, 56466: 22, 128585: 22, 96194: 22, 99590: 21, 1413: 21, 97523: 21, 49109: 21, 37094: 21, 16139: 20, 19842: 20, 174806: 20, 24129: 20, 175011: 20, 17579: 20, 119711: 20, 157577: 19, 107570: 19, 155597: 19, 105517: 19, 161760: 19, 164442: 19, 85206: 19, 83158: 19, 111514: 18, 34427: 18, 158187: 18, 79259: 18, 27217: 18, 166245: 18, 37725: 18, 20515: 17, 18166: 17, 57653: 17, 161379: 17, 24270: 16, 60722: 16, 30337: 16, 6277: 16, 22499: 16, 102828: 16, 30405: 16, 145824: 16, 82912: 16, 107573: 16, 144706: 16, 49623: 16, 50523: 15, 101906: 15, 47635: 15, 139923: 15, 174182: 15, 145979: 15, 6193: 15, 167690: 15, 13104: 15, 121212: 14, 88885: 14, 4220: 14, 91274: 14, 39934: 14, 20916: 14, 157743: 14, 86059: 14, 20680: 14, 9217: 14, 121774: 14, 32068: 14, 85761: 13, 123063: 13, 7792: 13, 116060: 13, 126934: 13, 138211: 13, 83700: 13, 74807: 12, 65102: 12, 148433: 12, 116618: 12, 108279: 12, 37817: 12, 90213: 12, 125677: 12, 145109: 12, 143922: 11, 12486: 11, 93077: 11, 161953: 11, 48801: 11, 105691: 11, 55451: 11, 162765: 11, 74756: 11, 15148: 11, 48949: 11, 162756: 11, 41711: 11, 23163: 11, 133152: 10, 77887: 10, 15330: 10, 8478: 10, 103520: 10, 101283: 10, 144611: 10, 106394: 10, 72218: 10, 95372: 10, 34602: 10, 97807: 9, 30100: 9, 106024: 9, 171497: 9, 50660: 9, 22732: 9, 24794: 9, 115386: 9, 29902: 9, 135905: 9, 101704: 9, 149017: 9, 33715: 9, 151931: 8, 60204: 8, 107289: 8, 127251: 8, 5788: 8, 6832: 8, 69542: 8, 85815: 8, 85316: 8, 41907: 8, 117416: 8, 123427: 8, 118252: 8, 73215: 8, 7830: 7, 151479: 7, 150717: 7, 138859: 7, 82921: 7, 57359: 7, 164233: 7, 139614: 7, 110272: 7, 167231: 7, 118838: 7, 94327: 7, 158448: 7, 149816: 7, 65965: 7, 138110: 7, 105664: 7, 148951: 6, 100192: 6, 131035: 6, 138255: 6, 97924: 6, 102085: 6, 140100: 6, 73581: 6, 147840: 6, 134417: 6, 111288: 6, 36557: 6, 15604: 6, 24263: 6, 90006: 6, 113648: 6, 123303: 6, 151559: 6, 60913: 6, 143111: 6, 76666: 6, 9482: 5, 119542: 5, 56350: 5, 123723: 5, 114388: 5, 24154: 5, 64326: 5, 105583: 5, 76580: 5, 119437: 5, 19594: 5, 43056: 5, 59602: 5, 168918: 5, 18492: 5, 97190: 5, 24475: 5, 94587: 5, 118435: 5, 151022: 5, 124797: 5, 80909: 5, 70919: 5, 92489: 5, 135466: 5, 159520: 5, 80686: 5, 114148: 5, 169474: 5, 139330: 5, 150334: 5, 106820: 5, 56209: 5, 1858: 5, 142204: 5, 154537: 5, 13712: 5, 28944: 5, 100368: 5, 47331: 5, 64576: 5}

    
    SNPsList = list(SNPsDic.keys())
    print(len(SNPsList))
    # needs to be replaced by the path of files
    path = ''

    print("current path of working space= " + path)
  
        
    # This is the smaller dataset (we found the SNPs from this dataset)
    print("Reading 2Autopsy.csv ...")
    df= pd.read_csv(path + "2Autopsy.csv")
    
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
