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

    print("Select the most common SNPs from Familial dataset and check those SNPs quality for getting the accuracy on NINDS1 (I got intersection between them):")    
    
    SNPsDic = {192820: 50, 262707: 50, 202428: 49, 220643: 48, 142843: 48, 295949: 47, 214360: 47, 183521: 46, 242933: 46, 202861: 45, 255628: 44, 126102: 43, 154380: 43, 186329: 41, 71689: 41, 41204: 41, 62433: 40, 227220: 40, 2579: 39, 180298: 38, 609: 38, 251034: 35, 180805: 34, 89265: 33, 167820: 32, 43797: 32, 126986: 30, 270197: 29, 128216: 29, 267716: 29, 115746: 28, 131585: 28, 147025: 28, 278349: 28, 46903: 28, 133956: 27, 209508: 27, 127057: 27, 91818: 26, 135173: 25, 31717: 25, 83351: 25, 213601: 25, 9696: 24, 165199: 23, 224040: 23, 268095: 23, 117930: 22, 253368: 22, 8395: 22, 294780: 22, 263495: 22, 84442: 22, 125418: 21, 125691: 21, 200495: 21, 209376: 21, 145287: 20, 27660: 20, 286392: 20, 189827: 20, 38884: 20, 50442: 20, 186884: 20, 16141: 20, 233964: 20, 25252: 20, 257878: 20, 261735: 20, 87124: 19, 244474: 19, 227408: 19, 24574: 19, 135473: 19, 135700: 19, 51103: 18, 82125: 18, 30343: 18, 108515: 18, 146750: 18, 223658: 17, 245149: 17, 119679: 17, 108208: 17, 297254: 17, 174283: 17, 12475: 17, 180754: 17, 1462: 17, 254680: 17, 55989: 17, 278579: 17, 296697: 17, 69594: 16, 170794: 16, 165152: 16, 275966: 16, 246364: 16, 85607: 15, 216409: 15, 133673: 15, 183176: 15, 164807: 15, 131455: 15, 38112: 14, 159298: 14, 284904: 14, 44889: 14, 203818: 14, 234433: 14, 131841: 14, 180493: 14, 20279: 14, 271457: 13, 199359: 13, 59747: 13, 179866: 13, 95974: 13, 166671: 13, 274847: 13, 275875: 13, 95329: 13, 106988: 12, 272003: 12, 239468: 12, 141997: 12, 37798: 12, 66783: 12, 281855: 12, 183704: 12, 274641: 12, 189461: 12, 34987: 12, 185109: 12, 39749: 11, 207492: 11, 146253: 11, 187994: 11, 183233: 11, 13291: 11, 236694: 11, 94116: 11, 265200: 11, 104907: 11, 234980: 11, 220108: 11, 6783: 10, 236883: 10, 293837: 10, 63662: 10, 198601: 10, 163108: 10, 100783: 10, 32980: 10, 251504: 10, 246475: 10, 19624: 10, 189615: 10, 296555: 10, 270910: 10, 215120: 10, 30070: 10, 81710: 10, 17127: 10, 144357: 10, 292534: 10, 60522: 10, 205661: 10, 251796: 9, 97617: 9, 55715: 9, 200713: 9, 27112: 9, 123707: 9, 189507: 9, 260481: 9, 188976: 9, 131365: 9, 22297: 9, 255100: 9, 37916: 9, 102033: 9, 93088: 9, 226571: 9, 29849: 9, 75007: 9, 178982: 9, 6108: 9, 49348: 9, 246115: 9, 157251: 9, 192606: 9, 245674: 9, 89848: 8, 24299: 8, 273399: 8, 113806: 8, 180511: 8, 174610: 8, 14217: 8, 63802: 8, 84151: 8, 209451: 8, 132020: 8, 34034: 8, 231602: 8, 143972: 8, 191522: 7, 15397: 7, 174389: 7, 195952: 7, 202811: 7, 255268: 7, 177034: 7, 37875: 7, 122807: 7, 202938: 7, 281236: 7, 188396: 7, 280973: 7, 166129: 7, 227416: 7, 25503: 7, 32995: 7, 2641: 7, 174126: 7, 255909: 7, 72340: 7, 109974: 6, 45709: 6, 49778: 6, 4487: 6, 22095: 6, 87114: 6, 124248: 6, 273864: 6, 19740: 6, 135299: 6, 18011: 6, 189462: 6, 191837: 6, 194063: 6, 297293: 6, 53456: 6, 173571: 6, 54985: 6, 269971: 6, 61379: 6, 169295: 6, 122851: 6, 131014: 6, 117813: 6, 199512: 6, 17180: 6, 275748: 6, 22340: 6, 113957: 6, 153526: 6, 279552: 6, 274000: 6, 134400: 5, 260597: 5, 43826: 5, 234037: 5, 211587: 5, 272473: 5, 10748: 5, 154858: 5, 67919: 5, 289232: 5, 7448: 5, 13917: 5, 97864: 5, 126462: 5, 273317: 5, 72161: 5, 235830: 5, 79605: 5, 264197: 5, 103235: 5, 99097: 5, 65075: 5, 140864: 5, 257999: 5, 181142: 5, 280438: 5, 189492: 5, 131513: 5, 39878: 5, 135594: 5}

    
    SNPsList = list(SNPsDic.keys())
    print(len(SNPsList))

    path = '/scratch/fs2/usefi/pd/datasets/All_Aproaches/Approach2/Datasets/IntersectionFamilyAndNINDS1/'

    print("current path of working space= " + path)
  
        
    # This is the smaller dataset (we found the SNPs from this dataset)
    print("Reading NINDS1_12May2022.csv ...")
    df= pd.read_csv(path + "NINDS1_12May2022.csv")
    
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