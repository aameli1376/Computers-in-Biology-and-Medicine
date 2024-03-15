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

    print("Select the most common SNPs from merged version of (Familial and NINDS2) dataset and check those SNPs quality for getting the accuracy:")    
    
    SNPsDic = {2077: 3, 2487: 3, 1859: 3, 295: 3, 3138: 3, 3619: 3, 4020: 2, 247: 2, 2248: 2, 2742: 2, 2844: 2, 3126: 2, 3694: 2, 1927: 2, 248: 2, 659: 2, 805: 2, 3499: 2, 2866: 2, 202: 1, 181: 1, 54: 1, 108: 1, 21: 1, 151: 1, 2105: 1, 233: 1, 841: 1, 184: 1, 1496: 1, 606: 1, 119: 1, 1457: 1, 3244: 1, 1034: 1, 1887: 1, 2823: 1, 3234: 1, 3745: 1, 2394: 1, 1767: 1, 1132: 1, 3740: 1, 1370: 1, 179: 1, 3003: 1, 22: 1, 136: 1, 154: 1, 658: 1, 1830: 1, 175: 1, 1690: 1, 177: 1, 3737: 1, 1678: 1, 2147: 1, 3564: 1, 1059: 1, 3624: 1, 1593: 1, 2416: 1, 2497: 1, 2556: 1, 3137: 1, 874: 1, 3445: 1, 153: 1, 1215: 1, 3478: 1, 3955: 1, 1831: 1, 1477: 1, 3841: 1, 3913: 1, 1188: 1, 3978: 1, 2183: 1, 2157: 1, 828: 1, 39: 1, 2222: 1, 193: 1, 529: 1, 2196: 1, 3741: 1, 1974: 1, 3640: 1, 1726: 1, 3388: 1, 1668: 1, 788: 1, 2728: 1, 3870: 1, 1735: 1, 3630: 1, 3237: 1, 1773: 1, 3209: 1, 710: 1, 2760: 1, 3292: 1, 1329: 1, 1747: 1, 385: 1, 1083: 1, 1283: 1, 3852: 1, 534: 1, 1611: 1, 833: 1, 3312: 1, 2211: 1, 370: 1, 3249: 1, 3361: 1, 262: 1, 126: 1, 3230: 1, 70: 1, 1441: 1, 2663: 1, 1878: 1, 2992: 1, 3018: 1}

    
    SNPsList = list(SNPsDic.keys())
    print(SNPsList)
    # needs to be replaced by the path of files
    path = ''

    print("current path of working space= " + path)
  
        
    # This is the smaller dataset (we found the SNPs from this dataset)
    print("Reading FamilyNINDS2(MergedAndBalanced).csv ...")
    df= pd.read_csv(path + "FamilyNINDS2(MergedAndBalanced).csv")
    
    count_row = df.shape[0]  # gives number of row count
    count_col = df.shape[1]  # gives number of col count

    print("The Number Samples in This Dataset: " + str(count_row))
    print("The Number Features in This Dataset: " + str(count_col))

    # X contains selected features except the labels
    x = df.iloc[:, SNPsList]
    # x = df.iloc[:, df.columns.isin(SNPsList)]
    
    print(x.shape)
    
    for col in x.columns:
      print(col)     
