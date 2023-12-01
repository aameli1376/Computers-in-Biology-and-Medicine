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
    
    SNPsDic = {2077: 4, 150: 3, 3363: 3, 2842: 3, 2487: 3, 1240: 2, 148: 2, 119: 2, 2844: 2, 295: 2, 0: 2, 3294: 2, 1746: 2, 465: 2, 3593: 2, 1593: 2, 1377: 2, 3306: 2, 1246: 2, 4020: 2, 3525: 2, 338: 2, 659: 2, 3240: 2, 1752: 2, 3841: 2, 152: 1, 159: 1, 937: 1, 1955: 1, 1623: 1, 202: 1, 534: 1, 1784: 1, 1850: 1, 868: 1, 2735: 1, 3175: 1, 2428: 1, 1024: 1, 2464: 1, 2230: 1, 1748: 1, 3057: 1, 650: 1, 700: 1, 2282: 1, 270: 1, 3654: 1, 2195: 1, 263: 1, 320: 1, 1251: 1, 3237: 1, 1059: 1, 2340: 1, 1514: 1, 586: 1, 3624: 1, 3426: 1, 3910: 1, 2121: 1, 351: 1, 420: 1, 649: 1, 3743: 1, 2837: 1, 516: 1, 1027: 1, 2386: 1, 2811: 1, 2840: 1, 1014: 1, 3022: 1, 3158: 1, 262: 1, 3621: 1, 386: 1, 2433: 1, 1365: 1, 3404: 1, 1887: 1, 2187: 1, 2424: 1, 1511: 1, 3: 1, 2683: 1, 1565: 1, 193: 1, 1534: 1, 3955: 1, 541: 1, 14: 1, 2768: 1, 2623: 1, 126: 1, 592: 1, 1134: 1, 3604: 1, 391: 1, 2616: 1, 1794: 1, 3651: 1, 3526: 1, 3513: 1, 1712: 1, 797: 1, 3440: 1, 2972: 1, 1847: 1, 3090: 1, 1160: 1, 2359: 1, 104: 1, 1312: 1, 1049: 1, 3564: 1, 1132: 1, 3998: 1, 2217: 1, 2470: 1, 3608: 1, 2867: 1, 3862: 1, 3978: 1, 3338: 1, 3871: 1, 1717: 1, 775: 1, 1764: 1, 156: 1, 4: 1, 2592: 1, 1668: 1, 368: 1, 3114: 1, 2108: 1, 1915: 1, 1039: 1, 2045: 1, 2066: 1, 1537: 1, 2506: 1, 2832: 1, 39: 1, 3001: 1, 2088: 1, 2775: 1, 2093: 1, 1780: 1, 3380: 1, 1083: 1, 1356: 1, 968: 1, 2296: 1, 1032: 1, 1111: 1, 3573: 1, 3416: 1, 1690: 1, 795: 1, 2179: 1, 2589: 1, 3886: 1}

    
    SNPsList = list(SNPsDic.keys())
    print(SNPsList)

    path = '/scratch/fs2/usefi/pd/datasets/All_Aproaches/Approach3/Datasets/ID/'

    print("current path of working space= " + path)
  
        
    # This is the smaller dataset (we found the SNPs from this dataset)
    print("Reading FamilyNINDS2(Merged).csv ...")
    df= pd.read_csv(path + "FamilyNINDS2(Merged).csv")
    
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