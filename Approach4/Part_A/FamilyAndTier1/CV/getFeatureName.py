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

    print("Select the most common SNPs from merged version of (Familial and Tier1) dataset and check those SNPs quality for getting the accuracy:")    
    
    SNPsDic = {9146: 13, 22105: 10, 27499: 10, 7754: 10, 18082: 10, 20679: 9, 13672: 9, 8417: 8, 16621: 8, 11704: 7, 2496: 7, 12603: 7, 5705: 6, 28378: 5, 3798: 5, 27: 5, 23895: 5, 28131: 5, 19382: 5, 22279: 5, 15671: 5, 20853: 5, 25110: 5, 28484: 5, 1280: 5, 9788: 5, 5257: 5, 21930: 5, 23934: 4, 11442: 4, 5738: 4, 19876: 4, 15678: 4, 27872: 4, 18321: 4, 3740: 4, 16277: 4, 9033: 4, 27096: 4, 15866: 4, 22250: 4, 23937: 4, 28214: 4, 21750: 3, 6540: 3, 14956: 3, 11124: 3, 17578: 3, 26704: 3, 14775: 3, 28502: 3, 21024: 3, 6136: 3, 19455: 3, 4234: 3, 28095: 3, 23229: 3, 27943: 3, 19451: 3, 7695: 3, 17729: 3, 26279: 3, 7623: 3, 138: 3, 17835: 3, 4618: 3, 7472: 3, 12932: 3, 24344: 3, 22974: 3, 1648: 3, 17229: 3, 28574: 3, 23024: 3, 7387: 3, 276: 3, 14395: 3, 18755: 3, 24617: 3, 13008: 3, 9558: 3, 17366: 3, 26061: 3, 22691: 3, 13866: 2, 744: 2, 3624: 2, 28439: 2, 11154: 2, 15574: 2, 24647: 2, 24063: 2, 5830: 2, 4849: 2, 6770: 2, 5203: 2, 5605: 2, 11905: 2, 17267: 2, 12799: 2, 1524: 2, 6211: 2, 15259: 2, 21607: 2, 11430: 2, 12809: 2, 26600: 2, 28473: 2, 2581: 2, 2828: 2, 25198: 2, 36: 2, 27743: 2, 24771: 2, 26952: 2, 2532: 2, 6053: 2, 7718: 2, 15525: 2, 12699: 2, 17297: 2, 11700: 2, 11617: 2, 7206: 2, 9498: 2, 17013: 2, 15646: 2, 9123: 2, 22011: 2, 11526: 2, 11914: 2, 18259: 2, 9983: 2, 28377: 2, 7827: 2, 6354: 2, 21079: 2, 25181: 2, 14825: 2, 26100: 2, 9845: 2, 13977: 2, 6411: 2, 5312: 2, 254: 2, 28610: 2, 2442: 2, 20201: 2, 22855: 2, 22451: 2, 19435: 2, 17702: 2, 28306: 2, 27568: 2, 8630: 2, 9300: 2, 1812: 2, 15426: 2, 17183: 2, 9550: 2, 27436: 2, 25115: 2, 18931: 2, 20772: 2, 6347: 2, 23589: 2, 26409: 2, 21902: 2}

    
    SNPsList = list(SNPsDic.keys())
    print(SNPsList)
    # needs to be replaced by the path of files
    path = ''

    print("current path of working space= " + path)
  
        
    # This is the smaller dataset (we found the SNPs from this dataset)
    print("Reading FamilyTier1(MergedAndBalanced).csv ...")
    df= pd.read_csv(path + "FamilyTier1(MergedAndBalanced).csv")
    
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
