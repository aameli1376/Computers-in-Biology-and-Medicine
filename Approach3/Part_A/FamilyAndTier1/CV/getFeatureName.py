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
    
    SNPsDic = {9146: 16, 20679: 11, 13813: 9, 27499: 8, 2496: 7, 1932: 7, 18904: 6, 5738: 6, 2418: 6, 12603: 6, 15259: 6, 10561: 6, 23934: 5, 5705: 5, 16621: 5, 16277: 5, 15678: 5, 8417: 5, 22691: 5, 20305: 5, 21930: 5, 9123: 5, 21024: 5, 17695: 5, 24240: 4, 27624: 4, 19455: 4, 12799: 4, 13866: 4, 22451: 4, 4211: 4, 11151: 4, 28306: 4, 744: 4, 18808: 4, 20853: 4, 6136: 4, 7602: 4, 7754: 4, 7718: 4, 13388: 4, 22974: 4, 9074: 4, 9558: 3, 21196: 3, 27689: 3, 12486: 3, 21220: 3, 13008: 3, 14775: 3, 2516: 3, 27436: 3, 652: 3, 12252: 3, 2229: 3, 5297: 3, 6540: 3, 18144: 3, 12751: 3, 23346: 3, 8430: 3, 18755: 3, 25110: 3, 26279: 3, 9478: 3, 14182: 3, 5653: 3, 7455: 3, 25829: 3, 15525: 3, 14540: 3, 24617: 3, 12216: 3, 24576: 3, 12771: 3, 5312: 3, 1280: 3, 22133: 3, 1812: 3, 11247: 3, 25390: 3, 12782: 3, 14491: 3, 28551: 3, 17229: 3, 7387: 3, 27014: 3, 25181: 3, 27: 2, 26100: 2, 2199: 2, 20918: 2, 16020: 2, 10559: 2, 25198: 2, 9628: 2, 22869: 2, 276: 2, 12932: 2, 10006: 2, 16976: 2, 8352: 2, 26741: 2, 14130: 2, 12962: 2, 24270: 2, 4489: 2, 11199: 2, 7572: 2, 7472: 2, 7131: 2, 15914: 2, 19382: 2, 17637: 2, 22279: 2, 15338: 2, 8046: 2, 17207: 2, 6459: 2, 23024: 2, 27465: 2, 25085: 2, 18925: 2, 16498: 2, 9033: 2, 3595: 2, 11732: 2, 21038: 2, 5148: 2, 17013: 2, 230: 2, 15738: 2, 10308: 2, 4234: 2, 3409: 2, 17011: 2, 16784: 2, 22453: 2, 27025: 2, 11442: 2, 12849: 2, 20336: 2, 17999: 2, 22430: 2, 1524: 2, 12699: 2, 2929: 2, 13103: 2, 18470: 2, 27642: 2, 17267: 2, 2409: 2, 22752: 2, 26: 2, 1007: 2, 27936: 2, 5604: 2, 17646: 2, 3429: 2, 26576: 2, 11905: 2, 3798: 2, 1914: 2, 20186: 2, 15866: 2, 28490: 2, 5257: 2, 44: 2, 22842: 2, 6496: 2, 11526: 2, 1022: 2, 9939: 2, 1263: 2, 6286: 2, 22094: 2, 20317: 2, 28588: 2, 12012: 2, 13977: 2, 27680: 2, 26061: 2, 13075: 2, 9109: 2, 14956: 2, 14825: 2, 24324: 2, 28508: 2, 17831: 2, 1611: 2, 13377: 2, 6347: 2, 15631: 2, 11094: 2, 16804: 2, 27907: 2}

    
    SNPsList = list(SNPsDic.keys())
    print(SNPsList)
    # needs to be replaced by the path of files
    path = ''

    print("current path of working space= " + path)
  
        
    # This is the smaller dataset (we found the SNPs from this dataset)
    print("Reading FamilyTier1(Merged).csv ...")
    df= pd.read_csv(path + "FamilyTier1(Merged).csv")
    
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
