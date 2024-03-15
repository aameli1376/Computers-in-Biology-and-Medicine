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

    print("Select the most common SNPs from Familial dataset and check those SNPs quality for getting the accuracy on Tier1 (I got intersection between them):")    
    
    SNPsDic = {3044: 38, 16361: 35, 7914: 34, 12699: 33, 8885: 33, 18459: 33, 6293: 33, 19455: 32, 12898: 32, 11094: 31, 16193: 30, 5738: 29, 12106: 28, 16086: 27, 13008: 26, 15926: 26, 24910: 26, 13018: 25, 15054: 25, 17999: 25, 20305: 25, 21459: 24, 9045: 23, 1914: 23, 9179: 22, 17366: 22, 22974: 22, 19316: 22, 27465: 22, 1021: 22, 17952: 21, 5892: 21, 24240: 21, 8046: 21, 12315: 21, 23895: 21, 15259: 21, 7290: 21, 10512: 21, 21024: 21, 6272: 20, 25592: 20, 12771: 20, 14015: 20, 19390: 20, 9146: 19, 18925: 19, 3798: 19, 10730: 19, 21079: 19, 21208: 19, 2349: 19, 4489: 18, 22855: 18, 21687: 18, 21433: 18, 8844: 18, 22842: 18, 15043: 17, 7602: 17, 8492: 17, 21220: 17, 10644: 17, 12751: 17, 12687: 17, 7970: 17, 27499: 17, 16621: 17, 12216: 17, 2496: 17, 26200: 16, 7387: 16, 1007: 16, 21795: 16, 16918: 16, 4271: 16, 25155: 16, 23937: 16, 20480: 16, 27436: 16, 2532: 16, 5373: 16, 16644: 15, 14328: 15, 16471: 15, 17835: 15, 1524: 15, 4142: 15, 22105: 15, 2554: 15, 1594: 15, 24063: 14, 17706: 14, 15679: 14, 22752: 14, 11131: 14, 1806: 14, 20696: 14, 26840: 14, 4520: 14, 22094: 14, 5084: 14, 15129: 14, 11557: 14, 6136: 13, 22391: 13, 13605: 13, 13813: 13, 10561: 13, 19451: 13, 23834: 13, 8212: 13, 22731: 13, 27987: 13, 12782: 13, 14938: 13, 19447: 13, 24344: 13, 20838: 13, 12603: 13, 146: 13, 19271: 12, 8636: 12, 9478: 12, 22133: 12, 1870: 12, 22250: 12, 3616: 12, 25431: 12, 6211: 12, 4326: 12, 14103: 12, 7455: 12, 17207: 12, 20336: 12, 8630: 12, 12122: 12, 5472: 12, 9983: 12, 4630: 12, 21682: 12, 14491: 12, 27127: 12, 25291: 12, 17740: 11, 26741: 11, 10225: 11, 7953: 11, 21293: 11, 21341: 11, 14945: 11, 23734: 11, 3849: 11, 276: 11, 16060: 11, 1932: 11, 2262: 11, 11942: 11, 8124: 10, 16616: 10, 5257: 10, 17267: 10, 17561: 10, 18636: 10, 22761: 10, 20042: 10, 16527: 10, 11247: 10, 4234: 10, 3439: 10, 22430: 10, 22691: 10, 21038: 10, 1317: 10, 20732: 10, 27456: 10, 5676: 9, 20370: 9, 18904: 9, 20156: 9, 25110: 9, 13619: 9, 4526: 9, 9787: 9, 18658: 9, 16804: 9, 27522: 9, 12799: 9, 11442: 9, 9788: 9, 12202: 9, 25598: 9, 15407: 9, 25898: 9, 17845: 9, 16277: 9, 23556: 9, 6782: 9, 18755: 8, 15866: 8, 6786: 8, 15525: 8, 18055: 8, 17241: 8, 12650: 8, 11732: 8, 1459: 8, 6540: 8, 5597: 8, 9007: 8, 11570: 8, 21930: 8, 19782: 8, 6225: 8, 26054: 8, 1625: 8, 3930: 8, 12637: 8, 6623: 8, 15956: 8, 11579: 8, 12252: 8, 5203: 8, 11905: 8, 25301: 8, 16477: 8, 9620: 8, 11914: 8, 7754: 8, 23589: 7, 4282: 7, 1037: 7, 12759: 7, 26143: 7, 27317: 7, 13103: 7, 14182: 7, 9558: 7, 1494: 7, 25198: 7, 14775: 7, 138: 7, 1186: 7, 12230: 7, 22283: 7, 24802: 7, 2653: 7, 14527: 7, 23309: 7, 14832: 7, 24263: 7, 2418: 7, 17695: 7, 13753: 6, 3199: 6, 22640: 6, 24602: 6, 8207: 6, 23934: 6, 18470: 6, 22906: 6, 25349: 6, 4316: 6, 11700: 6, 3405: 6, 18464: 6, 22567: 6, 366: 6, 12793: 6, 6986: 6, 5004: 6, 942: 6, 21196: 6, 16036: 6, 1956: 6, 13302: 6, 25346: 6, 4488: 6, 24339: 6, 23967: 6, 10006: 6, 9498: 5, 4897: 5, 7786: 5, 20679: 5, 27495: 5, 5202: 5, 937: 5, 24200: 5, 1941: 5, 16392: 5, 23933: 5, 21739: 5, 4287: 5, 11603: 5, 24542: 5, 16678: 5, 26975: 5, 12932: 5, 25316: 5, 21144: 5, 27135: 5, 6347: 5, 22853: 5, 23040: 5, 9947: 5, 13534: 5, 9682: 5, 18637: 5, 204: 5, 7798: 5, 12778: 5, 2516: 5, 641: 5, 21445: 5, 7351: 5, 21386: 5}

    
    SNPsList = list(SNPsDic.keys())
    print(len(SNPsList))
    # needs to be replaced by the path of files
    path = ''

    print("current path of working space= " + path)
  
    # This is the smaller dataset (we found the SNPs from this dataset)
    print("Reading Tier1_23May2022.csv ...")
    df= pd.read_csv(path + "Tier1_23May2022.csv")
        
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
