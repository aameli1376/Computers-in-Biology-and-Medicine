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

    print("Select the most common SNPs from Familial dataset and check those SNPs quality for getting the accuracy on NINDS2 (I got intersection between them):")    
    
    SNPsDic = {150: 9, 193: 8, 277: 7, 90: 7, 2077: 7, 25: 7, 67: 7, 179: 7, 3404: 6, 101: 6, 191: 6, 169: 6, 30: 5, 2: 5, 158: 5, 355: 5, 1365: 5, 2078: 5, 272: 5, 99: 5, 2194: 5, 3619: 5, 1216: 5, 2062: 5, 471: 5, 2914: 5, 135: 5, 232: 5, 482: 5, 233: 5, 1464: 5, 3420: 5}

    
    SNPsList = list(SNPsDic.keys())
    print(len(SNPsList))
    # needs to be replaced by the path of files
    path = ''

    print("current path of working space= " + path)
  
        
    # This is the smaller dataset (we found the SNPs from this dataset)
    print("Reading NINDS2_23May2022.csv ...")
    df= pd.read_csv(path + "NINDS2_23May2022.csv")
    
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
