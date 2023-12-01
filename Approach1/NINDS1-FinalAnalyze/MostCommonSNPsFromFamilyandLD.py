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

    print("Select the most common SNPs from Familial dataset and check those SNPs quality for getting the accuracy on NINDS1 dataset after adding SNPs in LD")
    
    SNPsDic = {'rs4791252': 50, 'rs10062249': 50, 'rs4291034': 50, 'rs1784592': 50, 'rs1427217': 50, 'rs1767356': 50, 'rs34131647': 49, 'rs1054140': 49, 'rs6501638': 49, 'rs1974602': 48, 'rs7107938': 47, 'rs147263': 46, 'rs4878117': 45, 'rs7287107': 44, 'rs985292': 44, 'rs1991899': 43, 'rs117836': 43, 'rs9353427': 42, 'rs4691962': 42, 'rs265533': 40, 'rs1980236': 40, 'rs4619848': 40, 'rs1492660': 40, 'rs12269857': 39, 'rs11690832': 39, 'rs1117057': 38, 'rs1411023': 38, 'rs293290': 36, 'rs1335592': 36, 'rs1100479': 35, 'rs4843643': 34, 'rs1539847': 34, 'rs6463448': 34, 'rs1348657': 32, 'rs6944658': 31, 'rs1880342': 30, 'rs780179': 30, 'rs2298632': 29, 'rs977576': 29, 'rs1437656': 28, 'rs9957018': 27, 'rs2660888': 27, 'rs1730922': 27, 'rs504963': 27, 'rs4672667': 27, 'rs4364877': 26, 'rs854100': 26, 'rs6590810': 26, 'rs4667404': 26, 'rs531099': 26, 'rs13247910': 25, 'rs239330': 25, 'rs6573560': 25, 'rs4507403': 24, 'rs923784': 24, 'rs10400188': 24, 'rs2529179': 23, 'rs4845396': 23, 'rs4790007': 23, 'rs2370985': 22, 'rs7796909': 22, 'rs10783629': 22, 'rs2312664': 22, 'rs1476644': 21, 'rs2240308': 21, 'rs891404': 21, 'rs2279078': 20, 'rs7589928': 20, 'rs330575': 20, 'rs1466255': 20, 'rs2053730': 20, 'rs359486': 20, 'rs315100': 20, 'rs7741285': 19, 'rs216255': 19, 'rs4666548': 19, 'rs2648005': 19, 'rs7566044': 18, 'rs757134': 18, 'rs1594613': 18, 'rs4077636': 18, 'rs11785599': 17, 'rs243229': 17, 'rs6496120': 17, 'rs933724': 17, 'rs2064262': 17, 'rs7475754': 17, 'rs2067634': 17, 'rs10501819': 16, 'rs3026620': 16, 'rs1934656': 16, 'rs671098': 16, 'rs580333': 16, 'rs1090222': 16, 'rs1979260': 16, 'rs2185065': 15, 'rs2070762': 15, 'rs1372804': 15, 'rs3791426': 15, 'rs4349734': 15, 'rs965078': 14, 'rs6520090': 14, 'rs96264': 14, 'rs194435': 14, 'rs2419857': 14, 'rs763869': 14, 'rs10885087': 14, 'rs6942930': 13, 'rs595533': 13, 'rs2551043': 13, 'rs945255': 13, 'rs886364': 13, 'rs101646': 13, 'rs1793733': 13, 'rs5743942': 13, 'rs1539563': 13, 'rs7624896': 13, 'rs9514046': 12, 'rs10870149': 12, 'rs36720': 12, 'rs677508': 12, 'rs1943707': 12, 'rs2035875': 12, 'rs849165': 12, 'rs10016872': 12, 'rs1230544': 12, 'rs5759274': 12, 'rs11727767': 12, 'rs11003138': 12, 'rs4884865': 11, 'rs9687393': 11, 'rs59896': 11, 'rs12959212': 11, 'rs1388052': 11, 'rs9815397': 11, 'rs10511705': 11, 'rs7152906': 11, 'rs4783276': 11, 'rs7480563': 11, 'rs79139': 11, 'rs1801274': 11, 'rs755043': 11, 'rs2113608': 11, 'rs2044577': 11, 'rs10772939': 11, 'rs6754105': 10, 'rs597314': 10, 'rs1415612': 10, 'rs4972300': 10, 'rs3830': 10, 'rs751396': 10, 'rs408126': 10, 'rs4835530': 10, 'rs4766950': 10, 'rs848768': 10, 'rs7039377': 10, 'rs2279086': 10, 'rs4877475': 10, 'rs9303900': 10, 'rs4415': 10, 'rs448998': 10, 'rs374473': 10, 'rs1015759': 10, 'rs4731528': 9, 'rs3484853': 9, 'rs675829': 9, 'rs1881612': 9, 'rs2203805': 9, 'rs2125139': 9, 'rs4131667': 9, 'rs2374292': 9, 'rs1890020': 9, 'rs723157': 9, 'rs10858383': 9, 'rs4919908': 9, 'rs1200826': 8, 'rs99490': 8, 'rs736537': 8, 'rs432379': 8, 'rs600450': 8, 'rs219822': 8, 'rs157982': 8, 'rs10754849': 8, 'rs2091735': 8, 'rs3010815': 8, 'rs2031859': 8, 'rs4922518': 8, 'rs10885058': 8, 'rs2976562': 8, 'rs12490122': 8, 'rs1715385': 8, 'rs2434449': 8, 'rs263877': 8, 'rs591979': 8, 'rs80720': 8, 'rs2214534': 8, 'rs1817255': 8, 'rs223976': 7, 'rs2157640': 7, 'rs33557': 7, 'rs2604299': 7, 'rs962472': 7, 'rs31042': 7, 'rs1176486': 7, 'rs2398180': 7, 'rs659798': 7, 'rs314300': 7, 'rs662129': 7, 'rs1809005': 7, 'rs7826398': 7, 'rs2268550': 7, 'rs1447690': 7, 'rs2153417': 7, 'rs717966': 7, 'rs393486': 7, 'rs12505604': 7, 'rs1452454': 6, 'rs301975': 6, 'rs10783028': 6, 'rs3892715': 6, 'rs2204345': 6, 'rs10147042': 6, 'rs2692464': 6, 'rs5746800': 6, 'rs283427': 6, 'rs10800319': 6, 'rs7331762': 6, 'rs235456': 6, 'rs873134': 6, 'rs648356': 6, 'rs10919960': 6, 'rs2288404': 6, 'rs4272874': 6, 'rs2486378': 6, 'rs2241185': 6, 'rs1925636': 6, 'rs1964196': 5, 'rs8045257': 5, 'rs9451714': 5, 'rs115243': 5, 'rs7812836': 5, 'rs4318026': 5, 'rs3807663': 5, 'rs337113': 5, 'rs10500362': 5, 'rs1026825': 5, 'rs10800750': 5, 'rs757859': 5, 'rs10030740': 5, 'rs10470554': 5, 'rs253720': 5, 'rs134794': 5, 'rs4270088': 5, 'rs1328555': 5, 'rs1561234': 5, 'rs3858419': 5, 'rs3801263': 5, 'rs4984406': 5, 'rs183917': 5, 'rs7534836': 5, 'rs1950829': 5, 'rs2458424': 5, 'rs4783024': 5, 'rs2728487': 5, 'rs2160974': 5, 'rs7506330': 5, 'rs4668218': 5}

    
    SNPsList = list(SNPsDic.keys())
    print(len(SNPsList))

    #path = '/scratch/fs2/usefi/pd/datasets/NINDS2-FinalAnalyze/Datasets/'

    #print("current path of working space= " + path)
  
    
    # SNPs that are in LD with our most common SNPs
    LDList = []    
    f1 = open('/scratch/fs2/usefi/pd/datasets/All_Aproaches/Approach1/NINDS1-FinalAnalyze/SNPsinLDwithmostcommonSNPs.txt', 'r')
    for line1 in f1:
        LDList.append(line1.strip())  # We don't want newlines in our list, do we?
    
    ExtendedSNPsList = list(set(SNPsList + LDList))
    #common_elements = np.intersect1d(SNPsList, LDList)
        
    # This is the smaller dataset (we found the SNPs from this dataset)
    print("Reading PD_1_NINDS_28Jan2022.csv ...")
    df= pd.read_csv("PD_1_NINDS_28Jan2022.csv")
    
    count_row = df.shape[0]  # gives number of row count
    count_col = df.shape[1]  # gives number of col count

    print("The Number Samples in This Dataset: " + str(count_row))
    print("The Number Features in This Dataset: " + str(count_col))

    # X contains selected features except the labels
    x = df.iloc[:, df.columns.isin(ExtendedSNPsList)]

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
    print("SD of ACC is : %.2f" % (np.std(acc_list) * 100) )