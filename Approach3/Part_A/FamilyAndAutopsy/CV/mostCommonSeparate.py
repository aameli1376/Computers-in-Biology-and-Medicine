import os
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import RepeatedStratifiedKFold
import numpy as np
import math
import operator
# Formula to delete n random elements from a list:
import random


def delete_random_elems(input_list, n):
    to_delete = set(random.sample(range(len(input_list)), n))
    return [x for i, x in enumerate(input_list) if not i in to_delete]


# Note, this function doesn't take a seed value, so it will be different
# 	every time you run it.

if __name__ == "__main__":

    print("Select the most common SNPs from merged version of (Familial and Autopsy) dataset and check those SNPs quality for getting the accuracy:")    
    
    SNPsDic = {115824: 18, 136430: 18, 160285: 18, 70221: 16, 69808: 16, 37094: 16, 109477: 16, 102411: 16, 107594: 16, 68734: 15, 82229: 15, 113349: 15, 125677: 15, 12426: 14, 160725: 14, 65133: 14, 22732: 14, 164233: 14, 103520: 14, 74873: 14, 87755: 13, 100055: 13, 151022: 13, 175512: 13, 23163: 13, 30215: 13, 161461: 13, 25858: 13, 58583: 13, 161318: 13, 68775: 13, 102142: 13, 121073: 12, 173122: 12, 175410: 12, 69542: 12, 159332: 12, 67043: 12, 176003: 12, 96891: 12, 28996: 12, 119045: 12, 115862: 12, 133157: 12, 157353: 12, 59490: 12, 92613: 12, 113332: 12, 97789: 12, 83657: 12, 136754: 11, 157760: 11, 80202: 11, 65965: 11, 25353: 11, 104589: 11, 174182: 11, 137683: 11, 1159: 11, 62774: 11, 5977: 11, 159871: 11, 56981: 10, 36462: 10, 92646: 10, 170550: 10, 62958: 10, 35574: 10, 73414: 10, 17160: 10, 162756: 10, 149816: 10, 86870: 10, 51295: 10, 44104: 10, 7266: 10, 9287: 10, 171647: 10, 73343: 10, 123723: 10, 128315: 10, 90417: 10, 116352: 10, 38642: 10, 102355: 10, 161953: 10, 34630: 10, 4459: 10, 19842: 10, 31270: 10, 169176: 10, 138110: 10, 93131: 10, 73801: 10, 96477: 10, 72159: 10, 133152: 10, 131009: 10, 115386: 10, 166342: 10, 155434: 10, 76456: 10, 110738: 10, 144791: 9, 109144: 9, 153989: 9, 89763: 9, 56134: 9, 56209: 9, 76759: 9, 154329: 9, 142505: 9, 170744: 9, 168680: 9, 142649: 9, 113192: 9, 35621: 9, 20165: 9, 118368: 9, 105664: 9, 43056: 9, 95372: 9, 107573: 9, 76666: 9, 125975: 9, 138974: 9, 105583: 9, 110148: 9, 164691: 9, 144339: 9, 11751: 9, 71451: 9, 129381: 9, 168297: 9, 139455: 9, 3970: 9, 92807: 9, 29897: 9, 116576: 9, 8354: 9, 66358: 9, 124084: 9, 37640: 9, 81367: 9, 33517: 8, 53920: 8, 36964: 8, 11418: 8, 29040: 8, 77858: 8, 51610: 8, 162667: 8, 32196: 8, 99957: 8, 156920: 8, 36817: 8, 73219: 8, 110633: 8, 9070: 8, 134976: 8, 153460: 8, 84159: 8, 138690: 8, 135865: 8, 60525: 8, 170719: 8, 113702: 8, 77257: 8, 170380: 8, 36173: 8, 79516: 8, 120143: 8, 103243: 8, 8311: 8, 15003: 8, 59602: 8, 164479: 8, 161698: 8, 168918: 8, 144798: 8, 100326: 8, 110918: 8, 94439: 8, 142411: 8, 35264: 8, 94986: 8, 152901: 8, 64529: 8, 94169: 8, 142629: 8, 65625: 8, 10384: 8, 30426: 8, 110516: 8, 150717: 8, 1107: 8, 176391: 8, 144090: 8, 132835: 8, 79861: 8, 41052: 8, 114500: 8, 11651: 8, 43564: 8, 71780: 8, 109226: 8, 7370: 8, 160463: 8, 58512: 8, 36040: 8, 54408: 8, 85947: 7, 77360: 7, 49109: 7, 61577: 7, 13104: 7, 119081: 7, 97450: 7, 79259: 7, 143530: 7, 118443: 7, 26560: 7, 134631: 7, 70139: 7, 130560: 7, 166831: 7, 120979: 7, 29460: 7, 77484: 7, 108158: 7, 149770: 7, 144221: 7, 44467: 7, 37235: 7, 166941: 7, 39934: 7, 175875: 7, 58710: 7, 39457: 7, 131267: 7, 95228: 7, 7792: 7, 90006: 7, 127776: 7, 121212: 7, 7262: 7, 137236: 7, 134510: 7, 44698: 7, 139270: 7, 145368: 7, 79406: 7, 38375: 7, 4204: 7, 7523: 7, 133917: 7, 32981: 7, 24154: 7, 99357: 7, 78903: 7, 91167: 7, 168963: 7, 139507: 7, 142852: 6, 107570: 6, 46700: 6, 76449: 6, 8086: 6, 97107: 6, 139839: 6, 31610: 6, 161760: 6, 18378: 6, 155056: 6, 52316: 6, 11297: 6, 12307: 6, 90213: 6, 60308: 6, 112297: 6, 18166: 6, 97337: 6, 7515: 6, 16238: 6, 64500: 6, 163798: 6, 157743: 6, 159846: 6, 155597: 6, 114388: 6, 173844: 6, 44519: 6, 17756: 6, 114980: 6, 126677: 6, 112547: 6, 36826: 6, 39104: 6, 85316: 6, 2363: 6, 32272: 6, 125196: 6, 73215: 6, 73888: 6, 152371: 6, 38901: 6, 83814: 6, 134172: 6, 93147: 6, 167259: 6, 126143: 6, 133054: 6, 105691: 6, 114417: 6, 38830: 6, 89074: 6, 139114: 6, 171224: 6, 105297: 6, 175337: 6, 160136: 6, 170572: 6, 149609: 6, 79783: 6, 85815: 6, 171463: 6, 94192: 6, 134339: 6, 35859: 6, 143168: 6, 7813: 6, 31241: 6, 27189: 6, 167231: 6, 113221: 6, 47635: 6, 172295: 6, 170366: 6, 129816: 6, 137596: 6, 122953: 6, 173350: 6, 289: 6, 88735: 6, 21477: 6, 167719: 6, 143125: 6, 51164: 6, 161816: 5, 170757: 5, 116158: 5, 82277: 5, 13712: 5, 162765: 5, 151851: 5, 134018: 5, 168710: 5, 76799: 5, 8524: 5, 171268: 5, 123211: 5, 86059: 5, 20876: 5, 88596: 5, 93798: 5, 151191: 5, 90196: 5, 25988: 5, 143450: 5, 10826: 5, 138801: 5, 92211: 5, 146746: 5, 127251: 5, 175876: 5, 30556: 5, 158401: 5, 46832: 5, 98316: 5, 162507: 5, 157898: 5, 108645: 5, 110096: 5, 157369: 5, 32891: 5, 30718: 5, 81392: 5, 47629: 5, 121469: 5, 78908: 5, 20988: 5, 104083: 5, 139073: 5, 119999: 5, 47382: 5, 148190: 5, 157611: 5, 114703: 5, 53748: 5, 119410: 5, 23078: 5, 76018: 5, 85761: 5, 84635: 5, 37336: 5, 118435: 5, 20916: 5, 106820: 5, 36439: 5, 5788: 5, 132812: 5, 114358: 5, 107289: 5, 169835: 5, 50351: 5, 84048: 5, 166763: 5, 31852: 5, 23526: 5, 96794: 5, 76525: 5, 111288: 5, 124332: 5, 95043: 5, 97807: 5, 73194: 5, 118486: 5, 161331: 5, 48436: 5, 75019: 5, 152683: 5, 156085: 5, 43106: 5, 30498: 5, 41872: 5, 106180: 5, 139330: 5, 54725: 5, 48232: 5, 75421: 5, 44178: 5, 298: 5, 41907: 5, 90665: 5, 166178: 5, 174783: 5, 38836: 5, 8660: 5, 120126: 5, 17579: 5, 74504: 5, 66986: 5}

    
    SNPsList = list(SNPsDic.keys())
    print(len(SNPsList))
    # needs to be replaced by the path of files
    path = ''

    print("current path of working space= " + path)
  
        
    # This is the smaller dataset (we found the SNPs from this dataset)
    print("Reading FamilyAutopsy(MergedWithID).csv ...")
    df= pd.read_csv(path + "FamilyAutopsy(MergedWithID).csv")
    
    count_row = df.shape[0]  # gives number of row count
    count_col = df.shape[1]  # gives number of col count

    print("The Number Samples in This Dataset: " + str(count_row))
    print("The Number Features in This Dataset: " + str(count_col))


    # Generating a new column named 'fold' with values 1 to 5 that is randomly
    generatedFolds = np.random.permutation(np.repeat([1, 2, 3, 4, 5], math.ceil(count_row / 5)))

    extra_folds = len(generatedFolds) - count_row

    generatedFolds = delete_random_elems(generatedFolds, extra_folds)

    df['folds'] = generatedFolds
    # using counter
    print("Number of samples belong to fold 1: " + str(operator.countOf(df['folds'], 1)))
    print("Number of samples belong to fold 2: " + str(operator.countOf(df['folds'], 2)))
    print("Number of samples belong to fold 3: " + str(operator.countOf(df['folds'], 3)))
    print("Number of samples belong to fold 4: " + str(operator.countOf(df['folds'], 4)))
    print("Number of samples belong to fold 5: " + str(operator.countOf(df['folds'], 5)))

    print("Last 3 column of dataframe:")
    N = 3
    # Select last N columns of dataframe
    last_n_column = df.iloc[:, -N:]
    print(last_n_column)

    fold1 = df.loc[df["folds"] == 1, :]
    fold2 = df.loc[df["folds"] == 2, :]
    fold3 = df.loc[df["folds"] == 3, :]
    fold4 = df.loc[df["folds"] == 4, :]
    fold5 = df.loc[df["folds"] == 5, :]

    # train with fold 1,2,3,4 - test with fold 5
    fold1234 = pd.concat([fold1, fold2, fold3, fold4])

    # train with fold 1,2,3,5 - test with fold 4
    fold1235 = pd.concat([fold1, fold2, fold3, fold5])

    # train with fold 1,2,4,5 - test with fold 3
    fold1245 = pd.concat([fold1, fold2, fold4, fold5])

    # train with fold 1,3,4,5 - test with fold 2
    fold1345 = pd.concat([fold1, fold3, fold4, fold5])

    # train with fold 2,3,4,5 - test with fold 1
    fold2345 = pd.concat([fold2, fold3, fold4, fold5])

    acc_list = []
    acc_listF = []
    acc_listN2 = []

    # fold 5: train the model with fold 1,2,3,4 & test with fold 5
    print("########## fold 5: train the model with fold 1,2,3,4 & test with fold 5 ##########")
    x_train = fold1234.iloc[:, SNPsList]
    y_train = fold1234.Label
    x_test = fold5.iloc[:, SNPsList]
    y_test = fold5.Label

    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    # Store actual value vs predicted value
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'DatasetID': fold5.DatasetID, 'folds': fold5.folds})
    print(df)

    # accuracy for Family
    df1 = df.loc[df["DatasetID"] == 'F', :]
    accF = np.sum(np.equal(df1['Actual'], df1['Predicted'])) / len(df1['Actual'])

    # accuracy for NINDS2
    df2 = df.loc[df["DatasetID"] == 'A', :]
    accN2 = np.sum(np.equal(df2['Actual'], df2['Predicted'])) / len(df2['Actual'])

    acc = metrics.accuracy_score(y_test, y_pred)
    acc_list.append(acc)
    acc_listF.append(accF)
    acc_listN2.append(accN2)

    print("Accuracy Total: ", str(acc * 100))
    print("Accuracy Family: ", str(accF * 100))
    print("Accuracy Autopsy: ", str(accN2 * 100))
    print('\n')
    print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))
    print(f"Classification report for classifier {clf}:\n"
          f"{metrics.classification_report(y_test, y_pred)}\n")

    #################################################################

    # fold 4: train the model with fold 1,2,3,5 & test with fold 4
    print("########## fold 4: train the model with fold 1,2,3,5 & test with fold 4 ##########")
    x_train = fold1235.iloc[:, SNPsList]
    y_train = fold1235.Label
    x_test = fold4.iloc[:, SNPsList]
    y_test = fold4.Label

    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    # Store actual value vs predicted value
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'DatasetID': fold4.DatasetID, 'folds': fold4.folds})
    print(df)

    # accuracy for Family
    df1 = df.loc[df["DatasetID"] == 'F', :]
    accF = np.sum(np.equal(df1['Actual'], df1['Predicted'])) / len(df1['Actual'])

    # accuracy for NINDS2
    df2 = df.loc[df["DatasetID"] == 'A', :]
    accN2 = np.sum(np.equal(df2['Actual'], df2['Predicted'])) / len(df2['Actual'])

    acc = metrics.accuracy_score(y_test, y_pred)
    acc_list.append(acc)
    acc_listF.append(accF)
    acc_listN2.append(accN2)

    print("Accuracy Total: ", str(acc * 100))
    print("Accuracy Family: ", str(accF * 100))
    print("Accuracy Autopsy: ", str(accN2 * 100))

    print('\n')
    print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))
    print(f"Classification report for classifier {clf}:\n"
          f"{metrics.classification_report(y_test, y_pred)}\n")

    #################################################################

    # fold 3: train the model with fold 1,2,4,5 & test with fold 3
    print("########## fold 3: train the model with fold 1,2,4,5 & test with fold 3 ##########")
    x_train = fold1245.iloc[:, SNPsList]
    y_train = fold1245.Label
    x_test = fold3.iloc[:, SNPsList]
    y_test = fold3.Label

    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    # Store actual value vs predicted value
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'DatasetID': fold3.DatasetID, 'folds': fold3.folds})
    print(df)

    # accuracy for Family
    df1 = df.loc[df["DatasetID"] == 'F', :]
    accF = np.sum(np.equal(df1['Actual'], df1['Predicted'])) / len(df1['Actual'])

    # accuracy for NINDS2
    df2 = df.loc[df["DatasetID"] == 'A', :]
    accN2 = np.sum(np.equal(df2['Actual'], df2['Predicted'])) / len(df2['Actual'])

    acc = metrics.accuracy_score(y_test, y_pred)
    acc_list.append(acc)
    acc_listF.append(accF)
    acc_listN2.append(accN2)

    print("Accuracy Total: ", str(acc * 100))
    print("Accuracy Family: ", str(accF * 100))
    print("Accuracy Autopsy: ", str(accN2 * 100))

    print('\n')
    print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))
    print(f"Classification report for classifier {clf}:\n"
          f"{metrics.classification_report(y_test, y_pred)}\n")

    #################################################################

    # fold 2: train the model with fold 1,3,4,5 & test with fold 2
    print("########## fold 2: train the model with fold 1,3,4,5 & test with fold 2 ##########")
    x_train = fold1345.iloc[:, SNPsList]
    y_train = fold1345.Label
    x_test = fold2.iloc[:, SNPsList]
    y_test = fold2.Label

    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    # Store actual value vs predicted value
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'DatasetID': fold2.DatasetID, 'folds': fold2.folds})
    print(df)

    # accuracy for Family
    df1 = df.loc[df["DatasetID"] == 'F', :]
    accF = np.sum(np.equal(df1['Actual'], df1['Predicted'])) / len(df1['Actual'])

    # accuracy for NINDS2
    df2 = df.loc[df["DatasetID"] == 'A', :]
    accN2 = np.sum(np.equal(df2['Actual'], df2['Predicted'])) / len(df2['Actual'])

    acc = metrics.accuracy_score(y_test, y_pred)
    acc_list.append(acc)
    acc_listF.append(accF)
    acc_listN2.append(accN2)

    print("Accuracy Total: ", str(acc * 100))
    print("Accuracy Family: ", str(accF * 100))
    print("Accuracy Autopsy: ", str(accN2 * 100))

    print('\n')
    print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))
    print(f"Classification report for classifier {clf}:\n"
          f"{metrics.classification_report(y_test, y_pred)}\n")

    #################################################################

    # fold 1: train the model with fold 2,3,4,5 & test with fold 1
    print("########## fold 1: train the model with fold 2,3,4,5 & test with fold 1 ##########")
    x_train = fold2345.iloc[:, SNPsList]
    y_train = fold2345.Label
    x_test = fold1.iloc[:, SNPsList]
    y_test = fold1.Label

    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    # Store actual value vs predicted value
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'DatasetID': fold1.DatasetID, 'folds': fold1.folds})
    print(df)

    # accuracy for Family
    df1 = df.loc[df["DatasetID"] == 'F', :]
    accF = np.sum(np.equal(df1['Actual'], df1['Predicted'])) / len(df1['Actual'])

    # accuracy for NINDS2
    df2 = df.loc[df["DatasetID"] == 'A', :]
    accN2 = np.sum(np.equal(df2['Actual'], df2['Predicted'])) / len(df2['Actual'])

    acc = metrics.accuracy_score(y_test, y_pred)
    acc_list.append(acc)
    acc_listF.append(accF)
    acc_listN2.append(accN2)

    print("Accuracy Total: ", str(acc * 100))
    print("Accuracy Family: ", str(accF * 100))
    print("Accuracy Autopsy: ", str(accN2 * 100))

    print('\n')
    print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))
    print(f"Classification report for classifier {clf}:\n"
          f"{metrics.classification_report(y_test, y_pred)}\n")

    print("Average accuracy total (among 5 rounds) is : %.2f" % (np.average(acc_list) * 100))
    print("SD of ACC is : %.2f" % (np.std(acc_list) * 100))

    print("Average accuracy Family (among 5 rounds) is : %.2f" % (np.average(acc_listF) * 100))
    print("SD of ACC is : %.2f" % (np.std(acc_listF) * 100))

    print("Average accuracy Autopsy (among 5 rounds) is : %.2f" % (np.average(acc_listN2) * 100))
    print("SD of ACC is : %.2f" % (np.std(acc_listN2) * 100))
