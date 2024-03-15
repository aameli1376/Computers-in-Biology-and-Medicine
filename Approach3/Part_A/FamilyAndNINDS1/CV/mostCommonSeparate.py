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

    print("Select the most common SNPs from merged version of (Familial and NINDS1) dataset and check those SNPs quality for getting the accuracy:")    
    
    SNPsDic = {24427: 30, 54696: 29, 267064: 29, 26798: 28, 231972: 27, 3295: 27, 270632: 26, 189615: 26, 89265: 26, 107709: 26, 12441: 25, 273440: 25, 16243: 25, 61836: 24, 156841: 24, 257492: 24, 119981: 23, 165587: 23, 219229: 23, 115043: 23, 156834: 23, 180805: 23, 289232: 23, 295116: 23, 30005: 22, 94699: 22, 52507: 22, 209971: 22, 198601: 22, 188322: 22, 18022: 22, 262883: 21, 195799: 21, 237366: 21, 144816: 21, 267134: 21, 37021: 21, 213279: 21, 118865: 21, 131754: 21, 251451: 20, 197917: 20, 285668: 20, 278840: 20, 81186: 20, 49705: 20, 22296: 20, 47730: 20, 101338: 20, 154794: 20, 63607: 20, 166763: 20, 249287: 20, 10287: 19, 30003: 19, 100287: 19, 115163: 19, 139300: 19, 291647: 19, 260638: 19, 284951: 19, 249576: 19, 131480: 19, 101368: 19, 48378: 19, 182847: 19, 32385: 19, 152006: 18, 48202: 18, 251074: 18, 238498: 18, 188810: 18, 18403: 18, 22297: 18, 19574: 18, 274335: 18, 12475: 18, 287220: 18, 236872: 17, 61859: 17, 238431: 17, 52017: 17, 69605: 17, 287580: 17, 131479: 17, 110466: 17, 112730: 17, 82262: 17, 29714: 17, 175358: 17, 80764: 17, 80466: 17, 199229: 17, 75879: 16, 189462: 16, 164298: 16, 167820: 16, 189461: 16, 198467: 16, 22173: 16, 110449: 16, 165279: 16, 262472: 16, 223047: 16, 295963: 16, 63444: 16, 72468: 15, 266887: 15, 69022: 15, 133186: 15, 29581: 15, 299207: 15, 180807: 15, 75611: 15, 20216: 15, 205265: 15, 27239: 15, 169569: 15, 143475: 15, 251877: 15, 136443: 15, 127001: 15, 60513: 15, 84517: 15, 267066: 15, 133956: 15, 122049: 15, 40273: 15, 225830: 14, 187415: 14, 99328: 14, 219192: 14, 81165: 14, 136250: 14, 177727: 14, 258039: 14, 30428: 14, 191366: 14, 129311: 14, 3652: 14, 104820: 14, 148753: 14, 95974: 14, 68202: 14, 199080: 14, 31041: 14, 70560: 14, 130807: 14, 13118: 14, 170156: 14, 21566: 14, 210377: 14, 115005: 14, 188396: 14, 228179: 14, 92708: 14, 67835: 14, 28385: 14, 143655: 14, 146086: 14, 180556: 14, 160062: 13, 28929: 13, 8219: 13, 155171: 13, 200713: 13, 70609: 13, 55786: 13, 239681: 13, 50745: 13, 269228: 13, 151313: 13, 152240: 13, 187824: 13, 214142: 13, 237835: 13, 237509: 13, 159616: 13, 232724: 12, 302101: 12, 189519: 12, 100083: 12, 289438: 12, 302614: 12, 116247: 12, 279297: 12, 118785: 12, 165911: 12, 127579: 12, 171845: 12, 152306: 12, 231374: 12, 184997: 12, 28782: 12, 6348: 12, 9426: 12, 13917: 12, 239102: 12, 63054: 12, 165741: 12, 86666: 12, 270197: 12, 108347: 12, 40115: 12, 28729: 12, 65703: 12, 1133: 12, 53612: 12, 22172: 12, 165747: 12, 224839: 12, 271077: 11, 279558: 11, 302624: 11, 223944: 11, 198465: 11, 166890: 11, 263495: 11, 213364: 11, 99436: 11, 261186: 11, 6741: 11, 262626: 11, 261907: 11, 2736: 11, 280742: 11, 234627: 11, 223800: 11, 136111: 11, 142909: 11, 166129: 11, 245200: 11, 237401: 11, 247730: 11, 237045: 11, 128517: 11, 76285: 11, 305480: 11, 123535: 11, 266653: 11, 228051: 11, 131016: 11, 203888: 11, 35962: 11, 237242: 11, 84476: 11, 299234: 11, 240478: 11, 273864: 11, 160982: 11, 86665: 11, 224349: 11, 111021: 11, 109875: 11, 251454: 11, 26434: 11, 89377: 11, 270156: 11, 114288: 11, 250353: 10, 288366: 10, 188976: 10, 280973: 10, 195952: 10, 60522: 10, 189560: 10, 58034: 10, 220735: 10, 45662: 10, 276381: 10, 26348: 10, 127178: 10, 165050: 10, 173798: 10, 261906: 10, 47272: 10, 283408: 10, 172106: 10, 298766: 10, 199647: 10, 118349: 10, 44034: 10, 25200: 10, 6783: 10, 173980: 10, 246364: 10, 45471: 10, 289596: 10, 209379: 10, 60632: 10, 208606: 10, 137694: 10, 304350: 10, 122205: 10, 198008: 10, 236461: 10, 137696: 10, 54856: 10, 160157: 10, 85001: 10, 256732: 10, 15939: 10, 226298: 10, 183052: 10, 206237: 10, 194892: 10, 229630: 10, 271268: 10, 58374: 10, 207108: 10, 72750: 10, 275599: 10, 183233: 10, 51487: 10, 197096: 10, 280477: 10, 53648: 10, 117190: 10, 303278: 10, 84689: 10, 178898: 9, 271079: 9, 165049: 9, 218513: 9, 38160: 9, 101381: 9, 13100: 9, 154432: 9, 166259: 9, 65838: 9, 49348: 9, 304052: 9, 249250: 9, 305141: 9, 263679: 9, 40015: 9, 242746: 9, 221856: 9, 259286: 9, 189880: 9, 287717: 9, 142099: 9, 104723: 9, 175910: 9, 256344: 9, 38287: 9, 292534: 9, 207283: 9, 192414: 9, 20841: 9, 26639: 9, 231207: 9, 263688: 9, 81646: 9, 10552: 9, 229200: 9, 79998: 9, 131629: 9, 126830: 9, 50382: 9, 175159: 9, 246631: 9, 193783: 9, 21674: 9, 191075: 9, 207579: 9, 295511: 9, 176665: 9, 304199: 9, 302716: 9, 69162: 9, 209442: 8, 298715: 8, 61463: 8, 119166: 8, 1037: 8, 81562: 8, 275241: 8, 25252: 8, 200011: 8, 90791: 8, 256574: 8, 48637: 8, 304875: 8, 127086: 8, 279552: 8, 154431: 8, 157003: 8, 22418: 8, 63662: 8, 246720: 8, 43018: 8, 231798: 8, 62956: 8, 21069: 8, 227311: 8, 115440: 8, 221033: 8, 284260: 8, 33596: 8, 72166: 8, 82988: 8, 117813: 8, 228340: 8, 114392: 8, 64058: 8, 229325: 8, 166615: 8, 29849: 8, 61071: 8, 141654: 8, 165308: 8, 299800: 8, 211749: 8, 84890: 8, 244330: 8, 303416: 8, 158999: 8, 212479: 8, 13465: 8, 206236: 8, 120928: 8, 274706: 8, 89501: 8, 33261: 8, 100582: 8, 149305: 8, 70774: 8, 85150: 8, 123235: 8, 8929: 8, 172160: 8, 249479: 8, 71788: 8, 70641: 8, 27857: 8, 18219: 8, 230209: 8, 17565: 8, 126700: 8, 174408: 8, 183704: 8, 295418: 8, 166614: 8, 173699: 8, 42270: 7, 24611: 7, 243898: 7, 288528: 7, 19238: 7, 72625: 7, 275301: 7, 139711: 7, 186884: 7, 66401: 7, 220643: 7, 198463: 7, 56159: 7, 151551: 7, 17180: 7, 273638: 7, 244049: 7, 285964: 7, 177089: 7, 121035: 7, 144154: 7, 20127: 7, 96092: 7, 121611: 7, 82439: 7, 254932: 7, 116898: 7, 187256: 7, 158261: 7, 300296: 7, 204856: 7, 55931: 7, 245123: 7, 297636: 7, 171585: 7, 257332: 7, 83671: 7, 296555: 7, 56000: 7, 299422: 7, 81667: 7, 164211: 7, 168113: 7, 227775: 7, 70429: 7, 171362: 7, 285816: 7, 251595: 7, 13866: 7, 214768: 7, 67590: 7, 37733: 7, 298853: 7, 304200: 7, 179755: 7, 275957: 7, 163776: 7, 274621: 7, 273193: 7, 43826: 7, 77090: 7, 8354: 7, 299645: 7, 9661: 7, 96096: 7, 146849: 7, 74596: 7, 282140: 7, 137302: 7, 274430: 7, 305130: 7, 80809: 7, 289182: 7, 137311: 7, 247764: 7, 231577: 7, 232651: 7, 275008: 7, 158320: 7, 132020: 7, 45475: 6, 84946: 6, 128520: 6, 137056: 6, 66391: 6, 11102: 6, 21153: 6, 194157: 6, 199512: 6, 276656: 6, 214426: 6, 281582: 6, 277314: 6, 70346: 6, 125176: 6, 192721: 6, 304373: 6, 300003: 6, 213423: 6, 154433: 6, 258935: 6, 286962: 6, 230883: 6, 156430: 6, 159038: 6, 93309: 6, 183941: 6, 28735: 6, 187091: 6, 162356: 6, 248227: 6, 210876: 6, 84913: 6, 289183: 6, 39394: 6, 67541: 6, 296159: 6, 92586: 6, 10483: 6, 239148: 6, 127190: 6, 83669: 6, 300647: 6, 39007: 6, 265161: 6, 249904: 6, 284904: 6, 138918: 6, 204291: 6, 94891: 6, 178648: 6, 164568: 6, 177603: 6, 295010: 6, 110787: 6, 272473: 6, 250522: 6, 54391: 6, 286259: 6, 119040: 6, 40743: 6, 114673: 6, 293713: 6, 54301: 6, 140948: 6, 73611: 6, 229581: 6, 26524: 6, 256057: 6, 189507: 6, 9128: 6, 223356: 6, 197950: 6, 164267: 6, 17785: 6, 273197: 6, 257268: 6, 237552: 6, 257902: 6, 176752: 6, 277767: 6, 196091: 6, 242291: 6, 51929: 6, 226449: 6, 268874: 6, 145665: 6, 293708: 6, 134420: 6, 12859: 6, 61493: 6, 4185: 6, 265377: 6, 105287: 6, 277633: 6, 182658: 6, 303072: 6, 57364: 6, 153012: 6, 206235: 6, 32662: 6, 117308: 6, 157568: 6, 55997: 6, 266021: 6, 300006: 6, 214599: 6, 264042: 6, 283729: 6, 92372: 6, 206767: 6, 300088: 6, 49407: 6, 251964: 6, 39273: 6, 259608: 6, 304740: 6, 225288: 6, 242960: 6, 116926: 6, 264416: 6, 90442: 6, 238436: 6, 28693: 6, 267048: 6, 245202: 6, 162822: 6, 257678: 6, 285968: 6, 254680: 6, 145082: 6, 135173: 6, 141592: 5, 73250: 5, 79973: 5, 219874: 5, 181537: 5, 299206: 5, 260302: 5, 127402: 5, 203604: 5, 140864: 5, 181738: 5, 25972: 5, 236890: 5, 735: 5, 228005: 5, 240353: 5, 155936: 5, 256223: 5, 305538: 5, 3349: 5, 209625: 5, 293344: 5, 20908: 5, 257878: 5, 66615: 5, 221193: 5, 221196: 5, 79974: 5, 199901: 5, 148378: 5, 104907: 5, 253162: 5, 77521: 5, 70161: 5, 287824: 5, 190755: 5, 259411: 5, 207960: 5, 243903: 5, 269588: 5, 80156: 5, 288317: 5, 95351: 5, 117239: 5, 1829: 5, 304720: 5, 205689: 5, 244925: 5, 254142: 5, 282654: 5, 249285: 5, 275249: 5, 74832: 5, 151192: 5, 109079: 5, 42605: 5, 100251: 5, 207218: 5, 71711: 5, 154298: 5, 260010: 5, 3604: 5, 50703: 5, 69832: 5, 168949: 5, 45663: 5, 132716: 5, 7277: 5, 131708: 5, 181319: 5, 194712: 5, 298846: 5, 23698: 5, 62782: 5, 273994: 5, 129356: 5, 59537: 5, 44342: 5, 265370: 5, 169556: 5, 159078: 5, 300756: 5, 299497: 5, 137486: 5, 194916: 5, 74096: 5, 89496: 5, 155555: 5, 69594: 5, 299415: 5, 248206: 5, 237413: 5, 153718: 5, 299112: 5, 14312: 5, 205885: 5, 152728: 5, 146750: 5, 274000: 5, 291015: 5, 175547: 5, 269229: 5, 112394: 5, 151478: 5, 266640: 5, 3041: 5, 100783: 5, 299238: 5, 59356: 5, 200495: 5, 259477: 5, 250513: 5, 146702: 5, 293634: 5, 79700: 5, 69178: 5, 276749: 5, 262831: 5, 123427: 5, 25272: 5, 32517: 5, 191719: 5, 92882: 5, 201392: 5, 248032: 5, 160860: 5, 101286: 5, 131283: 5, 189792: 5, 184255: 5, 266592: 5, 111874: 5, 89848: 5, 202649: 5, 158321: 5, 115637: 5, 35029: 5, 72374: 5, 102422: 5, 170450: 5, 199649: 5, 204911: 5, 111731: 5, 61178: 5, 44932: 5, 156608: 5, 87705: 5, 191608: 5, 56086: 5, 257199: 5, 83351: 5, 63802: 5, 245149: 5, 182911: 5, 214506: 5, 300248: 5, 90583: 5, 284029: 5, 293279: 5, 75085: 5, 217631: 5, 168346: 5, 85583: 5, 111944: 5, 261152: 5, 258908: 5, 253088: 5, 7283: 5, 239177: 5, 180070: 5, 198446: 5, 14235: 5, 121206: 5, 3492: 5, 249608: 5, 46044: 5, 193215: 5, 156970: 5, 130064: 5, 304975: 5, 151622: 5, 45517: 5, 86019: 5, 179254: 5, 62591: 5, 153715: 5, 35198: 5, 295432: 5, 20480: 5, 228582: 5, 148607: 5, 31886: 5, 243278: 5, 299330: 5, 178062: 5, 246283: 5, 230234: 5, 304828: 5, 204666: 5, 252144: 5, 124941: 5}

    
    SNPsList = list(SNPsDic.keys())
    print(len(SNPsList))
    # needs to be replaced by the path of files
    path = ''

    print("current path of working space= " + path)
  
        
    # This is the smaller dataset (we found the SNPs from this dataset)
    print("Reading FamilyNINDS1(MergedWithID).csv ...")
    df= pd.read_csv(path + "FamilyNINDS1(MergedWithID).csv")
    
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
    df2 = df.loc[df["DatasetID"] == 'N1', :]
    accN2 = np.sum(np.equal(df2['Actual'], df2['Predicted'])) / len(df2['Actual'])

    acc = metrics.accuracy_score(y_test, y_pred)
    acc_list.append(acc)
    acc_listF.append(accF)
    acc_listN2.append(accN2)

    print("Accuracy Total: ", str(acc * 100))
    print("Accuracy Family: ", str(accF * 100))
    print("Accuracy NINDS1: ", str(accN2 * 100))
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
    df2 = df.loc[df["DatasetID"] == 'N1', :]
    accN2 = np.sum(np.equal(df2['Actual'], df2['Predicted'])) / len(df2['Actual'])

    acc = metrics.accuracy_score(y_test, y_pred)
    acc_list.append(acc)
    acc_listF.append(accF)
    acc_listN2.append(accN2)

    print("Accuracy Total: ", str(acc * 100))
    print("Accuracy Family: ", str(accF * 100))
    print("Accuracy NINDS1: ", str(accN2 * 100))

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
    df2 = df.loc[df["DatasetID"] == 'N1', :]
    accN2 = np.sum(np.equal(df2['Actual'], df2['Predicted'])) / len(df2['Actual'])

    acc = metrics.accuracy_score(y_test, y_pred)
    acc_list.append(acc)
    acc_listF.append(accF)
    acc_listN2.append(accN2)

    print("Accuracy Total: ", str(acc * 100))
    print("Accuracy Family: ", str(accF * 100))
    print("Accuracy NINDS1: ", str(accN2 * 100))

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
    df2 = df.loc[df["DatasetID"] == 'N1', :]
    accN2 = np.sum(np.equal(df2['Actual'], df2['Predicted'])) / len(df2['Actual'])

    acc = metrics.accuracy_score(y_test, y_pred)
    acc_list.append(acc)
    acc_listF.append(accF)
    acc_listN2.append(accN2)

    print("Accuracy Total: ", str(acc * 100))
    print("Accuracy Family: ", str(accF * 100))
    print("Accuracy NINDS1: ", str(accN2 * 100))

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
    df2 = df.loc[df["DatasetID"] == 'N1', :]
    accN2 = np.sum(np.equal(df2['Actual'], df2['Predicted'])) / len(df2['Actual'])

    acc = metrics.accuracy_score(y_test, y_pred)
    acc_list.append(acc)
    acc_listF.append(accF)
    acc_listN2.append(accN2)

    print("Accuracy Total: ", str(acc * 100))
    print("Accuracy Family: ", str(accF * 100))
    print("Accuracy NINDS1: ", str(accN2 * 100))

    print('\n')
    print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))
    print(f"Classification report for classifier {clf}:\n"
          f"{metrics.classification_report(y_test, y_pred)}\n")

    print("Average accuracy total (among 5 rounds) is : %.2f" % (np.average(acc_list) * 100))
    print("SD of ACC is : %.2f" % (np.std(acc_list) * 100))

    print("Average accuracy Family (among 5 rounds) is : %.2f" % (np.average(acc_listF) * 100))
    print("SD of ACC is : %.2f" % (np.std(acc_listF) * 100))

    print("Average accuracy NINDS1 (among 5 rounds) is : %.2f" % (np.average(acc_listN2) * 100))
    print("SD of ACC is : %.2f" % (np.std(acc_listN2) * 100))
