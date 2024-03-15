import pandas as pd
import numpy as np
import time
import os
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn import metrics
from reduction import svfs
from collections import Counter
import csv

def main():

    dataset_list = ['FamilyNINDS1(Merged)']
    # needs to be replaced by the path of files
    path = ""
    # Parameters
    k = 50
    th_irr = 3
    th_red = 4
    alpha = 50
    beta = 5

    print(os.path.abspath(os.getcwd()))
    for dataset in dataset_list:
        print("\nDataset: ", dataset)
        print("Loading Dataset")
        data = pd.read_csv(path + dataset + ".csv", header=None,skiprows=1)
        print(data.info())

        dataX = data.copy().iloc[:, :-1]
        dataY = data.copy().iloc[:, data.shape[1] - 1]

        
        file = open('selected_features_of_each_round.csv', 'w+', newline='')
        with file:
          counter = 0
          acc_list = []
          features_list = []
          index_features_list = []
          # Split into train and test
          k_fold = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)
          start_time = time.time()
          for train_idx, test_dix in k_fold.split(dataX, dataY):
              train_x, test_x = dataX.iloc[train_idx, :].copy(), dataX.iloc[test_dix, :].copy()
              train_y, test_y = dataY.iloc[train_idx].copy(), dataY.iloc[test_dix].copy()
              list_features = []
              best_acc = 0
              best_idx = 0
              # best_clf = 0
              index_features_list1 = []
              fs = svfs(train_x, train_y, th_irr, 1.7, th_red, k, alpha, beta)
              high_x = fs.high_rank_x()
              reduced_data = fs.reduction()
              clean_features = high_x[reduced_data]
              dic_cls = fs.selection()
              idx = 0
              for key, value in dic_cls.items():
                  list_features.append(clean_features[key])
                  X_train = train_x.iloc[:, list_features].copy()
                  X_test = test_x.iloc[:, list_features].copy()
                  clf = RandomForestClassifier()
  
                  clf.fit(X_train, train_y)
                  y_pred = clf.predict(X_test)
                  acc = metrics.accuracy_score(test_y, y_pred)
                  if acc > best_acc:
                      best_acc = acc
                      best_idx = list_features.copy()
                      # best_clf = clf
                  idx += 1
              print("Best ACC is: %.2f" % (best_acc * 100), "for ", len(best_idx), " # of features")

              acc_list.append(best_acc)
              features_list.append(len(best_idx))
              index_features_list.extend(best_idx)
              index_features_list1.extend(best_idx)
                            
              counter = counter + 1
              write = csv.writer(file)
              write.writerows([index_features_list1])
              print("Round " + str(counter) + ": ")
              print(index_features_list1)
              print("------------------------------------")
          file.close()   
          mostFreq = Counter(index_features_list)
          print(mostFreq)
  
          print("Average ACC is : %.2f" % (np.average(acc_list) * 100))
          print("Average # of Features is : %.2f" % (np.average(features_list)))
          print("SD of ACC is : %.2f" % (np.std(acc_list) * 100) + " SD of Feature is : %.2f" % (np.std(features_list)))
          print("Running Time: %s seconds\n\n" % round((time.time() - start_time), 2))
  

main()
