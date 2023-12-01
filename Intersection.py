import pandas as pd
import os

if __name__ == '__main__':
    current_directory = os.path.abspath(os.getcwd())
    print(current_directory)

    # 1st: read both dataset csv file
    Autopsy = pd.read_csv("1Autopsy.csv")
    print(Autopsy.info())

    Family = pd.read_csv("1Family.csv")
    print(Family.info())

    # 2nd: create intersection of those 2 datasets
    common_col = [col for col in Autopsy.columns if col in Family.columns]

    Autopsy = Autopsy[common_col]
    Family = Family[common_col]

    print(Autopsy.info())
    print("---------------------------------------------------------------------")
    print(Family.info())

    file1 = open("Common_Column.txt", "a")
    file1.write(str(common_col))
    file1.close()
    print(common_col)

    Autopsy.to_csv('AutopsyNew.csv', index=False)
    Family.to_csv('FamilyNew.csv', index=False)



