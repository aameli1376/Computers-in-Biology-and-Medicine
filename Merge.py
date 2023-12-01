import pandas as pd


df1 = pd.read_csv("2Family.csv")
df2 = pd.read_csv("2Autopsy.csv")

frames = [df1, df2]

result = pd.concat(frames)

# shuffle the DataFrame rows
result = result.sample(frac = 1)

col_count = result.shape[1]
row_count = result.shape[0]

print("Number of Columns: ")
print(col_count)

print("Number of rows: ")
print(row_count)

result.to_csv("FamilyAutopsy(Merged).csv", index=False)
