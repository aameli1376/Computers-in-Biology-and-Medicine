import pandas as pd


df1 = pd.read_csv("2Family.csv")
# Balancing dataframe 1
controls_subset = df1.loc[df1["Label"] == 1, :]
sampled_controls = controls_subset.sample(335)

cases_subset = df1.loc[df1["Label"] == 2, :]
sampled_cases = cases_subset.sample(642)

df1 = pd.concat([sampled_controls, sampled_cases])




df2 = pd.read_csv("2Autopsy.csv")
# Balancing dataframe 2
controls_subset = df2.loc[df2["Label"] == 1, :]
sampled_controls = controls_subset.sample(335)

cases_subset = df2.loc[df2["Label"] == 2, :]
sampled_cases = cases_subset.sample(642)

df2 = pd.concat([sampled_controls, sampled_cases])

# merging 2 dataframes with the same number of cases and controls
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

print("Number of label 1:")
print(len(result[result.Label == 1]))

print("Number of label 2:")
print(len(result[result.Label == 2]))

result.to_csv("FamilyAutopsy(MergedAndBalanced).csv", index=False)
