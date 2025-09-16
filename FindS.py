import pandas as pd

data = pd.read_csv("finds_data.csv")
concepts = data.iloc[:, :-1].values
target = data.iloc[:, -1].values

hypothesis = ['0'] * len(concepts[0])
for i, val in enumerate(target):
    if val == "Yes":
        hypothesis = concepts[i].copy()
        break

for i, val in enumerate(concepts):
    if target[i] == "Yes":
        for j in range(len(hypothesis)):
            if hypothesis[j] != val[j]:
                hypothesis[j] = '?'

print("Final Hypothesis:", hypothesis)
