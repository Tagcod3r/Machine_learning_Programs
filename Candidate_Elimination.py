import pandas as pd

data = pd.read_csv("finds_data.csv")
concepts = data.iloc[:, :-1].values
target = data.iloc[:, -1].values

def consistent(h, x):
    return all(h[i] == x[i] or h[i] == '?' for i in range(len(h)))

s = list(concepts[0]) if target[0] == 'Yes' else ['?' for _ in range(len(concepts[0]))]
g = [['?' for _ in range(len(s))]]

for i in range(len(concepts)):
    if target[i] == 'Yes':
        for j in range(len(s)):
            if s[j] != concepts[i][j]:
                s[j] = '?'
    else:
        new_g = []
        for h in g:
            for j in range(len(h)):
                if h[j] == '?':
                    for val in set(data.iloc[:, j]):
                        if val != s[j]:
                            h1 = h.copy()
                            h1[j] = val
                            if consistent(h1, concepts[i]) == False:
                                new_g.append(h1)
        g = new_g or g

print("S:", s)
print("G:", g)
