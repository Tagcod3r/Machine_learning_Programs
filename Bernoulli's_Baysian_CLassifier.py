import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

data = pd.read_csv("naive_bayes_data.csv")
X = pd.get_dummies(data.drop("PlayTennis", axis=1))
y = data["PlayTennis"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = CategoricalNB()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Predictions:", pred)
print("Accuracy:", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred, pos_label='Yes',zero_division=0))
print("Recall:", recall_score(y_test, pred, pos_label='Yes',zero_division=0))
