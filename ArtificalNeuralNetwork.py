import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

data = pd.read_csv("ann_data.csv")
X = data[["Feature1", "Feature2"]]
y = data["Label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

model = MLPClassifier(hidden_layer_sizes=(4,), max_iter=1000)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Predictions:", pred)
print("Actual     :", y_test.values)
