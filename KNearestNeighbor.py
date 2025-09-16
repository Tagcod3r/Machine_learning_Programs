import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Create sample dataset
X, y = make_classification(n_samples=20, n_features=2, n_redundant=0, 
                          n_informative=2, n_clusters_per_class=1, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predictions
predictions = knn.predict(X_test)

# Print results
print("KNN Classification Results:")
print("Actual vs Predicted:")
correct = 0
wrong = 0

for i, (actual, pred) in enumerate(zip(y_test, predictions)):
    status = "Correct" if actual == pred else "Wrong"
    print(f"Sample {i+1}: Actual={actual}, Predicted={pred} - {status}")
    if actual == pred:
        correct += 1
    else:
        wrong += 1

print(f"\nSummary:")
print(f"Correct predictions: {correct}")
print(f"Wrong predictions: {wrong}")
print(f"Accuracy: {correct/(correct+wrong)*100:.2f}%")
