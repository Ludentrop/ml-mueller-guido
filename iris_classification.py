# -*- coding: utf-8 -*-
"""
1. X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

2. knn = KNeighborsClassifier(n_neighbors=1)

3. knn.fit(X_train, y_train)

4. print("Correctness on the test set: {knn.score(X_test, y_test):.2f}")
"""
# !git clone https://github.com/amueller/introduction_to_ml_with_python

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import introduction_to_ml_with_python.mglearn as mglearn


knn = KNeighborsClassifier(n_neighbors=1)
iris_dataset = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

print(f'{X_train.shape=}')
print(f'{y_train.shape=}')
print()
print(f'{X_test.shape=}')
print(f'{y_test.shape=}')

# create dataframe from data of X_train massive
# mark columns using raws in iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# creating a scattering matrix from dataframe, set points colors by y_train
grr = scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)

knn.fit(X_train, y_train)

X_new = np.array([[5, 2.9, 1, 0.2]])

prediction = knn.predict(X_new)
print(f"Prediction: {prediction}")
print(f"Predicted marks: {iris_dataset['target_names'][prediction]}")
print()
y_pred = knn.predict(X_test)
print(f"Predictions for the test set:\n {y_pred}")
print()
print(f"Correctness on the test set: {np.mean(y_pred == y_test):.2f}")
print(f"Correctness on the test set: {knn.score(X_test, y_test):.2f}")

plt.show()
