import pandas as pd
import csv
import matplotlib.pyplot as plt

data = pd.read_csv('NISM.csv')
print(data.groupby('label').size())

print(data.columns[:-1])
X = data.loc[:, data.columns[:-1]]
y = data.label
X = X.values
print(X.shape)

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing

clf = ExtraTreesClassifier()
clf = clf.fit(X, y)
feature = data.columns[:-1]
importance = clf.feature_importances_
print(importance)
print(sorted(zip(importance, data.columns), reverse=True))
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)
X = X_new
print(X_new)
print(X.shape)

from sklearn import linear_model, ensemble