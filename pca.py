# https://stackabuse.com/implementing-pca-in-python-with-scikit-learn/

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
from sklearn import metrics  
import time as time

satdata = pd.read_csv("./../data/sat_data.csv")
pendata = pd.read_csv("./../data/pen_data.csv")

Xsat = satdata.iloc[:, :-1].values
ysat = satdata.iloc[:, 36].values

Xpen = pendata.iloc[:, :-1].values
ypen = pendata.iloc[:, 16].values

from sklearn.model_selection import train_test_split  
X_train_sat, X_test_sat, y_train_sat, y_test_sat = train_test_split(Xsat, ysat, test_size=0.20)
X_train_pen, X_test_pen, y_train_pen, y_test_pen = train_test_split(Xpen, ypen, test_size=0.20)

from sklearn.preprocessing import StandardScaler  
pen_scaler = StandardScaler()
sat_scaler = StandardScaler()

#We normalize the values so the kmeans doesn't gravitate towards very high values
pen_scaler.fit(X_train_pen)
X_train_pen = pen_scaler.transform(X_train_pen)

sat_scaler.fit(X_train_sat)
X_train_sat = sat_scaler.transform(X_train_sat)

pen_scaler.fit(X_test_pen)
X_test_pen = pen_scaler.transform(X_test_pen)

sat_scaler.fit(X_test_sat) 
X_test_sat = sat_scaler.transform(X_test_sat)

from sklearn.decomposition import PCA
sat_pca = PCA()
pen_pca = PCA()  
X_train_sat = sat_pca.fit_transform(X_train_sat)  
X_test_sat = sat_pca.fit_transform(X_test_sat) 
X_train_pen = pen_pca.fit_transform(X_train_pen)
X_test_pen = pen_pca.fit_transform(X_test_pen)

sat_explained_variance = sat_pca.explained_variance_ratio_
pen_explained_variance = pen_pca.explained_variance_ratio_

plt.figure(figsize=(12, 6))  
for i in range(36):
    plt.bar(i + 1, sat_explained_variance[i])
plt.title('Variance Ratio vs Principle Component - PCA (Satellite)')  
plt.xlabel('Principle Component')  
plt.ylabel('Variance Ratio')
plt.show()
plt.figure(figsize=(12, 6))
for i in range(16):
    plt.bar(i + 1, pen_explained_variance[i])
plt.title('Variance Ratio vs Principle Component - PCA (Digits)')  
plt.xlabel('Principle Component')  
plt.ylabel('Variance Ratio')
plt.show()

