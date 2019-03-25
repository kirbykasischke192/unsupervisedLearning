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
X_train_pen_og = pen_scaler.transform(X_train_pen)

sat_scaler.fit(X_train_sat)
X_train_sat_og = sat_scaler.transform(X_train_sat)

pen_scaler.fit(X_test_pen)
X_test_pen = pen_scaler.transform(X_test_pen)

sat_scaler.fit(X_test_sat) 
X_test_sat = sat_scaler.transform(X_test_sat)

sat_kurtosis_ica = []
pen_kurtosis_ica = []
sat_kurtosis_pca = []
pen_kurtosis_pca = []


from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from scipy.stats import kurtosis

# for i in range(1, 37):
sat_ica = FastICA(n_components=20)
sat_pca = PCA(n_components=20)
X_train_sat = sat_ica.fit_transform(X_train_sat_og)
sat_kurtosis_ica = kurtosis(X_train_sat)
X_train_sat = sat_pca.fit_transform(X_train_sat_og)
sat_kurtosis_pca = kurtosis(X_train_sat)

# for i in range(1,17):
pen_ica = FastICA(n_components=10)
pen_pca = PCA(n_components=10)
X_train_pen = pen_ica.fit_transform(X_train_pen_og)
pen_kurtosis_ica = kurtosis(X_train_pen)
X_train_pen = pen_pca.fit_transform(X_train_pen_og)
pen_kurtosis_pca = kurtosis(X_train_pen)

sat_kurtosis_ica[::-1].sort()
sat_kurtosis_pca[::-1].sort()
pen_kurtosis_ica[::-1].sort()
pen_kurtosis_pca[::-1].sort()

print(sat_kurtosis_ica)
print(sat_kurtosis_pca)
print(pen_kurtosis_ica)
print(pen_kurtosis_pca)

plt.figure(figsize=(12, 6))  
for i in range(20):
    plt.bar(2*i+1, sat_kurtosis_ica[i], color='red', tick_label=i+1)
for i in range(20):
    plt.bar(2*i + 2, sat_kurtosis_pca[i], color='green', tick_label=i+1)
plt.title('Kurtosis vs Component (20 Components) - ICA vs. PCA (Satellite)')  
plt.xlabel('Component (Red: ICA, Green: PCA)')  
plt.ylabel('Kurtosis')
plt.show()
plt.figure(figsize=(12, 6))  
for i in range(10):
    plt.bar(2*i+1, pen_kurtosis_ica[i], color='red', tick_label=i+1)
for i in range(10):
    plt.bar(2*i + 2, pen_kurtosis_pca[i], color='green', tick_label=i+1)
plt.title('Kurtosis vs Component (10 Components) - ICA vs. PCA (Digits)')  
plt.xlabel('Component (Red: ICA, Green: PCA)')  
plt.ylabel('Kurtosis')
plt.show()