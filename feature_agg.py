# https://turi.com/learn/userguide/feature-engineering/random_projection.html

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
import matplotlib.colors as color
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

y_train_pen = [float(num) for num in y_train_pen]
sat_params = [2,10,18]
pen_params = [2,8,14]

from sklearn.cluster import FeatureAgglomeration
for i in range(3):
    sat_fa = FeatureAgglomeration(n_clusters=sat_params[i])
    pen_fa = FeatureAgglomeration(n_clusters=pen_params[i])


    X_train_sat = sat_fa.fit_transform(X_train_sat_og)   
    X_train_pen = pen_fa.fit_transform(X_train_pen_og)

    plt.figure(figsize=(12, 6))  
    plt.scatter(X_train_sat[:,0], X_train_sat[:,1], c=y_train_sat)
    plt.title('Plot of Top Two Features - Feature Agglomeration of '+str(sat_params[i])+' Features (Satellite)')  
    plt.xlabel('Feature 1')  
    plt.ylabel('Feature 2')
    plt.show()
    plt.figure(figsize=(12, 6))
    plt.scatter(X_train_pen[:,0], X_train_pen[:,1], c=y_train_pen)
    plt.title('Plot of Top Two Features - Feature Agglomeration of '+str(pen_params[i])+' Features (Digits)')  
    plt.xlabel('Feature 1')  
    plt.ylabel('Feature 2')
    plt.show()



    