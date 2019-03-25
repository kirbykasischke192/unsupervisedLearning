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

from sklearn.random_projection import GaussianRandomProjection
sat_rp = GaussianRandomProjection(n_components=2)
pen_rp = GaussianRandomProjection(n_components=2)


X_train_sat = sat_rp.fit_transform(X_train_sat_og)   
X_train_pen = pen_rp.fit_transform(X_train_pen_og)

print(X_train_sat.shape)
print(y_train_sat.shape)
print(y_train_pen.shape)

y_train_pen = [float(num) for num in y_train_pen]

#=====Change the iteration number in the title depending on what number your trial is================
# This is how I saw that the top two features changed with each iteration of randomized projection
plt.figure(figsize=(12, 6))  
plt.scatter(X_train_sat[:,0], X_train_sat[:,1], c=y_train_sat)
plt.title('Plot of Top Two Features - Random Projection 4th Iteration (Satellite)')  
plt.xlabel('Feature 1')  
plt.ylabel('Feature 2')
plt.show()
plt.figure(figsize=(12, 6))
plt.scatter(X_train_pen[:,0], X_train_pen[:,1], c=y_train_pen)
plt.title('Plot of Top Two Features - Random Projection 4th Iteration (Digits)')  
plt.xlabel('Feature 1')  
plt.ylabel('Feature 2')
plt.show()