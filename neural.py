import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
from sklearn import metrics  
import time as time
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture

pendata = pd.read_csv("./../data/pen_data.csv")

Xpen = pendata.iloc[:, :-1].values
ypen = pendata.iloc[:, 16].values

from sklearn.model_selection import train_test_split  
X_train_pen, X_test_pen, y_train_pen, y_test_pen = train_test_split(Xpen, ypen, test_size=0.20)

from sklearn.preprocessing import StandardScaler  
pen_scaler = StandardScaler()

#We normalize the values so the kmeans doesn't gravitate towards very high values
X_train_pen_og = pen_scaler.fit_transform(X_train_pen)
X_test_pen_og = pen_scaler.fit_transform(X_test_pen)
#==============1. Uncomment this section to view neural net error and fitting time vs. feature selection algorithm==================================
# algs = ['No Algorithm','PCA','ICA','Random Projections','Feature Agglomeration']

# pen_error = []
# run_time = []

# #====================No Alg================================
# pen_classifier = MLPClassifier(hidden_layer_sizes=(110,110), max_iter=1000, activation='tanh')
# start = time.time()
# pen_classifier.fit(X_train_pen_og, y_train_pen)
# end = time.time()
# run_time.append(end - start)
# pen_pred = pen_classifier.predict(X_test_pen_og)
# pen_error.append(1 - metrics.accuracy_score(pen_pred, y_test_pen))
# #==========================================================
# #========================PCA===============================
# X_pen_scaled = pen_scaler.fit_transform(Xpen)

# from sklearn.decomposition import PCA
# pen_pca = PCA(n_components=10)    
# X_pen_scaled = pen_pca.fit_transform(X_pen_scaled)

# X_train_pen, X_test_pen, y_train_pen, y_test_pen = train_test_split(X_pen_scaled, ypen, test_size=0.20)

# start = time.time()
# pen_classifier.fit(X_train_pen, y_train_pen)
# end = time.time()
# run_time.append(end - start)
# pen_pred = pen_classifier.predict(X_test_pen)
# pen_error.append(1 - metrics.accuracy_score(pen_pred, y_test_pen))
# #===========================================================
# #===========================ICA=============================
# X_pen_scaled = pen_scaler.fit_transform(Xpen)

# from sklearn.decomposition import FastICA
# pen_ica = FastICA(n_components=10)     
# X_pen_scaled = pen_ica.fit_transform(X_pen_scaled)

# X_train_pen, X_test_pen, y_train_pen, y_test_pen = train_test_split(X_pen_scaled, ypen, test_size=0.20)

# start = time.time()
# pen_classifier.fit(X_train_pen, y_train_pen)
# end = time.time()
# run_time.append(end - start)
# pen_pred = pen_classifier.predict(X_test_pen)
# pen_error.append(1 - metrics.accuracy_score(pen_pred, y_test_pen))
# #===========================================================
# #=======================Random Projection===================
# X_pen_scaled = pen_scaler.fit_transform(Xpen)

# from sklearn.random_projection import GaussianRandomProjection
# pen_rp = GaussianRandomProjection(n_components=10)     
# X_pen_scaled = pen_rp.fit_transform(X_pen_scaled)

# X_train_pen, X_test_pen, y_train_pen, y_test_pen = train_test_split(X_pen_scaled, ypen, test_size=0.20)

# start = time.time()
# pen_classifier.fit(X_train_pen, y_train_pen)
# end = time.time()
# run_time.append(end - start)
# pen_pred = pen_classifier.predict(X_test_pen)
# pen_error.append(1 - metrics.accuracy_score(pen_pred, y_test_pen))
# #===========================================================
# #====================Feature Agglomeration==================
# X_pen_scaled = pen_scaler.fit_transform(Xpen)

# from sklearn.cluster import FeatureAgglomeration
# pen_fa = FeatureAgglomeration(n_clusters=10)    
# X_pen_scaled = pen_fa.fit_transform(X_pen_scaled)

# X_train_pen, X_test_pen, y_train_pen, y_test_pen = train_test_split(X_pen_scaled, ypen, test_size=0.20)

# start = time.time()
# pen_classifier.fit(X_train_pen, y_train_pen)
# end = time.time()
# run_time.append(end - start)
# pen_pred = pen_classifier.predict(X_test_pen)
# pen_error.append(1 - metrics.accuracy_score(pen_pred, y_test_pen))
# #===========================================================

# plt.figure(figsize=(12, 6)) 
# plt.bar(algs, pen_error)
# plt.title('Test Error vs. Dimensionality Reduction Algorithm - Neural Net (Digits)')  
# plt.xlabel('Dimensionality Reduction Algorithm')  
# plt.ylabel('Test Error')
# plt.show()
# plt.figure(figsize=(12, 6)) 
# plt.bar(algs, run_time)
# plt.title('Fitting Time vs. Dimensionality Reduction Algorithm - Neural Net (Digits)')  
# plt.xlabel('Dimensionality Reduction Algorithm')  
# plt.ylabel('Fitting Time')
# plt.show()
#==========================================================================================================================================
# #=====================2. Uncomment this section to graph the neural net testing error for each feature selection algorithm=======================
# algs = ['No Algorithm','PCA','ICA','Random Projections','Feature Agglomeration']

# pen_error = []
# pen_error_ica = []
# pen_error_pca = []
# pen_error_rp = []
# pen_error_fa = []
# run_time = []

# #====================No Alg================================
# pen_classifier = MLPClassifier(hidden_layer_sizes=(110,110), max_iter=1000, activation='tanh')
# start = time.time()
# pen_classifier.fit(X_train_pen_og, y_train_pen)
# end = time.time()
# run_time.append(end - start)
# pen_pred = pen_classifier.predict(X_test_pen_og)
# error = 1 - metrics.accuracy_score(pen_pred, y_test_pen)
# pen_error = [error for i in range(16)]
# #==========================================================
# #========================PCA===============================
# from sklearn.decomposition import PCA
# for i in range(1, 17):
#     X_pen_scaled = pen_scaler.fit_transform(Xpen)

#     pen_pca(n_components=i)    
#     X_pen_scaled = pen_pca.fit_transform(X_pen_scaled)

#     X_train_pen, X_test_pen, y_train_pen, y_test_pen = train_test_split(X_pen_scaled, ypen, test_size=0.20)

#     pen_classifier.fit(X_train_pen, y_train_pen)
#     pen_pred = pen_classifier.predict(X_test_pen)
#     pen_error_pca.append(1 - metrics.accuracy_score(pen_pred, y_test_pen))
# #===========================================================
# #===========================ICA=============================
# from sklearn.decomposition import FastICA
# for i in range(1, 17):
#     X_pen_scaled = pen_scaler.fit_transform(Xpen)

#     pen_ica = FastICA(n_components=i)     
#     X_pen_scaled = pen_ica.fit_transform(X_pen_scaled)

#     X_train_pen, X_test_pen, y_train_pen, y_test_pen = train_test_split(X_pen_scaled, ypen, test_size=0.20)

#     pen_classifier.fit(X_train_pen, y_train_pen)
#     pen_pred = pen_classifier.predict(X_test_pen)
#     pen_error_ica.append(1 - metrics.accuracy_score(pen_pred, y_test_pen))
# #===========================================================
# #=======================Random Projection===================
# from sklearn.random_projection import GaussianRandomProjection
# for i in range(1, 17):
#     X_pen_scaled = pen_scaler.fit_transform(Xpen)

#     pen_rp = GaussianRandomProjection(n_components=i)     
#     X_pen_scaled = pen_rp.fit_transform(X_pen_scaled)

#     X_train_pen, X_test_pen, y_train_pen, y_test_pen = train_test_split(X_pen_scaled, ypen, test_size=0.20)

#     pen_classifier.fit(X_train_pen, y_train_pen)
#     pen_pred = pen_classifier.predict(X_test_pen)
#     pen_error_rp.append(1 - metrics.accuracy_score(pen_pred, y_test_pen))
# #===========================================================
# #====================Feature Agglomeration==================
# from sklearn.cluster import FeatureAgglomeration
# for i in range(1, 17):
#     X_pen_scaled = pen_scaler.fit_transform(Xpen)

#     pen_fa = FeatureAgglomeration(n_clusters=i)    
#     X_pen_scaled = pen_fa.fit_transform(X_pen_scaled)

#     X_train_pen, X_test_pen, y_train_pen, y_test_pen = train_test_split(X_pen_scaled, ypen, test_size=0.20)

#     pen_classifier.fit(X_train_pen, y_train_pen)
#     pen_pred = pen_classifier.predict(X_test_pen)
#     pen_error_fa.append(1 - metrics.accuracy_score(pen_pred, y_test_pen))
# #===========================================================

# plt.figure(figsize=(12, 6)) 
# plt.plot(range(1, 17), pen_error, label='No PCA', color='red', linestyle='solid', marker='')
# plt.plot(range(1, 17), pen_error_pca, label='w/ PCA', color='green', linestyle='solid', marker='')
# plt.plot(range(1, 17), pen_error_ica, label='w/ ICA', color='blue', linestyle='solid', marker='')
# plt.plot(range(1, 17), pen_error_rp, label='w/ Random Projection', color='yellow', linestyle='solid', marker='')
# plt.plot(range(1, 17), pen_error_fa, label='w/ Feature Agglomeration', color='brown', linestyle='solid', marker='')
# plt.title('Test Error vs. Number of Components - Neural Net (Digits)')  
# plt.xlabel('Number of Components')  
# plt.ylabel('Test Error')
# plt.legend()
# plt.show()
# #=============================================================================================================================
#===================3. Uncomment this section to view how the clustering algorithms impact neural net test error=============================
algs = ['No Algorithm','KMeans','EM']

pen_error = []
pen_error_kmean = []
pen_error_em = []

#====================No Alg================================
pen_classifier = MLPClassifier(hidden_layer_sizes=(50,50), max_iter=500, activation='tanh')
pen_classifier.fit(X_train_pen_og, y_train_pen)
pen_pred = pen_classifier.predict(X_test_pen_og)
error = 1 - metrics.accuracy_score(pen_pred, y_test_pen)
pen_error = [error for i in range(30)]
#==========================================================
#========================KMeans===============================
from sklearn.decomposition import PCA
for i in range(1, 31):
    X_pen_scaled = pen_scaler.fit_transform(Xpen)

    pen_kmeans = KMeans(n_clusters=i)    
    X_pen_scaled = pen_kmeans.fit_predict(X_pen_scaled)
    X_pen_scaled = X_pen_scaled.reshape(-1, 1)

    X_train_pen, X_test_pen, y_train_pen, y_test_pen = train_test_split(X_pen_scaled, ypen, test_size=0.20)

    pen_classifier.fit(X_train_pen, y_train_pen)
    pen_pred = pen_classifier.predict(X_test_pen)
    pen_error_kmean.append(1 - metrics.accuracy_score(pen_pred, y_test_pen))
#===========================================================
#===========================EM=============================
from sklearn.decomposition import FastICA
for i in range(1, 31):
    X_pen_scaled = pen_scaler.fit_transform(Xpen)

    pen_bgm = BayesianGaussianMixture(n_components=i)     
    X_pen_scaled = pen_bgm.fit_predict(X_pen_scaled)
    X_pen_scaled = X_pen_scaled.reshape(-1, 1)

    X_train_pen, X_test_pen, y_train_pen, y_test_pen = train_test_split(X_pen_scaled, ypen, test_size=0.20)

    pen_classifier.fit(X_train_pen, y_train_pen)
    pen_pred = pen_classifier.predict(X_test_pen)
    pen_error_em.append(1 - metrics.accuracy_score(pen_pred, y_test_pen))
#===========================================================

plt.figure(figsize=(12, 6)) 
plt.plot(range(1, 31), pen_error, label='No Clustering', color='red', linestyle='solid', marker='')
plt.plot(range(1, 31), pen_error_kmean, label='w/ KMeans', color='green', linestyle='solid', marker='')
plt.plot(range(1, 31), pen_error_em, label='w/ EM', color='blue', linestyle='solid', marker='')
plt.title('Test Error vs. Number of Clusters - Neural Net (Digits)')  
plt.xlabel('Number of Clusters')  
plt.ylabel('Test Error')
plt.legend()
plt.show()
#==================================================================================================================================================