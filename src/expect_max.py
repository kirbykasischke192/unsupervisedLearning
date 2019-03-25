#https://scikit-learn.org/stable/modules/mixture.html#selecting-the-number-of-components-in-a-classical-gaussian-mixture-model

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
X_test_pen_og = pen_scaler.transform(X_test_pen)

sat_scaler.fit(X_test_sat) 
X_test_sat_og = sat_scaler.transform(X_test_sat)

# use this params array to change the covariance-type parameter. Type covariance_type=params[i] in the paramsfor kmeans_sat and kmeans_pen
# and use a for loop from 0 to 4 exclusive. Also, use Plot A for this parameter
# params = ['full','tied','diag','spherical']

# use this params array to set the weight-concentration prior parameter. Type weight_concentration_prior=params[i] in the BayesianGaussianMixture constuctors and
# wrap it in a for loop from 0 to 20 exclusive
params = np.linspace(.05, 1, 20, endpoint=True)

# use this params array to set the max_iter parameter. Type max_iter=params[i] in the BayesianGaussianMixture constuctors and
# wrap it in a for loop from 0 to 5 exclusive
#params = [100,200,300,400,500,600,700,800]

pen_time = 0
sat_time = 0

pen_accuracy = []
sat_accuracy = []

sat_homog = []
pen_homog = []
sat_comp = []
sat_comp_train = []
sat_comp_test = []
pen_comp = []
pen_comp_train = []
pen_comp_test = []
sat_v = []
pen_v = []

# y_train_sat = [label - 1 for label in y_train_sat]
# y_test_sat = [label - 1 for label in y_test_sat]
#============================1. Uncomment this section for use with plots A, B, C, D =======================================================
for i in range(20):

    from sklearn.mixture import BayesianGaussianMixture # Chose BayesianGaussianMixture because it received better accuracy scores than just GaussianMixture
    bgm_sat = BayesianGaussianMixture(n_components = 7, covariance_type='tied', weight_concentration_prior=params[i], max_iter=500) # 7 categories, domain knowledge
    bgm_pen = BayesianGaussianMixture(n_components = 10, covariance_type='full', weight_concentration_prior=params[i], max_iter=500) # 10 categories, domain knowledge
    start_time = time.time()
    sat_labels_pred = bgm_sat.fit_predict(X_train_sat_og)
    #=======2. Use only for Plot C==================================
    # sat_labels_train = bgm_sat.fit_predict(X_train_sat_og)
    # sat_labels_test = bgm_sat.predict(X_test_sat_og)
    #=======================================================
    end_time = time.time()
    sat_time = end_time - start_time
    start_time = time.time()
    pen_labels_pred = bgm_pen.fit_predict(X_train_pen_og)
    #==========3. Use only for making Plot C======================
    # pen_labels_train = bgm_sat.fit_predict(X_train_pen_og)
    # pen_labels_test = bgm_sat.predict(X_test_pen_og)
    #=====================================================
    end_time = time.time()
    pen_time = end_time - start_time
#=====================================================================================================================================
#========================4. Uncomment this section to find homogeneity measure for train and test data (Use Plot C)====================
#     comp = metrics.homogeneity_score(sat_labels_train, y_train_sat)
#     sat_comp_train.append(comp)
#     comp = metrics.homogeneity_score(sat_labels_test, y_test_sat)
#     sat_comp_test.append(comp)
#     comp = metrics.homogeneity_score(pen_labels_train, y_train_pen)
#     pen_comp_train.append(comp)
#     comp = metrics.homogeneity_score(pen_labels_test, y_test_pen)
#     pen_comp_test.append(comp)
#===============================================================================================================================
#=========5. Uncomment this section when producing graph with homogeneity, completeness, and v-measure (Plot B)====================
    # homog, comp, v = metrics.homogeneity_completeness_v_measure(sat_labels_pred, y_train_sat)
    # sat_homog.append(homog)
    # sat_comp.append(comp)
    # sat_v.append(v)

    # homog, comp, v = metrics.homogeneity_completeness_v_measure(pen_labels_pred, y_train_pen)
    # pen_homog.append(homog)
    # pen_comp.append(comp)
    # pen_v.append(v)
#==============================================================================================================================
#====================6. Uncomment section to see accuracy scores (Use plot D)=====================================================
    pred_sat_labels = bgm_sat.predict(X_test_sat_og)
    pred_pen_labels = bgm_pen.predict(X_test_pen_og)

    from scipy.stats import mode
    # match true labels with the labels generated by the algorithm so we can compare accuracy
    sat_labels = np.zeros_like(pred_sat_labels)
    for i in range(1,8):
        mask = (pred_sat_labels == i)
        sat_labels[mask] = mode(y_test_sat[mask])[0]

    from sklearn.metrics import accuracy_score
    sat_accuracy.append(accuracy_score(y_test_sat, sat_labels))

    pen_labels = np.zeros_like(pred_pen_labels)
    for i in range(10):
        mask = (pred_pen_labels == i)
        pen_labels[mask] = mode(y_test_pen[mask])[0]

    pen_accuracy.append(accuracy_score(y_test_pen, pen_labels))

sat_error = [1 - accuracy for accuracy in sat_accuracy]
pen_error = [1 - accuracy for accuracy in pen_accuracy]
#===================================================================================================================================

#==========================Plot A: Use this plot for nominal params, like covariance-type============================================
# plt.figure(figsize=(12, 6))  
# for i in range(4):
#     plt.bar(params[i], sat_error[i])
# plt.title('Error Rate vs Covariance Type - Expectation Maximization(Satellite)')  
# plt.xlabel('Covariance Type')  
# plt.ylabel('Error')
# plt.show()
# plt.figure(figsize=(12, 6))
# for i in range(4):
#     plt.bar(params[i], pen_error[i])
# plt.title('Error Rate vs Covariance Type - Expectation Maximization (Digits)')  
# plt.xlabel('Covariance Type')  
# plt.ylabel('Error')
# plt.show()
#======================================================================================================================================
#======================Plot B: Use this plot for the homogeneity, completeness and v measure graphs=====================================
# plt.figure(figsize=(12, 6))  
# plt.plot(range(1, 21), sat_comp, label='Completeness', color='red', linestyle='solid', marker='')
# plt.plot(range(1, 21), sat_homog, label='Homogeneity', color='green', linestyle='solid', marker='')
# plt.plot(range(1, 21), sat_v, label='V-Measure', color='blue', linestyle='solid', marker='')
# plt.title('Homogeneity, Completeness, and V Measure vs. Number of Components - Expectation Maximization (Satellite)')  
# plt.xlabel('Number of Components')  
# plt.ylabel('Score')
# plt.legend()
# plt.show()
# plt.figure(figsize=(12, 6))
# plt.plot(range(1, 21), pen_comp, label='Completeness', color='red', linestyle='solid', marker='')
# plt.plot(range(1, 21), pen_homog, label='Homogeneity', color='green', linestyle='solid', marker='')
# plt.plot(range(1, 21), pen_v, label='V-Measure', color='blue', linestyle='solid', marker='')
# plt.title('Homogeneity, Completeness, and V Measure vs. Number of Components - Expectation Maximization (Digits)')  
# plt.xlabel('Number of Components')  
# plt.ylabel('Score')
# plt.legend()
# plt.show()
#============================================================================================================================================
#===========================Plot C: Showing homogeneity for test and train data ============================================================
# plt.figure(figsize=(12, 6))  
# plt.plot(range(1, 21), sat_comp_train, label='Train', color='red', linestyle='solid', marker='')
# plt.plot(range(1, 21), sat_comp_test, label='Test', color='green', linestyle='solid', marker='')
# plt.title('Homogeneity (Test and Train) vs. Number of Components - Expectation Maximization (Satellite)')  
# plt.xlabel('Number of Components')  
# plt.ylabel('Homogeneity')
# plt.legend()
# plt.show()
# plt.figure(figsize=(12, 6))
# plt.plot(range(1, 21), pen_comp_train, label='Train', color='red', linestyle='solid', marker='')
# plt.plot(range(1, 21), pen_comp_test, label='Test', color='green', linestyle='solid', marker='')
# plt.title('Homogeneity (Test and Train) vs. Number of Components - Expectations Maximization (Digits)')  
# plt.xlabel('Number of Components')  
# plt.ylabel('Homogeneity')
# plt.legend()
# plt.show()
#==============================================================================================================================================
#===========================Plot D: Showing error for satellite and digits data ============================================================
plt.figure(figsize=(12, 6))  
plt.plot(params, sat_error, label='Train', color='red', linestyle='solid', marker='')
plt.title('Error vs. Weight Concentration Prior  - Expectation Maximization (Satellite)')  
plt.xlabel('Weight Concentration Prior')  
plt.ylabel('Error')
plt.legend()
plt.show()
plt.figure(figsize=(12, 6))
plt.plot(params, pen_error, label='Train', color='red', linestyle='solid', marker='')
plt.title('Error vs. Weight Concentration Prior - Expectation Maximization (Digits)')  
plt.xlabel('Weight Concentration Prior')  
plt.ylabel('Error')
plt.legend()
plt.show()
#==============================================================================================================================================
#================7. Uncomment this section to see how the different feature selection algorithms effect the homogeneity of EM components===============
# algs = ['PCA','ICA','Random Projections','Feature Agglomeration']

# sat_homog = []
# pen_homog = []

# from sklearn.decomposition import PCA
# sat_pca = PCA(n_components=20)
# pen_pca = PCA(n_components=10)  
# X_train_sat = sat_pca.fit_transform(X_train_sat_og)   
# X_train_pen = pen_pca.fit_transform(X_train_pen_og)

# from sklearn.mixture import BayesianGaussianMixture
# bgm_sat = BayesianGaussianMixture(n_components = 8, covariance_type='tied', weight_concentration_prior=0.55, max_iter=500)
# bgm_pen = BayesianGaussianMixture(n_components = 11, covariance_type='full', weight_concentration_prior=0.15, max_iter=500)
# sat_labels = bgm_sat.fit_predict(X_train_sat)
# pen_labels = bgm_pen.fit_predict(X_train_pen)
# sat_homog.append(metrics.homogeneity_score(sat_labels, y_train_sat))
# pen_homog.append(metrics.homogeneity_score(pen_labels, y_train_pen))

# from sklearn.decomposition import FastICA
# sat_ica = FastICA(n_components=20)
# pen_ica = FastICA(n_components=10)  
# X_train_sat = sat_ica.fit_transform(X_train_sat_og)   
# X_train_pen = pen_ica.fit_transform(X_train_pen_og)

# from sklearn.mixture import BayesianGaussianMixture
# bgm_sat = BayesianGaussianMixture(n_components = 8, covariance_type='tied', weight_concentration_prior=0.55, max_iter=500)
# bgm_pen = BayesianGaussianMixture(n_components = 11, covariance_type='full', weight_concentration_prior=0.15, max_iter=500)
# sat_labels = bgm_sat.fit_predict(X_train_sat)
# pen_labels = bgm_pen.fit_predict(X_train_pen)
# sat_homog.append(metrics.homogeneity_score(sat_labels, y_train_sat))
# pen_homog.append(metrics.homogeneity_score(pen_labels, y_train_pen))

# from sklearn.random_projection import GaussianRandomProjection
# sat_rp = GaussianRandomProjection(n_components=20)
# pen_rp = GaussianRandomProjection(n_components=10)  
# X_train_sat = sat_rp.fit_transform(X_train_sat_og)   
# X_train_pen = pen_rp.fit_transform(X_train_pen_og)

# from sklearn.mixture import BayesianGaussianMixture
# bgm_sat = BayesianGaussianMixture(n_components = 8, covariance_type='tied', weight_concentration_prior=0.55, max_iter=500)
# bgm_pen = BayesianGaussianMixture(n_components = 11, covariance_type='full', weight_concentration_prior=0.15, max_iter=500)
# sat_labels = bgm_sat.fit_predict(X_train_sat)
# pen_labels = bgm_pen.fit_predict(X_train_pen)
# sat_homog.append(metrics.homogeneity_score(sat_labels, y_train_sat))
# pen_homog.append(metrics.homogeneity_score(pen_labels, y_train_pen))

# from sklearn.cluster import FeatureAgglomeration
# sat_fa = FeatureAgglomeration(n_clusters=20)
# pen_fa = FeatureAgglomeration(n_clusters=10)  
# X_train_sat = sat_fa.fit_transform(X_train_sat_og)   
# X_train_pen = pen_fa.fit_transform(X_train_pen_og)

# from sklearn.mixture import BayesianGaussianMixture
# bgm_sat = BayesianGaussianMixture(n_components = 8, covariance_type='tied', weight_concentration_prior=0.55, max_iter=500)
# bgm_pen = BayesianGaussianMixture(n_components = 11, covariance_type='full', weight_concentration_prior=0.15, max_iter=500)
# sat_labels = bgm_sat.fit_predict(X_train_sat)
# pen_labels = bgm_pen.fit_predict(X_train_pen)
# sat_homog.append(metrics.homogeneity_score(sat_labels, y_train_sat))
# pen_homog.append(metrics.homogeneity_score(pen_labels, y_train_pen))


# plt.figure(figsize=(12, 6))  
# plt.bar(algs, sat_homog)
# plt.title('Homogeneity vs. Dimensionality Reduction Algorithm - Expectation Maximization (Satellite)')  
# plt.xlabel('Dimensionality Reduction Algorithm')  
# plt.ylabel('Homogeneity')
# plt.legend()
# plt.show()
# plt.figure(figsize=(12, 6)) 
# plt.bar(algs, pen_homog)
# plt.title('Homogeneity vs. Dimensionality Reduction Algorithm - Expectation Maximization (Digits)')  
# plt.xlabel('Dimensionality Reduction Algorithm')  
# plt.ylabel('Homogeneity')
# plt.legend()
# plt.show()
#==================================================================================================================================================================

# ==============================8. Uncomment this section to test how the number of components in random projection effects the homgeneity =========================
# sat_homog_alg = []
# from sklearn.random_projection import GaussianRandomProjection
# from sklearn.mixture import BayesianGaussianMixture
# bgm_sat = BayesianGaussianMixture(n_components = 7, max_iter=500)
# for i in range(2,37):
#     sat_rp = GaussianRandomProjection(n_components=i)
#     sat_labels = bgm_sat.fit_predict(X_train_sat_og)
#     sat_homog.append(metrics.homogeneity_score(sat_labels, y_train_sat))
    
#     X_train_sat = sat_rp.fit_transform(X_train_sat_og)
#     sat_labels = bgm_sat.fit_predict(X_train_sat)
#     sat_homog_alg.append(metrics.homogeneity_score(sat_labels, y_train_sat))

# pen_homog_alg = [] 
# bgm_pen = BayesianGaussianMixture(n_components = 11, max_iter=500)
# for i in range(2, 17):  
#     pen_rp = GaussianRandomProjection(n_components=i)  
#     pen_labels = bgm_pen.fit_predict(X_train_pen_og)
#     pen_homog.append(metrics.homogeneity_score(pen_labels, y_train_pen))

#     X_train_pen = pen_rp.fit_transform(X_train_pen_og)
#     pen_labels = bgm_pen.fit_predict(X_train_pen)
#     pen_homog_alg.append(metrics.homogeneity_score(pen_labels, y_train_pen))

# plt.figure(figsize=(12, 6))  
# plt.plot(range(2, 37), sat_homog, label='No Random Projections', color='red', linestyle='solid', marker='')
# plt.plot(range(2, 37), sat_homog_alg, label='w/ Random Projections', color='green', linestyle='solid', marker='')
# plt.title('Homogeneity vs. Number of Features - Expectation Maximization (Satellite)')  
# plt.xlabel('Number of Features')  
# plt.ylabel('Homogeneity')
# plt.legend()
# plt.show()
# plt.figure(figsize=(12, 6)) 
# plt.plot(range(2, 17), pen_homog, label='No Random Projection', color='red', linestyle='solid', marker='')
# plt.plot(range(2, 17), pen_homog_alg, label='w/ Random Projection', color='green', linestyle='solid', marker='')
# plt.title('Homogeneity vs. Number of Features - Expectation Maximization (Digits)')  
# plt.xlabel('Number of Features')  
# plt.ylabel('Homogeneity')
# plt.legend()
# plt.show()
#==============================================================================================================================================================
# print(sat_score)
# print(pen_score)