# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Created on Sun Jul 12 10:27:24 2015

Supervised learning for the Hillenbrand vowel data.
Algorithms implemented: 

Perceptron
kNN
SVM
Random forest
AdABoost

targets: vowel class

predictors: 
formant values (steady-state, percentages of steady-state), formant ratios,
formants and formant ratios reduced to first two principal components

"""

import os

source_data = '~/Documents/python_scripts/hillenbrand-data-setup.py'

%run os.path.expanduser(source_data)
#------------------------------------------------------------------------------ 
# PCA to reduce

from sklearn import decomposition

pca = decomposition.PCA(n_components = 2)

formant_reduced = pca.fit_transform(formant_mtx)
formant_ratio_reduced = pca.fit_transform(formant_ratio_mtx)


#------------------------------------------------------------------------------ 
# data split into train and test sets

# 80/20 data split

# randomize (could also use built-in functions)
n_samples = len(formant_mtx)

np.random.seed(3475)

indices = np.random.permutation(len(formant_reduced))

# test and train data for PCA reduction
formant_reduced_train_data = formant_reduced[indices[:0.8 * n_samples]]
formant_reduced_test_data = formant_reduced[indices[-0.2 * n_samples:]]

formant_ratio_reduced_train_data = \
  formant_ratio_reduced[indices[:0.8 * n_samples]]
		
formant_ratio_reduced_test_data = \
  formant_ratio_reduced[indices[-0.2 * n_samples:]]
		
# test and train data for non-PCA data
formant_train_data = formant_mtx[indices[:0.8 * n_samples]]
formant_test_data = formant_mtx[indices[-0.2 * n_samples:]]

formant_ratio_train_data = \
  formant_ratio_mtx[indices[:0.8 * n_samples]]

formant_ratio_test_data = \
  formant_ratio_mtx[indices[-0.2 * n_samples:]]

# targets
train_target = target_mtx[indices[:0.8 * n_samples]]
test_target = target_mtx[indices[-0.2 * n_samples:]]


#------------------------------------------------------------------------------ 
# perceptron

from sklearn.linear_model import Perceptron
# from sklearn.grid_search import GridSearchCV

# alpha values to search over
# alphas = np.logspace(-5, 3)

# parameters = {'clf__alpha': alphas}

percept = \
  Perceptron(penalty = 'l2', fit_intercept = False, random_state = 23094)

# gs_percept = GridSearchCV(percept, parameters, n_jobs = -1)

percept_predicted_formant_ratio_reduced = \
  percept.fit(formant_ratio_reduced_train_data, train_target[:, 2]).predict(formant_ratio_reduced_test_data)

percept_predicted_formant_reduced = \
  percept.fit(formant_reduced_train_data, train_target[:, 2]).predict(formant_reduced_test_data)

percept_predicted_formant_ratio = \
  percept.fit(formant_ratio_train_data, train_target[:, 2]).predict(formant_ratio_test_data)
		
percept_predicted_formant = \
  percept.fit(formant_train_data, train_target[:, 2]).predict(formant_test_data)

#------------------------------------------------------------------------------ 
# k nearest neighbors

from sklearn import neighbors

knn = neighbors.KNeighborsClassifier()

knn_predicted_formant_ratio_reduced = \
  knn.fit(formant_ratio_reduced_train_data, train_target[:, 2]).predict(formant_ratio_reduced_test_data)

knn_predicted_formant_reduced = \
  knn.fit(formant_reduced_train_data, train_target[:, 2]).predict(formant_reduced_test_data)

knn_predicted_formant_ratio = \
  knn.fit(formant_ratio_train_data, train_target[:, 2]).predict(formant_ratio_test_data)

knn_predicted_formant = \
  knn.fit(formant_train_data, train_target[:, 2]).predict(formant_test_data)

#------------------------------------------------------------------------------ 
# support vector machine

from sklearn import svm

svc = svm.SVC(kernel = 'rbf') # polynomial might be bteer

svc_predicted_formant_ratio_reduced = \
  svc.fit(formant_ratio_reduced_train_data, train_target[:, 2]).predict(formant_ratio_reduced_test_data)

svc_predicted_formant_reduced = \
  svc.fit(formant_reduced_train_data, train_target[:, 2]).predict(formant_reduced_test_data)

svc_predicted_formant_ratio = \
  knn.fit(formant_ratio_train_data, train_target[:, 2]).predict(formant_ratio_test_data)

svc_predicted_formant = \
  knn.fit(formant_train_data, train_target[:, 2]).predict(formant_test_data)


#------------------------------------------------------------------------------ 
# random forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf_predicted_formant_ratio_reduced = \
  rf.fit(formant_ratio_reduced_train_data, train_target[:, 2]).predict(formant_ratio_reduced_test_data)

rf_predicted_formant_reduced = \
  rf.fit(formant_reduced_train_data, train_target[:, 2]).predict(formant_reduced_test_data)

rf_predicted_formant_ratio = \
  rf.fit(formant_ratio_train_data, train_target[:, 2]).predict(formant_ratio_test_data)

rf_predicted_formant = \
  rf.fit(formant_train_data, train_target[:, 2]).predict(formant_test_data)
				

#------------------------------------------------------------------------------ 
# adapative boosting
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(learning_rate = 0.9, random_state = 23094)

ada_predicted_formant_ratio_reduced = \
  ada.fit(formant_ratio_reduced_train_data, train_target[:, 2]).predict(formant_ratio_reduced_test_data)

ada_predicted_formant_reduced = \
  ada.fit(formant_reduced_train_data, train_target[:, 2]).predict(formant_reduced_test_data)

ada_predicted_formant_ratio = \
  ada.fit(formant_ratio_train_data, train_target[:, 2]).predict(formant_ratio_test_data)

ada_predicted_formant = \
  ada.fit(formant_train_data, train_target[:, 2]).predict(formant_test_data)
				

#------------------------------------------------------------------------------ 
# performance metrics

from sklearn.metrics import classification_report, confusion_matrix


# perceptron
print("Classification report for classifier %s:\n%s\n"
      % (percept, classification_report(test_target[:, 2], \
						percept_predicted_formant_ratio_reduced)))
												
print("Confusion matrix:\n%s" % \
  confusion_matrix(test_target[:, 2], percept_predicted_formant_ratio_reduced))

		
print("Classification report for classifier %s:\n%s\n"
      % (percept, classification_report(test_target[:, 2], \
						percept_predicted_formant_reduced)))
												
print("Confusion matrix:\n%s" % \
  confusion_matrix(test_target[:, 2], percept_predicted_formant_reduced))


print("Classification report for classifier %s:\n%s\n"
      % (percept, classification_report(test_target[:, 2], \
                        percept_predicted_formant_ratio)))
                                                
print("Confusion matrix:\n%s" % \
  confusion_matrix(test_target[:, 2], percept_predicted_formant_ratio))

        
print("Classification report for classifier %s:\n%s\n"
      % (percept, classification_report(test_target[:, 2], \
                        percept_predicted_formant)))
                                                
print("Confusion matrix:\n%s" % \
  confusion_matrix(test_target[:, 2], percept_predicted_formant))


# kNN
print("Classification report for classifier %s:\n%s\n"
      % (knn, classification_report(test_target[:, 2], \
                        knn_predicted_formant_ratio_reduced)))
                                                
print("Confusion matrix:\n%s" % \
  confusion_matrix(test_target[:, 2], knn_predicted_formant_ratio_reduced))

        
print("Classification report for classifier %s:\n%s\n"
      % (knn, classification_report(test_target[:, 2], \
                        knn_predicted_formant_reduced)))
                                                
print("Confusion matrix:\n%s" % \
  confusion_matrix(test_target[:, 2], knn_predicted_formant_reduced))

# SVM
print("Classification report for classifier %s:\n%s\n"
      % (svc, classification_report(test_target[:, 2], \
                        svc_predicted_formant_ratio_reduced)))
                                                
print("Confusion matrix:\n%s" % \
  confusion_matrix(test_target[:, 2], svc_predicted_formant_ratio_reduced))

        
print("Classification report for classifier %s:\n%s\n"
      % (svc, classification_report(test_target[:, 2], \
                        svc_predicted_formant_reduced)))
                                                
print("Confusion matrix:\n%s" % \
  confusion_matrix(test_target[:, 2], svc_predicted_formant_reduced))


# Random forest
print("Classification report for classifier %s:\n%s\n"
      % (rf, classification_report(test_target[:, 2], \
                        rf_predicted_formant_ratio_reduced)))
                                                
print("Confusion matrix:\n%s" % \
  confusion_matrix(test_target[:, 2], rf_predicted_formant_ratio_reduced))

        
print("Classification report for classifier %s:\n%s\n"
      % (rf, classification_report(test_target[:, 2], \
                        rf_predicted_formant_reduced)))
                                                
print("Confusion matrix:\n%s" % \
  confusion_matrix(test_target[:, 2], rf_predicted_formant_reduced))

# Ada Boost

print("Classification report for classifier %s:\n%s\n"
      % (ada, classification_report(test_target[:, 2], \
                        ada_predicted_formant_ratio_reduced)))
                                                
print("Confusion matrix:\n%s" % \
  confusion_matrix(test_target[:, 2], ada_predicted_formant_ratio_reduced))

        
print("Classification report for classifier %s:\n%s\n"
      % (ada, classification_report(test_target[:, 2], \
                        ada_predicted_formant_reduced)))
                                                
print("Confusion matrix:\n%s" % \
  confusion_matrix(test_target[:, 2], ada_predicted_formant_reduced))


#------------------------------------------------------------------------------ 
# plotting results (vowel only)
# order: decision boundary, train data, test data


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# need 12 colors
# taken from colorbrewer2

cmap_light = \
  ListedColormap(["#A6CEE3", "#1F78B4", "#B2DF8A", "#33A02C","#FB9A99",\
		"#E31A1C", "#FDBF6F", "#FF7F00", "#CAB2D6" ,"#6A3D9A", "#FFFF99",\
		"#B15928"])

cmap_bold = \
  ListedColormap(["#8DD3C7", "#FFFFB3", "#BEBADA", "#FB8072", "#80B1D3",\
		"#FDB462", "#B3DE69", "#FCCDE5", "#D9D9D9", "#BC80BD", "#CCEBC5",\
		"#FFED6F"])

# k nearest neighbors
# fit model, plot decision boundary

# perceptron
knn = neighbors.KNeighborsClassifier()

knn.fit(train_data, train_target[:, 2])

h = 0.01
x_min, x_max = train_data[:, 0].min() - 1, train_data[:, 0].max() + 1
y_min, y_max = train_data[:, 1].min() - 1, train_data[:, 1].max() + 1

xx, yy = \
  np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)

plt.pcolormesh(xx, yy, Z, cmap = plt.cm.Paired)

# plot training points
plt.scatter(train_data[:, 0], train_data[:, 1], c = train_target[:, 2],\
 cmap = plt.cm.Paired, s = 80, alpha = 0.5)

plt.xlim(-0.60, 0.60)
plt.ylim(-0.35, 0.35)
plt.xlabel('1st component')
plt.ylabel('2nd component')

title_cap = \
  'K-nearest neighbors (n = 5) of Hillenbrand vowel data: training data'
plt.title(title_cap)


# plot test points
plt.pcolormesh(xx, yy, Z, cmap = plt.cm.Paired)

plt.scatter(test_data[:, 0], test_data[:, 1], c = test_target[:, 2], \
  cmap = plt.cm.Paired, marker = 's', s = 80, alpha = 0.5)

plt.xlim(-0.60, 0.60)
plt.ylim(-0.35, 0.35)
plt.xlabel('1st component')
plt.ylabel('2nd component')

title_cap = \
  'K-nearest neighbors (n = 5) of Hillenbrand vowel data: test data'
plt.title(title_cap)

# support vector machine
# fit model, plot decision boundary
svm.SVC(kernel = 'rbf')

svc.fit(train_data, train_target[:, 2])

h = 0.01
x_min, x_max = train_data[:, 0].min() - 1, train_data[:, 0].max() + 1
y_min, y_max = train_data[:, 1].min() - 1, train_data[:, 1].max() + 1

xx, yy = \
  np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap = plt.cm.Paired)

# plot training points
plt.scatter(train_data[:, 0], train_data[:, 1], c = train_target[:, 2],\
 cmap = plt.cm.Paired, s = 80, alpha = 0.5)

plt.xlim(-0.60, 0.60)
plt.ylim(-0.35, 0.35)
plt.xlabel('1st component')
plt.ylabel('2nd component')

title_cap = \
  'SVM (radial basis function) of Hillenbrand vowel data: training data'
plt.title(title_cap)


# plot test points
plt.pcolormesh(xx, yy, Z, cmap = plt.cm.Paired)

plt.scatter(test_data[:, 0], test_data[:, 1], c = test_target[:, 2], \
  cmap = cmap_bold, marker = 's', s = 80, alpha = 0.5)

plt.xlim(-0.60, 0.60)
plt.ylim(-0.60, 0.60)
plt.xlabel('1st component')
plt.ylabel('2nd component')

title_cap = \
  'SVM (radial basis function) of Hillenbrand vowel data: test data'
plt.title(title_cap)
