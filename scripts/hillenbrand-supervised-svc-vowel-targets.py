# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Created on Sun Jul 12 10:27:24 2015

Supervised learning for the Hillenbrand vowel data.
This script only implements a Support Vector Classifier
with Polynomial kernel

targets: vowel class
only vowel ID implemented

predictors: 
formant values (steady-state, percentages of steady-state);

formant ratios and principal components (2D) are also added to 
DataFrame for curiosity's sake

"""

%run '~/GitHub/hillenbrand-vowel-clustering/scripts/hillenbrand-data-setup.py'

#------------------------------------------------------------------------------ 
# PCA to reduce

from sklearn.decomposition import PCA

pca = PCA(n_components = 2)

formant_reduced = pca.fit_transform(formant_mtx)
formant_ratio_reduced = pca.fit_transform(formant_ratio_mtx)

# add PCA columns to hillenbrand_data DataFrame
# going to use vowel labels to make interpretation easier
# for classification reports and confusion matrices
# perhaps not needed for this implementation, but 
# could be used as additional predictors

hillenbrand_data['Formant_PC1'] = formant_reduced[:, 0]
hillenbrand_data['Formant_PC2'] = formant_reduced[:, 1]

hillenbrand_data['Formant_Ratio_PC1'] = formant_ratio_reduced[:, 0]
hillenbrand_data['Formant_Ratio_PC2'] = formant_ratio_reduced[:, 1]

#------------------------------------------------------------------------------ 

# import needed modules and functions

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedKFold, train_test_split 

#------------------------------------------------------------------------------ 
# data split into train and test sets

# 80/20 data split

# using untransformed (non-normal and PCA) columns to predict
# other columns there just in case
# normed columns should give same results; 
# difference is that model should converge faster

formant_cols = ['F1', 'F2', 'F3', 'F4', 
                'F1_20', 'F2_20', 'F3_20', 
                'F1_50', 'F2_50', 'F3_50',
                'F1_80', 'F2_80', 'F3_80']


formant_norm_cols = ['F1_zscore', 'F2_zscore', 'F3_zscore', 'F4_zscore', 
                     'F1_20_zscore', 'F2_20_zscore', 'F3_20_zscore', 
                     'F1_50_zscore', 'F2_50_zscore', 'F3_50_zscore',
                     'F1_80_zscore', 'F2_80_zscore', 'F3_80_zscore']


vowel_train, vowel_test, vowel_label_train, vowel_label_test = \
  train_test_split(hillenbrand_data[formant_cols], hillenbrand_data['Vowel'], 
                   test_size = 0.2, random_state = 8393)


#------------------------------------------------------------------------------ 

# set up classifier parameters
  
svc = ('classifier', SVC(kernel = 'poly', probability = True, random_state = 4748))

# creating pipeline

svc_pipeline = Pipeline([svc])

# set up parameter grid search

svc_params = \
  {'classifier__C': (np.logspace(-4, 3, num = 7)),
  'classifier__gamma': (np.logspace(-4, 3, num = 7)),
  'classifier__degree': (3, 5, 7)}

# create search grid

vowel_grid = \
  GridSearchCV(svc_pipeline, svc_params, refit = True, n_jobs = -1,
               scoring = 'accuracy', cv = StratifiedKFold(vowel_label_train, n_folds = 5))

#------------------------------------------------------------------------------ 

# fitting using search parameters

%time vowel_detector = vowel_grid.fit(vowel_train, vowel_label_train)


# predictions and performance metrics


vowel_prediction = vowel_detector.predict(vowel_test)

vowel_prediction_probs = vowel_detector.predict_proba(vowel_test)

print 'Target class: Vowel'
print 'Classification report'
print classification_report(vowel_label_test, vowel_prediction)
print 'Confusion matrix'
print confusion_matrix(vowel_label_test, vowel_prediction)
