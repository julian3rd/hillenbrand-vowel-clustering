# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 14:22:18 2015

@author: julian
"""


import os


%run '~/GitHub/hillenbrand-vowel-clustering/hillenbrand-data-setup.py'

#===============================================================================
# Part 1: formants (absolute and steady state values)
#===============================================================================

#------------------------------------------------------------------------------ 
# PCA to reduce

from sklearn import decomposition

pca = decomposition.PCA(n_components = 2)

formant_reduced = pca.fit_transform(formant_mtx)
formant_ratio_reduced = pca.fit_transform(formant_ratio_mtx)

#------------------------------------------------------------------------------ 

# k-means clustering
# clusters: 12 (vowel), 6 (height), 3 (position)

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

kmeans = KMeans(n_clusters = 12)

kmeans.fit(formant_reduced)

labels = kmeans.labels_

plt.scatter(formant_reduced[:, 0], formant_reduced[:, 1], \
 c = labels.astype(np.float))

# put reduced data, labels, ground truth, vowel into DataFrame
# plot components against each other
# match target colors to cluster labels

kmeans_data = pd.DataFrame(formant_reduced, columns = ['PC1', 'PC2'])

kmeans_data['Label'] = kmeans.labels_
kmeans_data['Target'] = target_mtx[:, 2]
kmeans_data['Vowel'] = hillenbrand_data.Vowel

mk_dict = OrderedDict()
for vowel in np.unique(kmeans_data.Vowel):
    mk_dict[vowel] = '$' + vowel + '$'
        
kmeans_data['VowelMarker'] = kmeans_data.Vowel.map(mk_dict)

kmeans_data['ID'] = hillenbrand_data.ID
kmeans_data['Sex'] = hillenbrand_data.SexTarget
kmeans_data['Height'] = hillenbrand_data.Height
kmeans_data['Position'] = hillenbrand_data.Position
kmeans_data['Rounding'] = hillenbrand_data.Rounding

kmeans_data.head()


kmeans_data.plot(kind = 'scatter', x = 'PC1', y = 'PC2', \
  c = 'Label', s  = 60, alpha = 0.5)


#------------------------------------------------------------------------------ 
# Gaussian mixture models

from sklearn.mixture import GMM

gaussmm = GMM(n_components = 12)

gaussmm.fit(formant_reduced)

labels = gaussmm.predict(formant_reduced)

gaussmm_data = pd.DataFrame(formant_reduced, columns = ['PC1', 'PC2'])

gaussmm_data['Label'] = gaussmm.predict(formant_reduced)
gaussmm_data['Target'] = target_mtx[:, 2]
gaussmm_data['Vowel'] = hillenbrand_data.Vowel

      
gaussmm_data['VowelMarker'] = gaussmm_data.Vowel.map(mk_dict)

gaussmm_data['ID'] = hillenbrand_data.ID
gaussmm_data['Sex'] = hillenbrand_data.SexTarget
gaussmm_data['Height'] = hillenbrand_data.Height
gaussmm_data['Position'] = hillenbrand_data.Position
gaussmm_data['Rounding'] = hillenbrand_data.Rounding

gaussmm_data.head()


gaussmm_data.plot(kind = 'scatter', x = 'PC1', y = 'PC2', \
  c = 'Label', s  = 60, alpha = 0.5)

# decided to say screw it and plot in ggplot2

kmeans_data.to_csv(os.path.expanduser(save_path + 'hillenbrand-kmeans-formant-data.csv'), Index = False)
gaussmm_data.to_csv(os.path.expanduser(save_path + 'hillenbrand-gaussmm-formant-data.csv'), Index = False)

#===============================================================================
# Part 2: formant ratios
#===============================================================================

# k-means clustering
# clusters: 12 (vowel), 6 (height), 3 (position)

kmeans = KMeans(n_clusters = 12)

kmeans.fit(formant_ratio_reduced)

labels = kmeans.labels_

plt.scatter(formant_ratio_reduced[:, 0], formant_ratio_reduced[:, 1], \
 c = labels.astype(np.float))

# put reduced data, labels, ground truth, vowel into DataFrame
# plot components against each other
# match target colors to cluster labels

kmeans_data = pd.DataFrame(formant_ratio_reduced, columns = ['PC1', 'PC2'])

kmeans_data['Label'] = kmeans.labels_
kmeans_data['Target'] = target_mtx[:, 2]
kmeans_data['Vowel'] = hillenbrand_data.Vowel

mk_dict = OrderedDict()
for vowel in np.unique(kmeans_data.Vowel):
    mk_dict[vowel] = '$' + vowel + '$'
        
kmeans_data['VowelMarker'] = kmeans_data.Vowel.map(mk_dict)

kmeans_data['ID'] = hillenbrand_data.ID
kmeans_data['Sex'] = hillenbrand_data.SexTarget
kmeans_data['Height'] = hillenbrand_data.Height
kmeans_data['Position'] = hillenbrand_data.Position
kmeans_data['Rounding'] = hillenbrand_data.Rounding

kmeans_data.head()


kmeans_data.plot(kind = 'scatter', x = 'PC1', y = 'PC2', \
  c = 'Label', s  = 60, alpha = 0.5)


#------------------------------------------------------------------------------ 
# Gaussian mixture models

gaussmm = GMM(n_components = 12)

gaussmm.fit(formant_ratio_reduced)

labels = gaussmm.predict(formant_ratio_reduced)

gaussmm_data = pd.DataFrame(formant_ratio_reduced, columns = ['PC1', 'PC2'])

gaussmm_data['Label'] = gaussmm.predict(formant_ratio_reduced)
gaussmm_data['Target'] = target_mtx[:, 2]
gaussmm_data['Vowel'] = hillenbrand_data.Vowel

      
gaussmm_data['VowelMarker'] = gaussmm_data.Vowel.map(mk_dict)

gaussmm_data['ID'] = hillenbrand_data.ID
gaussmm_data['Sex'] = hillenbrand_data.SexTarget
gaussmm_data['Height'] = hillenbrand_data.Height
gaussmm_data['Position'] = hillenbrand_data.Position
gaussmm_data['Rounding'] = hillenbrand_data.Rounding

gaussmm_data.head()


gaussmm_data.plot(kind = 'scatter', x = 'PC1', y = 'PC2', \
  c = 'Label', s  = 60, alpha = 0.5)

kmeans_data.to_csv(os.path.expanduser(save_path + 'hillenbrand-kmeans-formant-ratio-data.csv'), Index = False)
gaussmm_data.to_csv(os.path.expanduser(save_path + 'hillenbrand-gaussmm-formant-ratio-data.csv'), Index = False)