# Jupyter/IPython notebooks

The directory contains the notebooks that implement the data formatting, clustering and visualization of the vowel observations.

## Notebooks

### `hillenbrand-vowel-data-setup.ipynb` (Python)
This notebook imports, cleans (removes observations with missing values), and sets up the data for clustering in `scikit-learn`.  

### `hillenbrand-unsupervised-complete.ipynb` (Python)
This notebook implements the unsupervised learning of vowel clusters by using k-means clustering and Gaussian mixture models. Features are principal components of the formants (normalized) and formant ratios.  

### `vowel-formant-clustering-plots.ipynb` (R)
This notebook takes the formant clustering data, plots the predictors against each other and highlights the observations according to cluster placement and vowel position.  

### ### `vowel-formant-ratios-clustering-plots.ipynb` (R)  
This notebook takes the formant ratio clustering data, plots the predictors against each other and highlights the observations according to cluster placement and vowel position.  