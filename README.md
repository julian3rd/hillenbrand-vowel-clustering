# Clustering of the Hillenbrand vowel data

This repo contains Jupyter/IPython notebooks and Python and R scripts using the data from [Hillenbrand *et al.* 1995](http://homepages.wmich.edu/~hillenbr/voweldata.html) on the acoustic measurements on American English vowels. The vowels are separated into different clusters using k-means clustering and Gaussian mixture models. Clustering is implemented in `scikit-learn` and visualization is performed in R using `ggplot2`. Formants and formant ratios are used as the features for the clustering.  

# Steps:  
1. Import data and remove rows with at least one missing observation  
2. Identify observation by word, vowel, and sex of speaker  
3. Map vowel characteristics to observation (e.g., front, open-mid, etc.)  
4. Create targets for speaker sex, word, and vowel (for supervised learning)  
5. Normalizing features (z-score)  
6. Create feature matrices  
7. Implement clustering algorithms  
8. Visualize clusters and feature space  

# Directories and contents
## `data`:  
  - Vowel measurement data in CSV format  
  - Vowel observations and cluster assignments for both k-means and Gaussian mixture models  

## `notebooks`:  
  - Jupyter/IPython notebook using Python kernel to format data for unsupervised learning  
  - Jupyter/IPython notebook using Python kernel to implement clustering of vowel observations  
  - Jupyter/IPython notebooks using R kernel to plot the data  

## `scripts`:  
  - Python script to format data for unsupervised learning  
  - Python script to implement clustering of vowel observations  
  - R scripts to plot the data 