# -*- coding: utf-8 -*-
#!/usr/bin/env python

import pandas as pd
import numpy as np
import os


#------------------------------------------------------------------------------ 
# import data
data_path = '~/GitHub/hillenbrand-vowel-clustering/data/'

def parse_data(filename):
  df = pd.read_csv(filename, index_col = 0)
  return df

hillenbrand_file = \
  os.path.expanduser(data_path + 'hillenbrand-vowel-formatted.csv')

hillenbrand_data = parse_data(hillenbrand_file)

# remove rows with ANY zero values
hillenbrand_data = hillenbrand_data.replace(0, np.nan)
hillenbrand_data = hillenbrand_data.dropna()


hillenbrand_data = hillenbrand_data.reset_index()

hillenbrand_data = hillenbrand_data.drop('Index', 1)

hillenbrand_data.head()


#------------------------------------------------------------------------------ 
# additional ID columns
# adding columns to ID by Sex, Word (Vowel)


# Sex column
for x in range(hillenbrand_data.shape[0]):
    if hillenbrand_data.ix[x, 'ID'].startswith('m'):
        hillenbrand_data.ix[x, 'Sex'] = 'male'
    elif hillenbrand_data.ix[x, 'ID'].startswith('w'):
        hillenbrand_data.ix[x, 'Sex'] = 'female'
    elif hillenbrand_data.ix[x, 'ID'].startswith('b'):
        hillenbrand_data.ix[x, 'Sex'] = 'boy'
    elif hillenbrand_data.ix[x, 'ID'].startswith('g'):
        hillenbrand_data.ix[x, 'Sex'] = 'girl'

hillenbrand_data.head()

# Word/Vowel columns
for x in range(hillenbrand_data.shape[0]):
    if hillenbrand_data.ix[x, 'ID'].endswith('ae'):
        hillenbrand_data.ix[x, 'Word'] = 'had'
    elif hillenbrand_data.ix[x, 'ID'].endswith('ah'):
        hillenbrand_data.ix[x, 'Word'] = 'hod'
    elif hillenbrand_data.ix[x, 'ID'].endswith('aw'):
        hillenbrand_data.ix[x, 'Word'] = 'hawed'
    elif hillenbrand_data.ix[x, 'ID'].endswith('eh'):
        hillenbrand_data.ix[x, 'Word'] = 'head'
    elif hillenbrand_data.ix[x, 'ID'].endswith('er'):
        hillenbrand_data.ix[x, 'Word'] = 'heard'
    elif hillenbrand_data.ix[x, 'ID'].endswith('ei'):
        hillenbrand_data.ix[x, 'Word'] = 'haid'
    elif hillenbrand_data.ix[x, 'ID'].endswith('ih'):
        hillenbrand_data.ix[x, 'Word'] = 'hid'
    elif hillenbrand_data.ix[x, 'ID'].endswith('iy'):
        hillenbrand_data.ix[x, 'Word'] = 'heed'
    elif hillenbrand_data.ix[x, 'ID'].endswith('oa'):
        hillenbrand_data.ix[x, 'Word'] = 'boat'
    elif hillenbrand_data.ix[x, 'ID'].endswith('oo'):
        hillenbrand_data.ix[x, 'Word'] = 'hood'
    elif hillenbrand_data.ix[x, 'ID'].endswith('uh'):
        hillenbrand_data.ix[x, 'Word'] = 'hud'
    elif hillenbrand_data.ix[x, 'ID'].endswith('uw'):
        hillenbrand_data.ix[x, 'Word'] = 'whod'

hillenbrand_data.head()

# Vowel column (redundant, but easy to understand)
for x in range(hillenbrand_data.shape[0]):
    if hillenbrand_data.ix[x, 'ID'].endswith('ae'):
        hillenbrand_data.ix[x, 'Vowel'] = 'ae'
    elif hillenbrand_data.ix[x, 'ID'].endswith('ah'):
        hillenbrand_data.ix[x, 'Vowel'] = 'ah'
    elif hillenbrand_data.ix[x, 'ID'].endswith('aw'):
        hillenbrand_data.ix[x, 'Vowel'] = 'aw'
    elif hillenbrand_data.ix[x, 'ID'].endswith('eh'):
        hillenbrand_data.ix[x, 'Vowel'] = 'eh'
    elif hillenbrand_data.ix[x, 'ID'].endswith('er'):
        hillenbrand_data.ix[x, 'Vowel'] = 'er'
    elif hillenbrand_data.ix[x, 'ID'].endswith('ei'):
        hillenbrand_data.ix[x, 'Vowel'] = 'ei'
    elif hillenbrand_data.ix[x, 'ID'].endswith('ih'):
        hillenbrand_data.ix[x, 'Vowel'] = 'ih'
    elif hillenbrand_data.ix[x, 'ID'].endswith('iy'):
        hillenbrand_data.ix[x, 'Vowel'] = 'iy'
    elif hillenbrand_data.ix[x, 'ID'].endswith('oa'):
        hillenbrand_data.ix[x, 'Vowel'] = 'oa'
    elif hillenbrand_data.ix[x, 'ID'].endswith('oo'):
        hillenbrand_data.ix[x, 'Vowel'] = 'oo'
    elif hillenbrand_data.ix[x, 'ID'].endswith('uh'):
        hillenbrand_data.ix[x, 'Vowel'] = 'uh'
    elif hillenbrand_data.ix[x, 'ID'].endswith('uw'):
        hillenbrand_data.ix[x, 'Vowel'] = 'uw'

hillenbrand_data.head()

from collections import OrderedDict

height_dict = OrderedDict()
pos_dict = OrderedDict()

height_dict = \
  {'ae': 'near-open', 'ah': 'near-open', 'aw': 'open', 'eh': 'open-mid',
   'er': 'mid', 'ei': 'close-mid', 'ih': 'close-mid','iy': 'close',
   'oa': 'close-mid', 'oo': 'close-mid', 'uh': 'open-mid', 'uw': 'close'}

pos_dict = \
  {'ae': 'front', 'ah': 'central', 'aw': 'back', 'eh': 'front',
   'er': 'central', 'ei': 'front', 'ih': 'front','iy': 'back',
   'oa': 'back', 'oo': 'back', 'uh': 'back', 'uw': 'back'}

hillenbrand_data['Height'] = hillenbrand_data.Vowel.map(height_dict)
hillenbrand_data['Position'] = hillenbrand_data.Vowel.map(pos_dict)


# rounding; iterative loop, but fine since relatively small
for x in xrange(hillenbrand_data.shape[0]):
	if hillenbrand_data.ix[x, 'Vowel'] == 'ae':
		hillenbrand_data.ix[x, 'Rounding'] = 1
	elif hillenbrand_data.ix[x, 'Vowel'] == 'aw':
		hillenbrand_data.ix[x, 'Rounding'] = 1
	elif hillenbrand_data.ix[x, 'Vowel'] == 'oa':
		hillenbrand_data.ix[x, 'Rounding'] = 1
	elif hillenbrand_data.ix[x, 'Vowel'] == 'uw':
		hillenbrand_data.ix[x, 'Rounding'] = 1
	else:
		hillenbrand_data.ix[x, 'Rounding'] = 0
		

# map target names to values


sex_dict = {'male': 0, 'female':1, 'boy':2, 'girl': 3}
word_dict = OrderedDict()
vowel_dict = OrderedDict()

word_list = np.unique(hillenbrand_data.Word.values.tolist())

vowel_list = np.unique(hillenbrand_data.Vowel.values.tolist())

for idx, word in enumerate(word_list):
    word_dict[word] = idx

for idx, vowel in enumerate(vowel_list):
    vowel_dict[vowel] = idx

hillenbrand_data['SexTarget'] = hillenbrand_data.Sex.map(sex_dict)
hillenbrand_data['WordTarget'] = hillenbrand_data.Word.map(word_dict)
hillenbrand_data['VowelTarget'] = hillenbrand_data.Vowel.map(vowel_dict)

hillenbrand_data.head()

#------------------------------------------------------------------------------ 
# normalization of formant values
# Q: should this be done by Vowel, ID or Vowel X ID?
# Done across entire dataset for now

from scipy import stats

hillenbrand_data['F0_zscore'] = stats.zscore(hillenbrand_data.F0)
hillenbrand_data['F1_zscore'] = stats.zscore(hillenbrand_data.F1)
hillenbrand_data['F2_zscore'] = stats.zscore(hillenbrand_data.F2)
hillenbrand_data['F3_zscore'] = stats.zscore(hillenbrand_data.F3)
hillenbrand_data['F4_zscore'] = stats.zscore(hillenbrand_data.F4)

hillenbrand_data['F1_20_zscore'] = stats.zscore(hillenbrand_data.F1_20)
hillenbrand_data['F2_20_zscore'] = stats.zscore(hillenbrand_data.F2_20)
hillenbrand_data['F3_20_zscore'] = stats.zscore(hillenbrand_data.F3_20)

hillenbrand_data['F1_80_zscore'] = stats.zscore(hillenbrand_data.F1_80)
hillenbrand_data['F2_80_zscore'] = stats.zscore(hillenbrand_data.F2_80)
hillenbrand_data['F3_80_zscore'] = stats.zscore(hillenbrand_data.F3_80)

hillenbrand_data.head()

# additional features: formant ratios
hillenbrand_data['F1_F2_ratio'] = \
  np.divide(hillenbrand_data.F1.values, hillenbrand_data.F2.values)
  
hillenbrand_data['F1_F3_ratio'] = \
  np.divide(hillenbrand_data.F1.values, hillenbrand_data.F3.values)
  
hillenbrand_data['F2_F3_ratio'] = \
  np.divide(hillenbrand_data.F2.values, hillenbrand_data.F3.values)


#------------------------------------------------------------------------------ 
# Features and conversion to NumPy matrix

formant_columns = \
  ['F0_zscore', 'F1_zscore', 'F2_zscore', 'F3_zscore', 'F4_zscore',
  'F1_20_zscore', 'F2_20_zscore', 'F3_20_zscore',
  'F1_80_zscore', 'F2_80_zscore', 'F3_80_zscore']

formant_ratio_columns = ['F1_F2_ratio', 'F1_F3_ratio', 'F2_F3_ratio']

def create_np_matrix(input_data):
    matrix_rep = input_data.as_matrix()
    matrix_rep = np.float64(matrix_rep)
    return matrix_rep

formant_ratio_data = hillenbrand_data.ix[:, formant_ratio_columns]

formant_ratio_mtx = create_np_matrix(formant_ratio_data)

formant_data = hillenbrand_data.ix[:, formant_columns]

formant_mtx = create_np_matrix(formant_data)

target_columns = ['SexTarget', 'WordTarget', 'VowelTarget']

target_data = hillenbrand_data.ix[:, target_columns]

target_mtx = create_np_matrix(target_data)

