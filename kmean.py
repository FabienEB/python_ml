# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:21:21 2017
@author: fbaker
"""

import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr as pr


#matrix = pd.read_csv('/Users/fabien.baker/Desktop/cluster_saigon_20180717_1_0.csv')
matrix = ped_hcm_df_2


feature_x = matrix[matrix.columns[1:5]]     
y = matrix['geohash']  

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans

scaler = StandardScaler()
x_standardized = scaler.fit_transform(feature_x)
#Set 7 centroids, there is a test for this based on the cohort size but I cant remember 
cluster_mini = MiniBatchKMeans(n_clusters=12)
#matrix['cluster'] = cluster.fit_predict(x_standardized[x_standardized.columns[1:]])
matrix['cluster_mini'] = cluster_mini.fit_predict(feature_x)
#matrix.cluster.value_counts()
#matrix.to_csv("full_cluster_list_crm_v5.csv")
matrix.to_csv('/Users/fabien.baker/Desktop/Everything/work/heatmap/cluster_output/hcm_cluster_12.csv')

##############################################################

# END OF THE CLUSTERING PART #

##############################################################
""""
x_cols = matrix.columns[1:]
x_cols2 = matrix.columns[:]
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
matrix['x'] = pca.fit_transform(matrix[x_cols])[:,0]
matrix['y'] = pca.fit_transform(matrix[x_cols])[:,1]
matrix = matrix.reset_index()
customer_clusters = matrix[['unique_passenger_id','cluster','x','y']]
y = matrix['y']
x = matrix['x']
"""

# df[1].fillna(0, inplace=True)
# matrix_x['matrix_y']=matrix_y