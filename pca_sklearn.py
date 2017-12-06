# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 13:52:13 2017

@author: fbaker
"""

import pandas as pd
df_behaviour = pd.read_csv('pca_ride_data2.csv')
col_names = df_behaviour.columns.tolist()
features= df_behaviour[df_behaviour.columns[2:]]
eats_installed =df_behaviour['new_y']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(features)

from sklearn import decomposition
pca = decomposition.PCA(n_components=1, svd_solver='full')
pca.fit(features)
#print(pca.explained_variance_ratio_ )  
pcac = pca.components_   
pca.n_components =1
X_reduced = pca.fit_transform(features)
X_reduced.shape

a = features.columns

pd.DataFrame(pcac, columns=[a])