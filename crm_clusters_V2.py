# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:21:21 2017

@author: fbaker
"""

import os
import sys
sys.path.append(
    os.path.normpath(os.path.expanduser('~/Documents/GitHub/business-intelligence/python'))
)
from datavirtuality.datavirtuality import DataVirtualityConnection
from utilities import utilities
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr as pr
from datavirtuality.datavirtuality import DataVirtualityConnection


from utilities.utilities import get_credentials
cred = get_credentials()
with DataVirtualityConnection(cred.data_virtuality_user, cred.data_virtuality_pw) as dv_cnxn:

            query_a = """
            select
            *
            FROM sandbox_marketing.crm_persona_test
            LIMIT %(limit)s
            """
            matrix = dv_cnxn.query(query_a, parameters={'limit': 500000}).get_as_dataframe()
            matrix.to_pickle('matrix')

matrix.fillna(0, inplace=True)    

feature_x = matrix[matrix.columns[1:]]     
y = matrix['unique_passenger_id']  

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

scaler = StandardScaler()
x_standardized = scaler.fit_transform(feature_x)


#Set 7 centroids, there is a test for this based on the cohort size but I cant remember 
cluster = KMeans(n_clusters=7)
#matrix['cluster'] = cluster.fit_predict(x_standardized[x_standardized.columns[1:]])
matrix['cluster'] = cluster.fit_predict(x_standardized)
matrix.cluster.value_counts()


matrix.to_csv("full_cluster_list_crm_v5.csv")

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
