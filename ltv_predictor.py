# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 12:07:09 2018

@author: fbaker
"""


import pandas
import datetime
from sklearn.cluster import AffinityPropagation
import sklearn.preprocessing 
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.externals import joblib
from sklearn.decomposition import PCA
import pickle

import sys
# check if we are on adhoc where xgb is installed

import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import AdaBoostRegressor,ExtraTreesClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn import neural_network
from sklearn.feature_selection import SelectKBest

import sklearn.kernel_ridge
from sklearn.svm import SVC


import sqlite3
conn = sqlite3.connect('ltv_predicitions.db')

bins = np.linspace(1, 10, 19)


from classification_visualization import classification_visualizer
import training_manager
import data_cleanup

# needed for db saves of performance

# only use one time for logging so that it is easier to identify one "run"


if sys.platform == 'linux':
    import xgboost as xgb

#    def plot_confusion_matrix(cm, classes,
#                              normalize=False,
#                              title='Confusion matrix',
#                              cmap=plt.cm.Blues):
#        """
#        This function prints and plots the confusion matrix.
#        Normalization can be applied by setting `normalize=True`.
#        """
#        if normalize:
#            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#            print("Normalized confusion matrix")
#        else:
#            print('Confusion matrix, without normalization')
#    
#        print(cm)
#    
#        plt.imshow(cm, interpolation='none', cmap=cmap)
#        plt.title(title)
#        plt.colorbar()
#        tick_marks = np.arange(len(classes))
#        plt.xticks(tick_marks, classes, rotation=45)
#        plt.yticks(tick_marks, classes)
#    
#        fmt = '.2f' if normalize else 'd'
#        thresh = cm.max() / 2.
#        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#            plt.text(j, i, format(cm[i, j], fmt),
#                     horizontalalignment="center",
#                     color="white" if cm[i, j] > thresh else "black")
#    
#        plt.tight_layout()
#        plt.ylabel('True label')
#        plt.xlabel('Predicted label')
#else:
    # for windows memory leak
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    sns.heatmap(cm)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def data_prep(df):

    df = df.replace(np.inf, np.nan)
    tf = df.fillna(0)  
    tf["non_cancelled_rides_after_first"]= (tf["non_cancelled_rides_after_first"]).clip(0,1)
    tf["bookings_pre_first_ride"] = (tf.bookings_pre_first_ride -1).clip(0,1)    
    return tf

def r2(y_true, y_pred):
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return  1-  (sum((y_true - y_pred)** 2) / sum((y_true - np.mean(y_true))** 2) )

def log_scaling(X, name):
    return np.log10(1+ X[name])


def y_proba(clf, X):
    yproba= [y_proba_i[1] for y_proba_i in clf.predict_proba(X)]
    return yproba

def y_proba_class(clf,X, boundary):
    yproba =  y_proba(clf, X)
    return [0 if x < boundary else 1 for x in yproba]

def clean_df(df, good_cols):
    
    cols = df.columns
    
    return df[ [c for c in cols if ("_prediction" in c) or ( c in good_cols )] ]

#def plot_confusion_matrix(cm, classes,
#                          normalize=False,
#                          title='Confusion matrix',
#                          cmap=plt.cm.Blues):
#    """
#    This function prints and plots the confusion matrix.
#    Normalization can be applied by setting `normalize=True`.
#    """
#    if normalize:
#        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#        print("Normalized confusion matrix")
#    else:
#        print('Confusion matrix, without normalization')
#
#    print(cm)
#
#    plt.imshow(cm, interpolation='none', cmap=cmap)
#    plt.title(title)
#    print("done")
#
#    plt.colorbar()
#    print("done2")
#
#    tick_marks = np.arange(len(classes))
#    print("ticks")
#    
#
#
#    plt.yticks(tick_marks, classes)
#    print("tick2s")    
#    
#    plt.xticks(tick_marks, classes, rotation=45)
#    
#    print("done3")
#
#    fmt = '.2f' if normalize else 'd'
#    thresh = cm.max() / 2.
#    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#        plt.text(j, i, format(cm[i, j], fmt),
#                 horizontalalignment="center",
#                 color="white" if cm[i, j] > thresh else "black")
#    print("done4")
#
#    plt.tight_layout()
#    plt.ylabel('True label')
#    plt.xlabel('Predicted label')
regressions = [

# {
#  'name': 'Dummy stratified',
#  'instance': DummyClassifier(strategy='stratified')
#  # score 0.
#  },

{
"name": "NN Class",
"instance": sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(50,50,50 ))

},
{
"name": "NN Class 2",
"instance": sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(50,10,10 ))

},
{
"name": "NN Class 3",
"instance": sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(50,50,50,50,50,50 ))

},
{
"name": "NN Class 4",
"instance": sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(30,30,20,20,10))

},
{
"name": "Bagged NN",
"instance": sklearn.ensemble.BaggingClassifier(
    sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(50,50,50,20,20,20,10,10))
    , n_estimators =  20
    , max_samples = 0.6
    )
},



                     
{
"name": "NN Class $",
"instance": sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(50
                                                                     ,50
                                                                     ,50
                                                                     ,50
                                                                     ,50
                                                                     ,50
                                                                     ,50
                                                                     ,50
                                                                     ,50
                                                                     ,50
                                                                     ,50)
                                                   , max_iter =2000
                                                   , tol= 0.00001
                                                   # , verbose = True
                                                   )

},
{
"name": "NN Class $ 3",
"instance": sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(120
                                                                     ,50
                                                                     ,30
                                                                     ,30
                                                                     ,10
                                                                     ,10
                                                                     ,10
                                                                     ,10
                                                                     ,5
                                                                     ,5
                                                                     ,5)
                                                   , max_iter =2000
                                                   , tol= 0.00001
                                                   # , verbose = True
                                                   )

},    
   {
"name": "NN Class $2",
"instance": sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(120
                                                                     ,120
                                                                     ,120
                                                                     ,50
                                                                     ,50
                                                                     ,50
                                                              )
                                                   , max_iter =2000
                                                   , tol= 0.00001
                                                   # , verbose = True
                                                   )

}    ,
#{
# 'name': 'Dummy most_frequent',
# 'instance': DummyClassifier(strategy='most_frequent')
## score 0.
# },
# {
# 'name': 'Dummy uniform',
# 'instance': DummyClassifier(strategy='uniform') 
# # score  -0.224207294589
#
# }, 
 {
 "name": "SGDClassifier",
 "instance": SGDClassifier(loss = "log", shuffle = True)
 
 },
  {
 "name": "SGDClassifier (l1)",
 "instance": SGDClassifier(loss = "log", shuffle = True, penalty = "l1")
 
 },

 {
"name": "GaussianNB",
"instance": GaussianNB()

},

{
 "name": "logistic regression",
 "instance": sklearn.linear_model.LogisticRegression()
 },

 
 {
 "name": "logistic regression (L1)",
 "instance": sklearn.linear_model.LogisticRegression(penalty = "l1", solver = "liblinear")
 },


{
 "name": "SVC",
 "instance": SVC()
 } ,
 
{
"name": "voting classifier soft",
"instance": sklearn.ensemble.VotingClassifier(
        estimators = [("logistic regression",sklearn.linear_model.LogisticRegression())
                     ,("sgd",SGDClassifier(loss = "log") )#
                     ,("xgb",xgb.XGBClassifier(max_depth =4, n_estimators =500) )
                     , ("xgb2", xgb.XGBClassifier(max_depth =8, n_estimators =800))
                     ,("nn", sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(50,50,50 )))
                     ,("svc",SVC(probability = True))
                     ,("bagged XGBC 1", sklearn.ensemble.BaggingClassifier(
                                           xgb.XGBClassifier(max_depth =4
                                                            ,n_estimators =400)
                                          , n_estimators =  5
                                          , max_samples = 0.7))
                     ,("bagged XGBC 2", sklearn.ensemble.BaggingClassifier(
                                           xgb.XGBClassifier(max_depth =10
                                                            ,n_estimators =1000)
                                          , n_estimators =  10
                                          , max_samples = 0.7)
                                      )
                      ,("NN Class $2", sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(120
                                                                    ,120
                                                                   
                                                                    ,50
                                                                    ,50
                                                                    ,50
                                                             )
                                                  , max_iter =2000
                                                  , tol= 0.0001
                                                  , verbose = True))


        ]
        
        , voting = "soft"
        , n_jobs=  2
        )
},
 {
 "name": "voting classifier hard",
 "instance": sklearn.ensemble.VotingClassifier(
         estimators = [("logistic regression",sklearn.linear_model.LogisticRegression())
                      ,("sgd",SGDClassifier(loss = "log") )#
                      ,("xgb",xgb.XGBClassifier(max_depth =4, n_estimators =500) )
                      , ("xgb2", xgb.XGBClassifier(max_depth =8, n_estimators =800))
                      ,("nn", sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(50,50,50 )))
                      ,("svc",SVC())
#                       ,("bagged XGBC 1", sklearn.ensemble.BaggingClassifier(
#                                             xgb.XGBClassifier(max_depth =4
#                                                              ,n_estimators =400)
#                                            , n_estimators =  5
#                                            , max_samples = 0.7))
#                       ,("bagged XGBC 2", sklearn.ensemble.BaggingClassifier(
#                                             xgb.XGBClassifier(max_depth =10
#                                                              ,n_estimators =1000)
#                                            , n_estimators =  10
#                                            , max_samples = 0.7)
#                                        )
                       ,("NN Class $2", sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(120
                                                                     ,120
                                                                    
                                                                     ,50
                                                                     ,50
                                                                     ,50
                                                              )
                                                   , max_iter =2000
                                                   , tol= 0.0001
                                                   , verbose = True))


         ]
         
         , voting = "hard"
         , n_jobs=  2
         )
}
]
if sys.platform == 'linux':
     regressions.extend(
        [
              {
         'name':'XGBC',
         'instance':  xgb.XGBClassifier(max_depth =4, n_estimators =500)
         }, 
               {
         'name':'XGBC 4',
         'instance':  xgb.XGBClassifier(max_depth =7, n_estimators =400,min_child_weight = 100)
         },                  
              {
         'name':'XGBC 2',
         'instance':  xgb.XGBClassifier(max_depth =8, n_estimators =800)
         }, 

               {
         'name':'XGBC 3',
         'instance':  xgb.XGBClassifier(max_depth =7, n_estimators =2000,min_child_weight = 50)
         },                
         {
         'name': 'bagged XGBC', 
         'instance': sklearn.ensemble.BaggingClassifier(
                                             xgb.XGBClassifier(max_depth =4
                                                              ,n_estimators =400)
                                            , n_estimators =  5
                                            , max_samples = 0.7)
          },
    
         {
         'name': 'bagged XGBC 2', 
         'instance': sklearn.ensemble.BaggingClassifier(
                                             xgb.XGBClassifier(max_depth =10
                                                              ,n_estimators =1000)
                                            , n_estimators =  10
                                            , max_samples = 0.7)
          }
])
    
       

# score sometimes crashes the predictors hence, i remade it 



#so far there has been no  big difference herel




kbest = 40
n_comp = 40

kbest = "all"

def parameter_runs(regressions, n_comp = 60):
    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df = pandas.read_csv("training_set_ltv.csv")

    # df = df[df["bookings_pre_first_ride"] <2]
    
    scales = ["StandardScaler"] #, "MinMax", "Robust"]

    tf = data_prep(df)
    
    print(tf["non_cancelled_rides_after_first"].value_counts())
    class_names =  ["return" if x > 0 else "one-time" for x in tf["non_cancelled_rides_after_first"]]


    tf_train, tf_train2, tf_test = data_cleanup.train_validate_test_split(tf
        ,train_percent = 0.8
        ,validate_percent = 0.1
        ,seed =20)


    # X_train, X_test, y_train, y_test, X_columns= data_cleanup.create_test_and_train_data(
    #     tf, "non_cancelled_rides_after_first")

    X_train, y_train, X_columns = data_cleanup.split_xyc(tf_train, "non_cancelled_rides_after_first")
    X_train2, y_train2, X_columns = data_cleanup.split_xyc(tf_train2, "non_cancelled_rides_after_first")
    X_test, y_test, X_columns = data_cleanup.split_xyc(tf_test, "non_cancelled_rides_after_first")

    scale = "StandardScaler"

    if scale == "StandardScaler":
        scaler = StandardScaler()
    elif scaler == "MinMax":
        scaler = MinMaxScaler()
    elif scaler == "Robust":
        scaler = RobustScaler()

    scaler.fit(X_train)  # Don't cheat - fit only on training data
    print (scaler.get_params())
    pickle.dump(scaler, open("data_scaler.pkl", "wb"))
    scaler_filename = "scaler.save"
    joblib.dump(scaler, scaler_filename) 

    # And now to load...
    # print (set(y_train))
    scaler = joblib.load(scaler_filename) 

    X_train = scaler.transform(X_train)


    # pca = PCA(n_components=n_comp, svd_solver='full')
    
    # pca.fit(X_train)
    # X_train = pca.transform(X_train)

    # X_test = pca.transform(scaler.transform(X_test))
    # X_train2 = pca.transform(scaler.transform(X_train2))


    

    selector = SelectKBest(k= kbest).fit(X_train, y_train)

    X_train = selector.transform(X_train)
    X_train2 = selector.transform(X_train2)
    X_test = selector.transform(X_test)

#    X_columns = X_columns[selector.get_support()]

    # create temporary dataframes to add predictions to
    X_train2_t = pandas.DataFrame(X_train2) #, columns = X_columns)
    X_test_t = pandas.DataFrame(X_test)# , columns = X_columns)

    for estimator_conf in regressions:
        input_name = "with_cc_"

        suffix = "_" + estimator_conf['name'].replace(" ", "_")  + "_"  + scale 

        output_name = input_name + suffix
        with open("test_log_" 
            + output_name
            + ".txt", "w") as test_log:


            print(output_name)
            # print (X_train)
            # print(len(X_train), len(X_train[0]))
            # print(len(X_test),len(X_test[0]), len(y_test))

            a= estimator_conf['instance'].fit(X_train, y_train)

            test_log.write(estimator_conf['name'] + "\n")

            output_name = input_name + "_" + estimator_conf['name'].replace(" ", "_")  
            pickle.dump(a, open(output_name + ".pkl", "wb"))
            
            #classification_visualizer(a, X_test, y_test, output_name)            
            test_log.write("score \t " +str(a.score(X_test, y_test)) + "\n")
            test_log.write("train performance\n")
            cnf_matrix =confusion_matrix(y_train, a.predict(X_train))
            test_log.write( str(cnf_matrix) + "\n" )
            test_log.write(str(
                cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis])+ "\n")
            test_log.write("test performance\n")
        
            cnf_matrix =confusion_matrix(y_test, a.predict(X_test))
            test_log.write(str( cnf_matrix) + "\n")
            test_log.write(str(
                (cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]))+ "\n")    

            


            training_manager.add_values_to_table("run_results"
                , [(output_name 
                    ,start_time # what time it was
                    ,str(a.score(X_test, y_test)) # classification score
                    ,int(cnf_matrix[0][0])
                    ,int(cnf_matrix[0][1])
                    ,int(cnf_matrix[1][0])
                    ,int(cnf_matrix[1][1])
                    )] , conn) 

            np.set_printoptions(precision=2)
            
            # Plot non-normalized confusion matrix
            plt.figure()
            plot_confusion_matrix(cnf_matrix, classes=class_names,
                                  title='Confusion matrix, without normalization')
            plt.savefig("cnf_m_" + output_name  + ".pdf")

            # Plot normalized confusion matrix
            plt.figure()
            plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                                  title='Normalized confusion matrix')
            plt.savefig("cnf_m_normed_" + output_name  + ".pdf")


            # print (len(y_proba(a, X_train)))
            # print (len(X_train))
            # print (type(X_train))
            # X_train["prediction_1"] = pandas.Series(y_proba(a, X_train))
            # X_test["prediction_" + output_name]  = y_proba(a, X_test)            



            X_train2_t[output_name + "_prediction"] = y_proba(a, X_train2) 

            X_test_t[output_name + "_prediction"] = y_proba(a, X_test)                 
            #y_test_dict[output_name] = y_proba(a, X_test) 




    good_cols = ["bookings_pre_first_ride"]

    good_cols = X_columns
    print (len(X_train2[0]))
    X_train2 = clean_df(X_train2_t, good_cols)

    print (len(X_train2.columns))
    X_test = clean_df(X_test_t, good_cols)

    for estimator_conf in regressions:

        input_name = "second_with_email_"

        suffix = "_" + estimator_conf['name'].replace(" ", "_")  + "_"  + scale + "_pca_" +str(n_comp)

        output_name = input_name + suffix
        with open("test_log_" 
            + output_name
            + ".txt", "w") as test_log:

            a= estimator_conf['instance'].fit(X_train2, y_train2)

            test_log.write(estimator_conf['name'] + "\n")

            output_name = input_name + "_" + estimator_conf['name'].replace(" ", "_")  
            pickle.dump(a, open(output_name + ".pkl", "wb"))
            
            classification_visualizer(a, X_test, y_test, output_name)            
            test_log.write("score \t " +str(a.score(X_test, y_test)) + "\n")
            test_log.write("train performance\n")
            cnf_matrix =confusion_matrix(y_train2, a.predict(X_train2))
            test_log.write( str(cnf_matrix) + "\n" )
            test_log.write(str(
                cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis])+ "\n")
            test_log.write("test performance\n")

            cnf_matrix =confusion_matrix(y_test, a.predict(X_test))
            test_log.write(str( cnf_matrix) + "\n")
            test_log.write(str(
                (cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]))+ "\n")    

            training_manager.add_values_to_table("run_results"
                , [(output_name 
                    ,start_time # what time it was
                    ,str(a.score(X_test, y_test)) # classification score
                    ,int(cnf_matrix[0][0])
                    ,int(cnf_matrix[0][1])
                    ,int(cnf_matrix[1][0])
                    ,int(cnf_matrix[1][1])
                    )] , conn) 

            np.set_printoptions(precision=2)
            
            # Plot non-normalized confusion matrix
            plt.figure()
            plot_confusion_matrix(cnf_matrix, classes=class_names,
                                  title='Confusion matrix, without normalization')
            plt.savefig("cnf_m_" + output_name  + ".pdf")

            # Plot normalized confusion matrix
            plt.figure()
            plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                                  title='Normalized confusion matrix')
            plt.savefig("cnf_m_normed_" + output_name  + ".pdf")    

    # pickle.dump(y_test_dict, open("y_test_dict.pkl", "wb"))


parameter_runs(regressions)

# kbest_list = ["all", 5, 10, 20, 40, 50, 60 ]

# for kbest in kbest_list:
#     parameter_runs(regressions, kbest)