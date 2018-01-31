# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 12:10:04 2018

@author: fbaker
"""

import random
import pylab

def proba_splitter(clf, X, y):
    
    try:
        y_proba = clf.predict_proba(X)
    except AttributeError:
        y_proba = clf.predict(X)     
    classes = set(y)
    # need a list here, not a generator as it will be used twice
    ylist = list(zip(y, y_proba))

    probas = {}
    
    for cl in classes:
        probas[cl] = [y_proba_i[1] for  y_i, y_proba_i in ylist if int(y_i)==int(cl)]

    return probas



def proba_plotter(proba_dict, name,n_bins =40):
    

    for cl in proba_dict:

        color = [random.random(), random.random(), random.random()]
        y_list = proba_dict[cl]
        pylab.hist(y_list
                    , bins = n_bins 
                    , label = str(cl)
                    , color = color
                    , histtype = "step"
                    #, cumulative = True
                    )

    pylab.legend()
    pylab.ylabel("Counts #")
    pylab.xlabel("predicted class probability")
    pylab.tight_layout()
    pylab.savefig("class_probability_" + name + ".pdf")
    pylab.clf()


def classification_visualizer(clf, X, y, name):

    proba_dict = proba_splitter(clf, X, y)
    proba_plotter(proba_dict, name)