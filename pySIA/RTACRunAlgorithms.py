# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 14:09:48 2017

@author: phil
"""

from curtis_lab_utils import clab_utils
from algorithms import k_algorithms
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from models import strflab_allModelInfo

def runkRTAC(stim,resp,options):
    print('runkRTAC')
    myModel = strflab_allModelInfo.kRTAC()
    stim = myModel._buildCalcAndScale(options,stim)
    
    X_train,X_reg,X_test,y_train,y_reg,y_test = clab_utils.splitDataSet(stim,resp)
    shapeFunc  = myModel.shapeStimulus()
    X_train,X_reg,X_test = [shapeFunc(x,options) for x in [X_train,X_reg,X_test]]
    options['Input_Shape'] =  myModel._getInputShape(X_train)
    
    #can use Adam to match the conv models, or SGD to match regression
#    result = k_algorithms.k_SIAAdam(myModel,X_train,y_train,X_reg,y_reg,X_test,y_test,options)
    result = k_algorithms.k_SIASGD(myModel,X_train,y_train,X_reg,y_reg,X_test,y_test,options)
    return result