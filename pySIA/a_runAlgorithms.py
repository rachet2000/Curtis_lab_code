# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 13:08:39 2017

@author: amol
"""


from algorithms import k_algorithms
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from models import a_allModelInfo
from curtis_lab_utils import clab_utils

def runKConvNet(stim,resp,options):
    print('runKConvNet')
        
    myModel = a_allModelInfo.kConvNet()
    stim = myModel._buildCalcAndScale(options,stim)

    X_train,X_reg,X_test,y_train,y_reg,y_test = clab_utils.splitDataSet(stim,resp)
    shapeFunc = myModel.shapeStimulus()
    X_train,X_reg, X_test = [shapeFunc(x,options) for x in [X_train,X_reg,X_test]]
    options['Input_Shape'] =  myModel._getInputShape(X_train)   
    result = k_algorithms.k_SIAAdam(myModel,X_train,y_train,X_reg,y_reg,X_test,y_test,options)
    
    return result

def runKConvNetDropOut(stim,resp,options):
    print('runKConvNetDropOut')
        
    myModel = a_allModelInfo.kConvNetDropout()
    stim = myModel._buildCalcAndScale(options,stim)

    X_train,X_reg,X_test,y_train,y_reg,y_test = clab_utils.splitDataSet(stim,resp)
    shapeFunc = myModel.shapeStimulus()
    X_train,X_reg, X_test = [shapeFunc(x,options) for x in [X_train,X_reg,X_test]]
    options['Input_Shape'] =  myModel._getInputShape(X_train)   
    result = k_algorithms.k_SIAOptimizer(myModel,X_train,y_train,X_reg,y_reg,X_test,y_test,options)
    
    return result
    
def runKConvNetTimeSeparable(stim,resp,options):
    print('runKConvNetTimeSeparable')
    
    myModel = a_allModelInfo.kConvNetTimeSeparable()
    stim = myModel._buildCalcAndScale(options,stim)

    X_train,X_reg,X_test,y_train,y_reg,y_test = clab_utils.splitDataSet(stim,resp)
    shapeFunc = myModel.shapeStimulus()
    X_train,X_reg, X_test = [shapeFunc(x,options) for x in [X_train,X_reg,X_test]]
    options['Input_Shape'] =  myModel._getInputShape(X_train)   
    result = k_algorithms.k_SIAAdam(myModel,X_train,y_train,X_reg,y_reg,X_test,y_test,options)
    
    return result
    

    
    
    
    
    
