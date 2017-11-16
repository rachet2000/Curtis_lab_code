# -*- coding: utf-8 -*-
"""
Created on Tue May 17 12:29:05 2016

@author: phil
"""

from utils import p_utils
from curtis_lab_utils import clab_utils
from algorithms import p_algorithms
from algorithms import k_algorithms
import sys
import os
import scipy.signal
import numpy as np
from scipy.signal import gaussian
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from skimage.transform import resize
from models import k_allModelInfo

    
def stimDownSamp(stim,newLength):
    stim= stim.reshape((stim.shape[0],np.sqrt(stim.shape[1]),np.sqrt(stim.shape[1])))  

    stim = stim.transpose((1,2,0))
    stim = resize(stim,(newLength,newLength))    
#    stim = np.transpose(np.reshape(stim,(newLength*newLength,np.size(stim,2))))
    stim = stim.transpose((2,0,1))
    stim = np.reshape(stim,(np.size(stim,0),newLength*newLength))
    return stim

def pullInitWeightsFromSTA(STA_result,numFrames):
    STA_weights = STA_result['model']['weights'][0]
    STA_weights = np.reshape(STA_weights,(numFrames,int(STA_weights.size/numFrames)))
    bestLag = np.argmax(np.var(STA_weights,axis=1))
    bestWeights = STA_weights[bestLag]
    maxDot = np.dot(bestWeights,bestWeights)
    initMapWeights = p_utils.standardize(bestWeights)
    
    initTempList = np.zeros((STA_weights.shape[0]))
    for lag in range(STA_weights.shape[0]):
        initTempList[lag] =np.dot(bestWeights,STA_weights[lag,:])/maxDot       
    
    meanVal = np.mean(bestWeights)
        
    return initTempList,initMapWeights,meanVal


def runKRegression(stim,resp,options):
    print('runKRegression')
    
    myModel = k_allModelInfo.kRegression()
    stim = myModel._buildCalcAndScale(options,stim)
    
    X_train,X_reg,X_test,y_train,y_reg,y_test = clab_utils.splitDataSet(stim,resp)
    shapeFunc  = myModel.shapeStimulus()
    X_train,X_reg,X_test = [shapeFunc(x,options) for x in [X_train,X_reg,X_test]]
    options['Input_Shape'] =  myModel._getInputShape(X_train)

    #SGD is usually better than ADAM for the regression model
    result = k_algorithms.k_SIASGD(myModel,X_train,y_train,X_reg,y_reg,X_test,y_test,options)
    
    return result



def runKConvNet(stim,resp,options):
    print('runKConvNet')
        
    myModel = k_allModelInfo.kConvNet()
    stim = myModel._buildCalcAndScale(options,stim)

    X_train,X_reg,X_test,y_train,y_reg,y_test = clab_utils.splitDataSet(stim,resp)
    shapeFunc = myModel.shapeStimulus()
    X_train,X_reg, X_test = [shapeFunc(x,options) for x in [X_train,X_reg,X_test]]
    options['Input_Shape'] =  myModel._getInputShape(X_train)
#    
#    initFilter = gaussian(options['Filter_Size'],np.sqrt(options['Filter_Size']))
#    tempGauss = np.random.rand(len(options['Frames']))
#    initFilter = np.einsum('i,j,k->ijk',tempGauss,initFilter,initFilter)
#    initFilter = np.expand_dims(initFilter,0)
#    options['Initial_Filter_Weights'] = [initFilter/np.float(options['Filter_Size']**2),np.asarray((0.0,))]
    
    result = k_algorithms.k_SIAAdam(myModel,X_train,y_train,X_reg,y_reg,X_test,y_test,options)
    
    return result
def runKConvGaussNet(stim,resp,options):
    print('runKConvGaussNet')        
    
    myModel = k_allModelInfo.kConvGaussNet()
    stim = myModel._buildCalcAndScale(options,stim)

    X_train,X_reg,X_test,y_train,y_reg,y_test = clab_utils.splitDataSet(stim,resp)
    shapeFunc  = myModel.shapeStimulus()
    X_train,X_reg, X_test = [shapeFunc(x,options) for x in [X_train,X_reg,X_test]]
    options['Input_Shape'] =  myModel._getInputShape(X_train)

#    initFilter = gaussian(options['Filter_Size'],np.sqrt(options['Filter_Size']))
#    tempGauss = np.random.rand(len(options['Frames']))
#    initFilter = np.einsum('i,j,k->ijk',tempGauss,initFilter,initFilter)
#    initFilter = np.expand_dims(initFilter,0)
#    options['Initial_Filter_Weights'] = [initFilter/np.float(options['Filter_Size']**2),np.asarray((0.0,))]




    result = k_algorithms.k_SIAAdam(myModel,X_train,y_train,X_reg,y_reg,X_test,y_test,options)
    
    return result
def runKConvDOGNet(stim,resp,options):
    print('runKConvDOGNet')
    
    myModel = k_allModelInfo.kConvDOGNet()
    stim = myModel._buildCalcAndScale(options,stim)

    X_train,X_reg,X_test,y_train,y_reg,y_test = clab_utils.splitDataSet(stim,resp)
    shapeFunc  = myModel.shapeStimulus()
    X_train,X_reg, X_test = [shapeFunc(x,options) for x in [X_train,X_reg,X_test]]
    options['Input_Shape'] =  myModel._getInputShape(X_train)

    result = k_algorithms.k_SIAAdam(myModel,X_train,y_train,X_reg,y_reg,X_test,y_test,options)
    
    return result
def runKConvGaussEXP(stim,resp,options):
    print('runKConvGaussEXP')        
    
    myModel = k_allModelInfo.kConvGaussEXP()
    stim = myModel._buildCalcAndScale(options,stim)

    X_train,X_reg,X_test,y_train,y_reg,y_test = clab_utils.splitDataSet(stim,resp)
    shapeFunc  = myModel.shapeStimulus()
    X_train,X_reg, X_test = [shapeFunc(x,options) for x in [X_train,X_reg,X_test]]
    options['Input_Shape'] =  myModel._getInputShape(X_train)

    result = k_algorithms.k_SIAAdam(myModel,X_train,y_train,X_reg,y_reg,X_test,y_test,options)
    
    return result
def runKConvTwoGaussNet(stim,resp,options):
    print('runKConvTwoGaussNet')
    
    myModel = k_allModelInfo.kConvTwoGaussNet()
    stim = myModel._buildCalcAndScale(options,stim)

    X_train,X_reg,X_test,y_train,y_reg,y_test = clab_utils.splitDataSet(stim,resp)
    shapeFunc  = myModel.shapeStimulus()
    X_train,X_reg, X_test = [shapeFunc(x,options) for x in [X_train,X_reg,X_test]]
    options['Input_Shape'] =  myModel._getInputShape(X_train)

    result = k_algorithms.k_SIAAdam(myModel,X_train,y_train,X_reg,y_reg,X_test,y_test,options)
    
    return result
def runKConvTwoAffineNet(stim,resp,options):
    print('runKConvTwoAffineNet')
    
    myModel = k_allModelInfo.kConvTwoAffineNet()
    stim = myModel._buildCalcAndScale(options,stim)
    
    X_train,X_reg,X_test,y_train,y_reg,y_test = clab_utils.splitDataSet(stim,resp)
    shapeFunc  = myModel.shapeStimulus()
    X_train,X_reg, X_test = [shapeFunc(x,options) for x in [X_train,X_reg,X_test]]
    options['Input_Shape'] =  myModel._getInputShape(X_train)

    result = k_algorithms.k_SIAAdam(myModel,X_train,y_train,X_reg,y_reg,X_test,y_test,options)
    
    return result


    

def runKConvNetTimeSeparable(stim,resp,options):
    print('runKConvNetTimeSeparable')
    #Usually better to run runKSTAintoConvNetTimeSeparable, since this is a
    #very layered model, need good initial conditions
    
    myModel = k_allModelInfo.kConvNetTimeSeparable()
    stim = myModel._buildCalcAndScale(options,stim)
    
    X_train,X_reg,X_test,y_train,y_reg,y_test = clab_utils.splitDataSet(stim,resp)
    shapeFunc  = myModel.shapeStimulus()
    X_train,X_reg, X_test = [shapeFunc(x,options) for x in [X_train,X_reg,X_test]]
    options['Input_Shape'] =  myModel._getInputShape(X_train)

    result = k_algorithms.k_SIAAdam(myModel,X_train,y_train,X_reg,y_reg,X_test,y_test,options)
    
    return result

def runKSTAintoConvNetTimeSeparableDOG(stim,resp,options):
    print('runKSTAintoConvNetTimeSeparableDOG')   
    from k_utils import conv_output_length
    convImagelength = conv_output_length(np.sqrt(stim.shape[1]),options['Filter_Size'],'valid',options['Stride'])
    mapLength = conv_output_length(convImagelength,options['Pool_Size'],'valid',options['Pool_Size'])

    #STA portion   
    stim_STA = stimDownSamp(stim,mapLength)    
    STA_result = runKRegression(stim_STA,resp,options)   
    initTempList,initMapWeights,meanVal = pullInitWeightsFromSTA(STA_result,len(options['Frames']))
    initFilter = gaussian(options['Filter_Size'],np.sqrt(options['Filter_Size'])/2)
    initFilter = np.outer(initFilter,initFilter) 
    initFilter = np.expand_dims(np.expand_dims(initFilter,0),0)    
    if 'Check_On_Off' in options and options['Check_On_Off'] is True:
        if meanVal < 0:
            initMapWeights = -initMapWeights
            
    options['Initial_Filter_Weights'] = [np.float32(initFilter/np.float(options['Filter_Size']**2)),np.asarray((0.0,),dtype=np.float32)]
    options['Initial_Temporal_Weights']=np.float32(initTempList/np.float(len(options['Frames'])))
    
    #TimeSep portion 
    myModel = k_allModelInfo.kConvNetTimeSeparableDOG() 
    stim = myModel._buildCalcAndScale(options,stim)
    
    X_train,X_reg,X_test,y_train,y_reg,y_test = clab_utils.splitDataSet(stim,resp)
    shapeFunc  = myModel.shapeStimulus()
    X_train,X_reg, X_test = [shapeFunc(x,options) for x in [X_train,X_reg,X_test]]
    options['Input_Shape'] =  myModel._getInputShape(X_train)
    
    result = k_algorithms.k_SIAAdam(myModel,X_train,y_train,X_reg,y_reg,X_test,y_test,options)
    
    return result
def runKSTAintoConvNetTimeSeparable(stim,resp,options):
    print('runKSTAintoConvNetTimeSeparable')
    from k_utils import conv_output_length

    #build stim for both STA and radialNet portion
    
    convImagelength = conv_output_length(np.sqrt(stim.shape[1]),options['Filter_Size'],'valid',options['Stride'])
    mapLength = conv_output_length(convImagelength,options['Pool_Size'],'valid',options['Pool_Size'])

    
    #STA portion   
    stim_STA = stimDownSamp(stim,mapLength)   
    STA_result = runKRegression(stim_STA,resp,options)   
    initTempList,initMapWeights,meanVal = pullInitWeightsFromSTA(STA_result,len(options['Frames']))
        
    #initial radial Filter
    initFilter = gaussian(options['Filter_Size'],np.sqrt(options['Filter_Size'])/2)
    initFilter = np.outer(initFilter,initFilter)   
    if 'Check_On_Off' in options and options['Check_On_Off'] is True:
        if meanVal < 0:
            initFilter = -initFilter
            initMapWeights = -initMapWeights
    

    

    initFilter = np.expand_dims(np.expand_dims(initFilter,0),0)
    options['Initial_Filter_Weights'] = [np.float32(initFilter/np.float(options['Filter_Size']**2)),np.asarray((0.0,),dtype=np.float32)]
    options['Initial_Temporal_Weights']=np.float32(initTempList/np.float(len(options['Frames'])))
    options['Initial_Map_Weights'] = [np.expand_dims(initMapWeights,axis=1)/np.float32(options['Filter_Size']**2), np.asarray((0.0,),dtype=np.float32)]

        
    myModel = k_allModelInfo.kConvNetTimeSeparable()
    stim = myModel._buildCalcAndScale(options,stim)
    
    X_train,X_reg,X_test,y_train,y_reg,y_test = clab_utils.splitDataSet(stim,resp)
    shapeFunc  = myModel.shapeStimulus()
    X_train,X_reg, X_test = [shapeFunc(x,options) for x in [X_train,X_reg,X_test]]
    options['Input_Shape'] =  myModel._getInputShape(X_train)

    result = k_algorithms.k_SIAAdam(myModel,X_train,y_train,X_reg,y_reg,X_test,y_test,options)
        
    return result


def runKRadialNet(stim,resp,options=None,mode = 'user'):
    ''' Will need to run kSTA first in order to initialize map layer '''
    print('runKRadialNet')
    from k_utils import conv_output_length

    myModel = k_allModelInfo.kRadialNet()
    stim = myModel._buildCalcAndScale(options,stim)
    
    X_train,X_reg,X_test,y_train,y_reg,y_test = clab_utils.splitDataSet(stim,resp)
    shapeFunc  = myModel.shapeStimulus()
    X_train,X_reg, X_test = [shapeFunc(x,options) for x in [X_train,X_reg,X_test]]

    
    convImagelength = np.sqrt(X_train[0].shape[1])
    mapLength = conv_output_length(convImagelength,options['Pool_Size'],'valid',options['Pool_Size'])
    

    
    #STA portion
    stim_STA = stimDownSamp(stim,mapLength)   
    STA_result = runKRegression(stim_STA,resp,options)   
    initTempList,initMapWeights,meanVal = pullInitWeightsFromSTA(STA_result,len(options['Frames']))
    
    if 'Check_On_Off' in options and options['Check_On_Off'] is True:
        if meanVal < 0:
            initFilter = -initFilter
            initMapWeights = -initMapWeights


            
    options['Input_Shape'] =  myModel._getInputShape(X_train)
    
    #initial radial Filter
    initFilter = gaussian(X_train[0].shape[2]*2,np.sqrt(X_train[0].shape[2])/2) 
    initFilter = initFilter[:X_train[0].shape[2]]
    initFilter = initFilter[::-1]
    #use this initial STA estimate as the map weights for CNN,normalize by the filtersize
    initFilter = np.expand_dims(np.expand_dims(np.expand_dims(initFilter,0),2),3)
    options['Initial_Filter_Weights'] = [np.float32(initFilter/np.float(options['Filter_Size']**2)),np.asarray((0.0,),dtype=np.float32)]
    options['Initial_Temporal_Weights']=np.float32(initTempList/np.float(len(options['Frames'])))
    options['Initial_Map_Weights'] = [np.expand_dims(initMapWeights,axis=1)/np.float(options['Filter_Size']**2), np.asarray((0.0,))]
    

    result = k_algorithms.k_SIAAdam(myModel,X_train,y_train,X_reg,y_reg,X_test,y_test,options)
        

    
    return result
    

def runKRadialNetDOG(stim,resp,options=None,mode = 'user'):
    ''' Will need to run kSTA first in order to initialize map layer '''
    print('runKRadialNetDOG')
    from k_utils import conv_output_length

    myModel = k_allModelInfo.kRadialNetDOG()
    stim = myModel._buildCalcAndScale(options,stim)
    
    X_train,X_reg,X_test,y_train,y_reg,y_test = clab_utils.splitDataSet(stim,resp)
    shapeFunc  = myModel.shapeStimulus()
    X_train,X_reg, X_test = [shapeFunc(x,options) for x in [X_train,X_reg,X_test]]

    
    convImagelength = np.sqrt(X_train[0].shape[1])
    mapLength = conv_output_length(convImagelength,options['Pool_Size'],'valid',options['Pool_Size'])
    

    
    #STA portion
    stim_STA = stimDownSamp(stim,mapLength)   
    STA_result = runKRegression(stim_STA,resp,options)   
    initTempList,initMapWeights,meanVal = pullInitWeightsFromSTA(STA_result,len(options['Frames']))
    
            
    options['Input_Shape'] =  myModel._getInputShape(X_train)
    
    #initial radial Filter
    initFilter = gaussian(X_train[0].shape[2]*2,np.sqrt(X_train[0].shape[2])/2) 
    initFilter = initFilter[:X_train[0].shape[2]]
    initFilter = initFilter[::-1]
    #use this initial STA estimate as the map weights for CNN,normalize by the filtersize
    initFilter = np.expand_dims(np.expand_dims(np.expand_dims(initFilter,0),2),3)
    options['Initial_Filter_Weights'] = [np.float32(initFilter/np.float(options['Filter_Size']**2)),np.asarray((0.0,),dtype=np.float32)]
    options['Initial_Temporal_Weights']=np.float32(initTempList/np.float(len(options['Frames'])))
    

    result = k_algorithms.k_SIAAdam(myModel,X_train,y_train,X_reg,y_reg,X_test,y_test,options)
        

    
    
    return result
    
def runKConvSplitNet(stim,resp,options):
    print('runKConvSplitNet')
    
    myModel = k_allModelInfo.kConvSplitNet()
    stim = myModel._buildCalcAndScale(options,stim)
    
    X_train,X_reg,X_test,y_train,y_reg,y_test = clab_utils.splitDataSet(stim,resp)
    shapeFunc = myModel.shapeStimulus()
    X_train,X_reg, X_test = [shapeFunc(x,options) for x in [X_train,X_reg,X_test]]
    options['Input_Shape'] =  myModel._getInputShape(X_train)

    result = k_algorithms.k_SIAAdam(myModel,X_train,y_train,X_reg,y_reg,X_test,y_test,options)
    
    return result

    
def runKOnOffConvNet(stim,resp,options=None,mode = 'user'):
    print('runKOnOffConvNet')
    myModel = k_allModelInfo.kOnOffConvNet()
    stim = myModel._buildCalcAndScale(options,stim)
    
    X_train,X_reg,X_test,y_train,y_reg,y_test = clab_utils.splitDataSet(stim,resp)
    shapeFunc = myModel.shapeStimulus()
    X_train,X_reg, X_test = [shapeFunc(x,options) for x in [X_train,X_reg,X_test]]
    options['Input_Shape'] =  myModel._getInputShape(X_train)

    result = k_algorithms.k_SIAAdam(myModel,X_train,y_train,X_reg,y_reg,X_test,y_test,options)
    
    return result
########################################
### OLD METHODS BUILT IN PURE THEANO ###
### NOT MAINTAINED                   ###
########################################

def runLasso(stim,resp,options=None,mode = 'user'):
    print('runLasso')
    if options ==None:
        options = dict()
        options['lambdas'] =p_utils.userQuery('lambdas', 'list', [0], mode)
    
    
    stim = p_utils.standardize(stim)    
    X = p_utils.dataDelay(stim,375,list(range(1)))    
    X_train,X_reg,X_test,y_train,y_reg,y_test = clab_utils.splitDataSet(X,resp)

        
    result = p_algorithms.lassoWithDelay(X_train,y_train,X_test,y_test,options)
    
    
    return result
def runSTARegression(stim,resp,options=None,mode = 'user'):
    print('runSTARegression')
    if options ==None:
        options = dict()      
        options['Batch_Size'] = p_utils.userQuery('Batch_Size', 'int', 375, mode)
        options['Learning_Rate'] = p_utils.userQuery('Learning_Rate', 'float', 0.01, mode)
        options['N_Epochs'] = p_utils.userQuery('N_Epochs', 'int', 500, mode)
        options['N_Kern'] = p_utils.userQuery('N_Kern', 'int', 1, mode)
        options['Frames'] = p_utils.userQuery('Frames List', 'list', list(range(8)), mode)
        options['L1'] = p_utils.userQuery('L1', 'float', 0, mode)
        options['L2'] = p_utils.userQuery('L2', 'float', 0, mode)
        options['Filter_Size'] = p_utils.userQuery('Filter Size', 'int', 12, mode)
        options['Pool_Size'] = p_utils.userQuery('Pool Size', 'int', 2, mode)
        options['Mid_Activation_Mode'] = p_utils.userQuery('Mid Activation Mode', 'int', 0, mode)
    stim = p_utils.standardize(stim)
    X = p_utils.dataDelay(stim,375,options['Frames'])
    X_train,X_reg,X_test,y_train,y_reg,y_test = clab_utils.splitDataSet(X,resp)
        
    result = p_algorithms.staRegression(X_train,y_train,X_reg,y_reg,X_test,y_test,options)
    
    return result        
    
def runAlphaNet(stim,resp,options=None,mode = 'user'):
    print('runAlphaNet')
    if options ==None:
        options = dict()      
        options['Batch_Size'] = p_utils.userQuery('Batch_Size', 'int', 375, mode)
        options['Learning_Rate'] = p_utils.userQuery('Learning_Rate', 'float', 0.01, mode)
        options['N_Epochs'] = p_utils.userQuery('N_Epochs', 'int', 500, mode)
        options['N_Kern'] = p_utils.userQuery('N_Kern', 'int', 1, mode)
        options['Frames'] = p_utils.userQuery('Frames List', 'list', list(range(8)), mode)
        options['L1'] = p_utils.userQuery('L1', 'float', 0, mode)
        options['L2'] = p_utils.userQuery('L2', 'float', 0, mode)
        options['Filter_Size'] = p_utils.userQuery('Filter Size', 'int', 12, mode)
        options['Pool_Size'] = p_utils.userQuery('Pool Size', 'int', 2, mode)
        options['Mid_Activation_Mode'] = p_utils.userQuery('Mid Activation Mode', 'int', 0, mode)
    stim = p_utils.standardize(stim)
    X = p_utils.dataDelay(stim,375,options['Frames'])
    X_train,X_reg,X_test,y_train,y_reg,y_test = clab_utils.splitDataSet(X,resp)
        
    result = p_algorithms.alphaNet(X_train,y_train,X_reg,y_reg,X_test,y_test,options)
    
    return result 
def runAlphaConvNet(stim,resp,options=None,mode = 'user'):
    print('runAlphaConvNet')
    if options ==None:
        options = dict()      
        options['Batch_Size'] = p_utils.userQuery('Batch_Size', 'int', 375, mode)
        options['Learning_Rate'] = p_utils.userQuery('Learning_Rate', 'float', 0.01, mode)
        options['N_Epochs'] = p_utils.userQuery('N_Epochs', 'int', 500, mode)
        options['N_Kern'] = p_utils.userQuery('N_Kern', 'int', 1, mode)
        options['Frames'] = p_utils.userQuery('Frames List', 'list', list(range(8)), mode)
        options['L1'] = p_utils.userQuery('L1', 'float', 0, mode)
        options['L2'] = p_utils.userQuery('L2', 'float', 0, mode)
        options['Filter_Size'] = p_utils.userQuery('Filter Size', 'int', 12, mode)
        options['Pool_Size'] = p_utils.userQuery('Pool Size', 'int', 2, mode)
        options['Mid_Activation_Mode'] = p_utils.userQuery('Mid Activation Mode', 'int', 0, mode)
    stim = p_utils.standardize(stim)
    X = p_utils.dataDelay(stim,375,options['Frames'])
    X = p_utils.dataDelayAsList(X,len(options['Frames']))
    X_train,X_reg,X_test,y_train,y_reg,y_test = clab_utils.splitDataSet(X,resp)
        
    result = p_algorithms.alphaConvNet(X_train,y_train,X_reg,y_reg,X_test,y_test,options)
    
    return result
    
def runPPAConvNet(stim,resp,options=None,mode = 'user'):
    print('runPPAConvNet')
    if options ==None:
        options = dict()      
        options['Batch_Size'] = p_utils.userQuery('Batch_Size', 'int', 375, mode)
        options['Learning_Rate'] = p_utils.userQuery('Learning_Rate', 'float', 0.01, mode)
        options['N_Epochs'] = p_utils.userQuery('N_Epochs', 'int', 500, mode)
        options['N_Kern'] = p_utils.userQuery('N_Kern', 'int', 1, mode)
        options['Frames'] = p_utils.userQuery('Frames List', 'list', list(range(8)), mode)
        options['L1'] = p_utils.userQuery('L1', 'float', 0, mode)
        options['L2'] = p_utils.userQuery('L2', 'float', 0, mode)
        options['Filter_Size'] = p_utils.userQuery('Filter Size', 'int', 12, mode)
        options['Pool_Size'] = p_utils.userQuery('Pool Size', 'int', 2, mode)
        options['Mid_Activation_Mode'] = p_utils.userQuery('Mid Activation Mode', 'int', 0, mode)
    stim = p_utils.standardize(stim)
    X = p_utils.dataDelay(stim,375,options['Frames'])
    X = p_utils.dataDelayAsList(X,len(options['Frames']))
    X_train,X_reg,X_test,y_train,y_reg,y_test = clab_utils.splitDataSet(X,resp)
        
    result = p_algorithms.PPAConvNet(X_train,y_train,X_reg,y_reg,X_test,y_test,options)
    
    return result

def runConvNet(stim,resp,options=None,mode = 'user'):
    print('runConvNet')
    if options ==None:
        options = dict()      
        options['Batch_Size'] = p_utils.userQuery('Batch_Size', 'int', 375, mode)
        options['Learning_Rate'] = p_utils.userQuery('Learning_Rate', 'float', 0.01, mode)
        options['N_Epochs'] = p_utils.userQuery('N_Epochs', 'int', 500, mode)
        options['N_Kern'] = p_utils.userQuery('N_Kern', 'int', 1, mode)
        options['Frames'] = p_utils.userQuery('Frames List', 'list', list(range(8)), mode)
        options['L1'] = p_utils.userQuery('L1', 'float', 0, mode)
        options['L2'] = p_utils.userQuery('L2', 'float', 0, mode)
        options['Filter_Size'] = p_utils.userQuery('Filter Size', 'int', 12, mode)
        options['Pool_Size'] = p_utils.userQuery('Pool Size', 'int', 2, mode)
        options['Mid_Activation_Mode'] = p_utils.userQuery('Mid Activation Mode', 'int', 0, mode)
    stim = p_utils.standardize(stim)
    X = p_utils.dataDelay(stim,375,options['Frames'])
    X = p_utils.dataDelayAsList(X,len(options['Frames']))
    X_train,X_reg,X_test,y_train,y_reg,y_test = clab_utils.splitDataSet(X,resp)
        
    result = p_algorithms.convNet(X_train,y_train,X_reg,y_reg,X_test,y_test,options)
    
    return result
def runConvNetStack(stim,resp,options=None,mode = 'user'):
    print('runConvNetStack')
    if options ==None:
        options = dict()      
        options['Batch_Size'] = p_utils.userQuery('Batch_Size', 'int', 375, mode)
        options['Learning_Rate'] = p_utils.userQuery('Learning_Rate', 'float', 0.01, mode)
        options['N_Epochs'] = p_utils.userQuery('N_Epochs', 'int', 500, mode)
        options['N_Kern'] = p_utils.userQuery('N_Kern', 'int', 1, mode)
        options['Frames'] = p_utils.userQuery('Frames List', 'list', list(range(8)), mode)
    stim = p_utils.standardize(stim)
    X = p_utils.dataDelay(stim,375,options['Frames'])
    X = p_utils.dataDelayAsList(X,len(options['Frames']))
    X_train,X_reg,X_test,y_train,y_reg,y_test = clab_utils.splitDataSet(X,resp)
        
    result = p_algorithms.convNetStack(X_train,y_train,X_reg,y_reg,X_test,y_test,options)
    
    return result
    
def runConvLSTM(stim,resp,options=None,mode = 'user'):
    print('runConvLSTM')
    if options ==None:
        options = dict()      
        options['Batch_Size'] = p_utils.userQuery('Batch_Size', 'int', 375, mode)
        options['Learning_Rate'] = p_utils.userQuery('Learning_Rate', 'float', 0.01, mode)
        options['N_Epochs'] = p_utils.userQuery('N_Epochs', 'int', 500, mode)
        options['N_Kern'] = p_utils.userQuery('N_Kern', 'int', 1, mode)
        options['Frames'] = p_utils.userQuery('Frames List', 'list', list(range(8)), mode)
        options['L1'] = p_utils.userQuery('L1', 'float', 0, mode)
        options['L2'] = p_utils.userQuery('L2', 'float', 0, mode)
        options['Filter_Size'] = p_utils.userQuery('Filter Size', 'int', 12, mode)
        options['Pool_Size'] = p_utils.userQuery('Pool Size', 'int', 2, mode)
        options['Mid_Activation_Mode'] = p_utils.userQuery('Mid Activation Mode', 'int', 0, mode)
    stim = p_utils.standardize(stim)
    X = p_utils.dataDelay(stim,375,options['Frames'])
    X = p_utils.dataDelayAsList(X,len(options['Frames']))
    X_train,X_reg,X_test,y_train,y_reg,y_test = clab_utils.splitDataSet(X,resp)
        
    result = k_algorithms.convLSTM(X_train,y_train,X_reg,y_reg,X_test,y_test,options)
    
    return result