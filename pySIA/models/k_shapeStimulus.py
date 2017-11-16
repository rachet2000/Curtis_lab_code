# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 16:23:41 2016

@author: phil
"""
from utils import p_utils
import numpy as np
from utils import k_utils


def minMax(X,given_axis = None,Xmin = 0,Xmax = 1):
    X_scaled = (X - Xmin) / (Xmax - Xmin)
    return X_scaled
def standardize(X, given_axis = None,given_mean = 0,given_std = 1):

    Xnorm = X - given_mean
    Xnorm = Xnorm / given_std
    Xnorm = np.nan_to_num(Xnorm)
    return Xnorm
def calcStandardize(X,given_axis = None):
    Xmean = np.mean(X,axis=given_axis)
    Xstd = np.std(X,axis=given_axis)
    return Xmean,Xstd
def calcMinMax(X,given_axis = None):
    Xmin = X.min(axis=given_axis)
    Xmax = X.max(axis=given_axis) 
    return Xmin,Xmax
    
    
class scaleStimulusClass(object):
    featureScalingParams = None
    axis = None
    scalingMethod = None
    def __init__(self,options):
        if 'Scaling_Method' in options:
            self.axis = options['Scaling_Method'][1]
            if options['Scaling_Method'][0] == 'Standardize':
                self.scalingCalcMethod = calcStandardize
                self.scalingMethod = standardize
            elif options['Scaling_Method'][0] == 'MinMax':
                self.scalingCalcMethod = calcMinMax
                self.scalingMethod = minMax
        else:
            self.scalingCalcMethod = calcStandardize
            self.scalingMethod = standardize
            self.axis = 0
        return
    def calcScaleParams(self,stim):
        self.featureScalingParams = (self.scalingCalcMethod(stim,self.axis))
        return self.featureScalingParams
    def applyScaleTransform(self,stim):
        assert self.featureScalingParams is not None
        stim = self.scalingMethod(stim,self.axis,*self.featureScalingParams)
        return stim

    
def kRegressionStyle(stim,options,trialSize = 375):
    ''' with standard a_movieClip, the movies are 375 frames long'''

    X = p_utils.dataDelay(stim,trialSize,options['Frames'])

    return X
def kConvNetStyle(stim,options,trialSize = 375):
    ''' with standard a_movieClip, the movies are 375 frames long'''    
    
    X = p_utils.dataDelay(stim,trialSize,options['Frames'])
    X = p_utils.dataDelayAsList(X,len(options['Frames']))
    #convert to (samples,frames,rows,cols)
    X = X.reshape(X.shape[0],X.shape[1],np.int(np.sqrt(X.shape[2])),np.int(np.sqrt(X.shape[2])),order=options['Reshape_Order'])
    
    return X
    
    
    
def kRadialStyle(stim,options,trialSize = 375):
    ''' with standard a_movieClip, the movies are 375 frames long'''
    stim = stim.reshape(stim.shape[0],np.sqrt(stim.shape[1]),np.sqrt(stim.shape[1]))
    X = k_utils.conv2dToRadialProfile(stim,options['Filter_Size'],options['Stride'])
    
    numSubImages = X.shape[1]
    radialProfileLength = X.shape[2]
    
    X = X.reshape(X.shape[0],numSubImages*radialProfileLength)
    X = p_utils.dataDelay(X,375,options['Frames'])
    X = p_utils.dataDelayAsList(X,len(options['Frames']))

    #convert to (samples,frames,rows,cols)
    X = X.reshape(X.shape[0],X.shape[1],numSubImages,radialProfileLength,order=options['Reshape_Order'])
    
    return X
    
def timeSeparateData(X):
    '''additional processing on stimulus for time separable models'''
    X = [ np.expand_dims(X[:,frame,:,:],axis=1) for frame in range(X.shape[1])]
    return X
def timeSeparateDataNoExpand(X):
    '''additional processing on stimulus for time separable models'''
    X = [ X[:,frame,:,:] for frame in range(X.shape[1])]
    return X
    

#def shapeStimulus(stim,trialSize = 375,options,method = None):
#    ''' Reshapes the stimulus to fit the model'''
#    ''' Only works for keras methods for now  '''
#    
#    if method == None:
#        method = p_utils.myInput('model name? :') 
#    
#    
#    kStandardList = ['kRegression']
#    kConvStyleList = ['kConvNet']
#    kRadialStyleList =['kRadialNet']
#    
#    if any(method == listModel for listModel in kStandardList):
#        stim = p_utils.standardize(stim)
#        X = p_utils.dataDelay(stim,trialSize,options['Frames'])
#    elif any(method == listModel for listModel in kConvStyleList):
#        stim = p_utils.standardize(stim)
#        X = p_utils.dataDelay(stim,trialSize,options['Frames'])
#        X = p_utils.dataDelayAsList(X,len(options['Frames']))
#        #convert to (samples,frames,rows,cols)
#        X = X.reshape(X.shape[0],X.shape[1],np.sqrt(X.shape[2]),np.sqrt(X.shape[2]))
#    elif any(method == listModel for listModel in kRadialStyleList):
#        stim = p_utils.standardize(stim)
#        stim = stim.reshape(stim.shape[0],np.sqrt(stim.shape[1]),np.sqrt(stim.shape[1]))
#        X = k_utils.conv2dToRadialProfile(stim,options['Filter_Size'],options['Stride'])
#        
#        numSubImages = X.shape[1]
#        radialProfileLength = X.shape[2]
#        
#        X = X.reshape(X.shape[0],numSubImages*radialProfileLength)
#        X = p_utils.dataDelay(X,375,options['Frames'])
#        X = p_utils.dataDelayAsList(X,len(options['Frames']))
#    
#        #convert to (samples,frames,rows,cols)
#        X = X.reshape(X.shape[0],X.shape[1],numSubImages,radialProfileLength)
#    
#        
#    else:
#        raise Exception('Model unknown, model:' + method)
#        
#    
#    return X