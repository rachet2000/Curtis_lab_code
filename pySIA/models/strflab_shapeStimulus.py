# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 15:19:45 2017

@author: phil
"""
from curtis_lab_utils import clab_mat  #import matlab.engie before matplotlib.pyplot
from utils import p_utils
import numpy as np
from utils.p_utils import optDef

def kRTACStyle(stim,options,trialSize=375):
    
    #Shape into style used by strflab
    X = stim.reshape(stim.shape[0],np.int(np.sqrt(stim.shape[1])),np.int(np.sqrt(stim.shape[1])),order=options['Reshape_Order'])
    X = X.transpose(1,2,0)
    
    #Set up params dict used by strflab
    params = dict()
    params['class'] = 'preprocRTAC'
    params['covtime']= optDef('covtime',options,0)
    params['covdelays']= optDef('covdelays',options,0)
    params['locality']= np.float32(options['RTAC'][1])
    if options['RTAC'] >0:
        fixedRTAC = [options['RTAC'][0], 1]
    else:
        fixedRTAC = [options['RTAC'][0], 0]
            
    
    params['RTAC']= np.float32(fixedRTAC)

#    params['normalize'] = optDef('normalize',options,0) # shouldn't need to normalize, since it's already done before hand
    params['normalize'] = optDef('normalize',options,1)
    
    stim = clab_mat.preprocRTAC(X,params)
    
    X = X.transpose(2,1,0)  
    X = p_utils.dataDelay(stim,trialSize,options['Frames'])
    #convert to (samples,frames,rows,cols)
#    X = X.reshape(X.shape[0],X.shape[1],np.sqrt(X.shape[2]),np.sqrt(X.shape[2]),order=options['Reshape_Order'])
      
    
    return X
