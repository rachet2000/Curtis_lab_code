# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 15:07:05 2017

@author: phil
"""

import matlab.engine
eng = matlab.engine.start_matlab()
import scipy.io
import numpy as np
from curtis_lab_utils import clab_utils
import os

def preprocRTAC(stim,params):
    print('This function uses temp data, will likely bug if used in multiple programs at once')
#    stim = eng.classWrapper(matlab.double(stim.tolist()),params)
#    matStim = matlab.double(stim.tolist())
#    newStim = np.zeros(stim.shape[0],stim.shape[1]*stim.shape[1] )
#    for imgIdx in range(np.size(stim,axis = 0)):
#        if imgIdx % 100 ==0:
#            print(imgIdx)
#        subStim = matStim[imgIdx]
#        aa = eng.preprocRTAC(subStim,params)
#        newStim[imgIdx] = aa
#
#    stim = eng.classWrapper(matlab.double(stim.tolist()))
    
    paths = clab_utils.getPaths()
    tempLocation = paths['tempLocation']
    os.chdir(tempLocation)
    
    tempFileName = 'preprocRTAC_tempData'
    preprocRTACDict = dict()
    preprocRTACDict['stim'] = stim
    preprocRTACDict['params'] = params

    scipy.io.savemat(tempFileName,preprocRTACDict)

    newStim = eng.tempPreprocRTAC(tempFileName)
    newStim = np.array(newStim._data,dtype=np.float32).reshape(newStim.size[::-1]).T
    return newStim