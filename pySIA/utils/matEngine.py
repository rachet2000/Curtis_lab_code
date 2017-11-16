# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 23:10:42 2016

@author: phil
"""

import matlab.engine
eng = matlab.engine.start_matlab()
import scipy.io
import numpy as np
from utils import p_utils
import os

def mat2Gray(taper):
    return eng.mat2gray(taper)
def mkCosTaper(imgSize,taperSize):
    return eng.mkCosTaper(imgSize,taperSize)
def linkGratingFromMovie(neuronName,grating):
    return eng.linkGratingFromMovie(neuronName,grating)

def generateQuickStimuli(fileName,gratingName):
    stim,info = eng.generateQuickStimuli(fileName,gratingName,nargout = 2)
    stimList = [] 
    for subStim in stim:

         newStim = np.array(subStim._data,dtype=np.float32).reshape(subStim.size[::-1]).T
         newStim = newStim.transpose(2,0,1)
         stimList.append(newStim)
    return stimList,info

def matlabResize(movie,cropKern):
   
    return eng.imresize(matlab.double(movie.tolist()),matlab.double([cropKern,cropKern]))
def createStrfMovie(movieName,cropKern,cropWin):
    stimMovie = eng.createSTRFMOVIE(movieName,cropKern,0,matlab.double(cropWin.tolist()))
    return np.array(stimMovie._data,dtype=np.float32).reshape(stimMovie.size[::-1])


#if __name__ == '__main__':
#
#    bb = generateQuickStimuli('H5302.022','sft')