# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 15:01:50 2016

@author: phil
"""
import numpy as np
import matEngine

def createDriftingGrating(lmbda,ori,tf,phase,winOptions = None):
    if winOptions == None:
        winOptions = dict()
        winOptions['bkg'] = 127
        winOptions['inc'] = 126
        winOptions['winSize'] = 480
        winOptions['refreshHz'] = 150
        
        
    if tf!=0:      
        nImages = int(np.round(winOptions['refreshHz']/tf))
    elif tf==0:
        nImages = 1
        


    phaseArrayDeg = np.zeros((nImages))

    for iImage in xrange(nImages):
        phaseArrayDeg[iImage] = (360*(iImage)/nImages) +phase          

                                        

    phaseArrayDeg = phaseArrayDeg %360
    phaseArrayRad = phaseArrayDeg * (2*np.pi/360);
   
    
    
    sf = 2*np.pi/lmbda
    a = np.cos(ori*np.pi/180)*sf
    b = np.sin(ori*np.pi/180)*sf
    
    wSize = winOptions['winSize']
    thisMovie = np.zeros((wSize,wSize,nImages))
    [x,y] = np.meshgrid(range(-wSize/2,wSize/2),range(-wSize/2,wSize/2))
    
    for iImage in range(nImages):
        thisMovie[:,:,iImage] = np.sin(a*x+b*y+phaseArrayRad[iImage])
        
    return thisMovie
    
    
def cosTaperedGrating(lmbda,ori,tf,phase=0,taperSize =0.3,winOptions = None):
    if winOptions == None:
        winOptions = dict()
        winOptions['bkg'] = 127
        winOptions['inc'] = 126
        winOptions['winSize'] = 480
        winOptions['refreshHz'] = 150
    thisMovie = createDriftingGrating(lmbda,ori,tf,phase,winOptions = winOptions)
    winMask = np.expand_dims(np.asarray(matEngine.mkCosTaper(np.float(winOptions['winSize']),taperSize)),axis = 2)
    
    return thisMovie*winMask
def getPlexLink(neuronName,grating):
    return matEngine.linkGratingFromMovie(neuronName,grating)

