# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 13:15:27 2017

@author: phil
"""
'''
initial change from loaded stimulus to cropped stimulus
'''
import numpy as np

from utils import p_utils

class stimCropper(object):
    cropWin = None
    cropKern = None
    frameAxis = None
    winLen = None
    cropWinIndexStyle =None
    cropList = [None,None,None]

    def __init__(self,x_axis,y_axis,t_axis,reshapeOrder = 'F',cropWinIndexStyle = 'MATLAB',interp = 'bilinear'):
        #store information about the stimulus
        
        self.cropWinIndexStyle = cropWinIndexStyle               
        self.reshapeOrder = reshapeOrder
        self.interp = interp
        #In loaded stimulus, keep track of which axes correspond to which dimension
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.t_axis = t_axis
        return


    def setCropWin(self,cropWin):
        
        if self.cropWinIndexStyle == 'MATLAB':
            #In matlab crop win style [1 480 1 480] represents the full window
            cropWin[0] = cropWin[0] -1 
            cropWin[2] = cropWin[2] -1 
        self.cropWin = cropWin
        #Cropping windows were stored as X window in cropWin[2] to cropWin[3]
        self.winLen = len(range(cropWin[2],cropWin[3]))
        #Cropping windows were stored as Y window in cropWin[0] to cropWin[1]
        #ensure that cropping windows are square

        assert self.winLen == len(range(cropWin[0],cropWin[1]))
        
        self.cropList[self.x_axis] = slice(self.cropWin[2],self.cropWin[3])
        self.cropList[self.y_axis] = slice(self.cropWin[0],self.cropWin[1])
        
        return
        
    def setCropKern(self,cropKern):
        self.cropKern = cropKern
        return

    def selectCropKern(self,downSampSettings):
        return selectCropKern(downSampSettings,self.winLen)
    def selectAndSetCropKern(self,downSampSettings):
        self.cropKern = self.selectCropKern(downSampSettings)
        return self.cropKern
    def crop(self,stim):
        #cropping is done in the (x_axis,y_axis,t_axis)
        numFrames = stim.shape[self.t_axis]       
        self.cropList[self.t_axis] = slice(0,numFrames) #don't crop across the time axis
        
        croppedStim = stim[self.cropList[0],self.cropList[1],self.cropList[2]]    
        
        return croppedStim
    def resize(self,stim):
        #resize the stimulus,
        return p_utils.movieResize(stim,self.cropKern,self.interp,self.t_axis)
    def shapeForModel(self,stim):
        #changes the stimulus from shape loaded from matlab 
        # to shape required by the keras models (t,x*y)
        
        numFrames = stim.shape[self.t_axis] 
        croppedStim = self.crop(stim)
        
        resizedStim = self.resize(croppedStim)
        #Model Standard
        shapedStim = resizedStim.transpose(self.t_axis,self.x_axis,self.y_axis)
        shapedStim = np.reshape(shapedStim,(numFrames,np.square(self.cropKern)),order = self.reshapeOrder)
        return shapedStim
        
    def cropString(self):
        return str(self.winLen) + ' to ' + str(self.cropKern)
        
        
def selectCropKern(downsampSetting,currMovieSize):
    ''' Selects the downsampling size depending on the size of the cropped.
        Possible options:  'Full', 'Downsample', 'Downsample_limited', 'Value'
        
    '''
    if downsampSetting[0] == 'Full':
        #keep full cropped movie
        #downsampSetting should be ('Full')
        cropKern = currMovieSize
    elif downsampSetting[0] == 'Downsample':
        #downsample the cropped movie by a certain proportion
        #downsampSetting should be ('Downsample',downsample_factor)
        cropKern = int(currMovieSize/downsampSetting[1])
    elif downsampSetting[0] == 'Downsample_limited':
        #downsample the cropped movie, but keep a limit on downsampled movie size
        #downsampSetting should be ('Downsample_limited',downsample_factor,min_limit,max_limit)
        cropKern = int(currMovieSize/downsampSetting[1])
        if cropKern < downsampSetting[2]:
            cropKern = downsampSetting[2]
        if cropKern > downsampSetting[3]:
            cropKern = downsampSetting[3]
    elif downsampSetting[0] == 'Value': 
        #downsample the movie to be a certain size
        #downsampSetting should be ('Value',final_movie_size)
        cropKern = downsampSetting[1]
    elif downsampSetting[0] ==  'Downsample_Minimum':
        cropKern = currMovieSize
        if cropKern > downsampSetting[1]:
            cropKern = downsampSetting[1]
    else:
        raise('unknown downsampSetting')
    
    print('cropKern: '+ str(cropKern))
    return cropKern