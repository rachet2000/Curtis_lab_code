# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 16:21:42 2017

@author: phil


Load Model and test against grating stimuli.
Slightly different than the general k_loadModel, however, we some tweaks, this module
is able to handle multiple gratings assigned to the same neuron.
"""


import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
import pickle
import scipy.io as io
from pySIA import p_utils
import numpy as np

from p_utils import myInput
from utils import k_drawMat
from utils import matEngine
from curtis_lab_utils import clab_utils

import pyResultsToMat

import p_matchModels
import p_stimCropper
import k_shapeStimulus


class drawGratingIterator(object):
    def __init__(self,gratingType =None):
        self.gratingType = gratingType
        self.counter = 0
            
        
    def set(self,gratingName):
        self.counter = 0
        
            
        [self.stimList,self.info] = matEngine.generateQuickStimuli(gratingName,self.gratingType)
        self.paramList = self.info['levVals']
        self.settingsNamesList = [self.info['vary']]
        self.settingsValuesList = [self.info['levVals']]
        self.size =len(self.paramList)
        self.gratingName = gratingName
                
    def __iter__(self):
        return self
    def __next__(self):
        if self.counter >= self.size:
            raise StopIteration
        else:
            stim = self.stimList[self.counter]
            stim = np.uint8(stim.transpose(1,2,0))
            self.counter +=1
            return stim
    def next(self):
        return self.__next__()        
    def options(self):
        optionsDict = dict()
        optionsDict['paramList'] = self.paramList
        optionsDict['settingsNamesList'] = self.settingsNamesList
        optionsDict['settingsValuesList'] = self.settingsValuesList
        return optionsDict
        
    def hasNext(self):
        if self.counter < self.size:
            return True
        else:
            return False
    def reset(self):
        self.counter = 0
    
        
def stimTransform(stim,cropper,model_class,model_options,nFrames):
    
    stim = cropper.shapeForModel(stim)
   
    if model_class.stimScaler is None:
        #If we are not given a scaling method,
        #use default standardization.
        defaultScaler = k_shapeStimulus.scaleStimulusClass(model_options)
        defaultScaler.calcScaleParams(stim)
        stim = defaultScaler.applyScaleTransform(stim)
    else:     
        stim = model_class.stimScaler.applyScaleTransform(stim)
    
    while(np.size(stim,0) <nFrames):
        stim = np.concatenate((stim,stim),axis =0)
    stim = model_class.shapeStimulus()(stim,model_options,np.size(stim,axis=0))
            
    return stim

def loadNeuronModel(neuronData,extra_defaults=None):
    neuronName = neuronData['neuronName']        
    print(neuronName)        
    #get model data#
    if 'model_class' in neuronData:
        model_class = neuronData['model_class']
    else: 
        model_class = p_matchModels.matchModel(neuronData['method'])
        
    model_options = model_class.defaultOptions(neuronData['options'])
    
    if 'Input_Shape' in neuronData['options']:
        model_options['Input_Shape'] = neuronData['options']['Input_Shape']
    else:
        model_options['Input_Shape'] = extra_defaults['Input_Shape']
        
    if 'Frames' in neuronData['options']:
        model_options['Frames'] = neuronData['options']['Frames']
    else:
        model_options['Frames'] = extra_defaults['Frames']
    if 'Reshape_Order' in neuronData['options']:
        model_options['Reshape_Order'] = neuronData['options']['Reshape_Order']
    else:
        model_options['Reshape_Order'] = extra_defaults['Reshape_Order']
          

    model = model_class.buildModel(model_options)
    #set the weights to the model
    model.set_weights(neuronData['model']['weights'])
    
    return model,model_class,model_options

def predictResultDictGratingResponse(neuronList,stimIterator,cropper,nFrames,transpose = False,extra_defaults= None,strfFile = None):
    '''
        Input: 
            - neuronList (for example from pyResultsToMat.getBestDict() function)
        Output:
            - ******
        
        Will predict the grating responses from keras models
    '''
    
    pathStruct = clab_utils.getPaths()
    responseDict = dict()
    
    neuronIdx = 1
    
    for key,neuronData in neuronList.items():        
        neuronName = neuronData['neuronName']
        model,model_class,model_options = loadNeuronModel(neuronData,extra_defaults)
        
        #get input(grating movie)#
        if 'Crop' in strfFile: 
            cropFileFullName =  os.path.sep.join((pathStruct['DATAlocation'],neuronName,neuronName +'_'+ strfFile +'.mat' ))                      
            cropWin = clab_utils.getCropWin(cropFileFullName)
        else:
#            cropWin = [0, 480, 0, 480] 
            cropWin = [1, 480, 1, 480]
        
        cropper.setCropWin(cropWin)
        cropper.setCropKern(model_options['Input_Shape'][-1])
        matchName = neuronName[:9]
        gratingNameList = k_drawMat.getPlexLink(matchName,stimIterator.gratingType)
        
        if len(gratingNameList) != 0 and gratingNameList[0] != '---------':
            for gratingName in gratingNameList:
                print('Grating Name : ' + str(gratingName))
                #if we are building the stimulus from grating param file, then 
                #we need to pass the neuronName
                stimIterator.set(gratingName)
                
                
                #generate response for each stim and go through
                fullNLResponse = []
                fullNoNLResponse = []
                
                while(stimIterator.hasNext()):
                    print(str(stimIterator.counter) +"/" +str(stimIterator.size) )
                    
                    stim = stimIterator.next()
                    stim = stimTransform(stim,cropper,model_class,model_options,nFrames)
                    noNLResponse = model.predict(stim)
                    NLResponse = p_utils.siaNLPredict(noNLResponse,neuronData['p_opt'])
        
                    fullNLResponse.append(NLResponse)
                    fullNoNLResponse.append(noNLResponse)
                    
                neuronKey = 'neuron' + str(neuronIdx)
                responseDict[neuronKey] = dict()
                responseDict[neuronKey]['NLResponse'] = fullNLResponse
                responseDict[neuronKey]['noNLResponse'] = fullNoNLResponse
                responseDict[neuronKey]['stimOptions'] = stimIterator.options()
                responseDict[neuronKey]['nFrames'] = nFrames
                responseDict[neuronKey]['neuronName'] = neuronName
                responseDict[neuronKey]['gratingName'] = gratingName
                
                stimIterator.reset()
                neuronIdx += 1
    
    return responseDict

def predictResultDictGUI():
    pathStruct = clab_utils.getPaths()
    os.chdir(pathStruct['channelLocation'])
   
    saveName = myInput('Input pkl file stored in the channelLocation directory:\n')
    os.chdir(pathStruct['channelLocation'])

    resultDict = pickle.load(open(saveName+'.pkl','rb'))
    resultDict = pyResultsToMat.fillMissingData(resultDict)
    bestList = pyResultsToMat.getBestDict(resultDict)    
    strfFile = myInput('Crop file?:\n')

    nFrames = int(myInput('length of stimulus (in frames i.e. 150 for 1Hz)?:\n'))
    
   
    iteratorFile = myInput('stimulus grating name??:\n')

    stimIterator = drawGratingIterator(iteratorFile)
    cropper = p_stimCropper.stimCropper(0,1,2,
                                  reshapeOrder = 'F',interp = 'bilinear',
                                  cropWinIndexStyle='MATLAB')
    iteratorString = '_matGen' + iteratorFile
        

        
    transpose_switch = myInput('transpose image? Will need to do this for old models <2017 (y/n):\n')
    if transpose_switch == 'y':
        transpose = True
    else:
        transpose = False
    
    extra_def_switch = myInput('Use default settings for Frames and Input Shape? (y/n):\n')
    if extra_def_switch == 'y':
        extra_defaults = dict()
        extra_defaults['Frames'] = list(range(8))
        extra_defaults['Input_Shape'] = (8,30,30)
        extra_defaults['Reshape_Order'] = 'F'
    else:
        extra_defaults = None


    responseDict = predictResultDictGratingResponse(bestList,stimIterator,cropper,nFrames,transpose,extra_defaults = extra_defaults,strfFile = strfFile)
    os.chdir(pathStruct['channelLocation'])
    io.savemat(saveName+ iteratorString + '_predictions' +'.mat',responseDict)
        
        
    return



    
if __name__ == '__main__':   
    predictResultDictGUI()