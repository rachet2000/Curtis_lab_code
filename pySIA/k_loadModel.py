# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 14:50:44 2016

@author: phil
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
import pickle
import scipy.io as io
from pySIA import p_utils
import matplotlib.pyplot as plt
import numpy as np
import itertools

import p_utils
from p_utils import myInput
from utils import k_drawMat
from utils import matEngine
from curtis_lab_utils import clab_utils

import pyResultsToMat

from skimage.transform import resize
import p_matchModels
import p_stimCropper
import k_shapeStimulus
#def loadModelFromOptions(options,model_name):
#    '''rebuild the model from options data
#        I didn't originally save a JSON/YAML file, 
#        so this code will take in the options and create a JSON/YAML string.
#        This string will be passed to other functions and thus we can rebuild
#        the models there.
#        
#        
#    '''    
#    
#    return model
#def modelPredict(model,stimulus):
#    pass
class aMovieClipIterator(object):
    def __init__(self,movieSet):
        pathStruct = clab_utils.getPaths()
        os.chdir(pathStruct['MOVIESLocation'])
        os.chdir(movieSet)
        self.movieLocation = os.getcwd()
        self.counter = 1
        self.size = 30
        self.movieName = movieSet
    def __iter__(self):
        return self
    def set(self,neuronName):
        #the aMovieClipIterator does not need to be set
        return
    def __next__(self):
        if self.counter > self.size:
            raise StopIteration
        else:
            os.chdir(self.movieLocation)
            
            if self.counter <= 20:
                movieNum = self.counter
                fullMovieName = '_'.join((self.movieName,'0000','{:02d}'.format(movieNum)))
                
            elif self.counter <=25:
                movieNum = self.counter - 20
                fullMovieName = '_'.join((self.movieName,'reg','0000','{:02d}'.format(movieNum)))
            else:
                movieNum = self.counter - 25
                fullMovieName = '_'.join((self.movieName,'pred','0000','{:02d}'.format(movieNum)))
            
                
            fullMovieName = fullMovieName + '.mat'
            
            stim = io.loadmat(fullMovieName)
            stim = stim['mvMovie']
            
            self.counter +=1
            return stim
    def next(self):
        return self.__next__()
    def hasNext(self):
        if self.counter < self.size:
            return True
        else:
            return False
    def options(self):
        optionsDict = dict()
        optionsDict['movieName'] = self.movieName

        return optionsDict
    def reset(self):
        self.counter = 1

class gratingMovieIterator(object):
    def __init__(self,movieSet):
        pathStruct = clab_utils.getPaths()
        os.chdir(pathStruct['Grating_MoviesLocation'])
        os.chdir(movieSet)
        self.movieLocation = os.getcwd()
        
        movieSettings = io.loadmat(movieSet +'_settings.mat')
        self.settingsDict = dict()
        self.settingsValuesList = []
        self.settingsNamesList = []

        for param in movieSettings['dataSettings'][0]:
            paramName = str(param[0][0])
            self.settingsDict[paramName] = dict()
            paramValues =param[1][0]
            self.settingsValuesList.append(paramValues)
            self.settingsNamesList.append(paramName)
            
            for idx,value in enumerate(paramValues):
                self.settingsDict[paramName][value] = idx
        self.paramList = list(itertools.product(*self.settingsValuesList))
        self.counter = 0
        self.size = len(self.paramList)
        self.movieName = movieSet
    def __iter__(self):
        return self
    def set(self,neuronName):
        #the gratingMovieIterator does not need to be set
        return
    def __next__(self):
        if self.counter >= self.size:
            raise StopIteration
        else:
            os.chdir(self.movieLocation)
            combSetting = self.paramList[self.counter]
            thisParamSetting = zip(self.settingsNamesList,combSetting)
            fullMovieName = self.movieName
            for paramName,paramVal in thisParamSetting:
                fullMovieName = '_'.join((fullMovieName,paramName + str(paramVal)))
                
            fullMovieName = fullMovieName + '.mat'
            
            stim = io.loadmat(fullMovieName)
            stim = stim['currData'][0][0][0].astype(np.uint32)
            
            self.counter +=1
            return stim
    def next(self):
        return self.__next__()
    def hasNext(self):
        if self.counter < self.size:
            return True
        else:
            return False
    def options(self):
        optionsDict = dict()
        optionsDict['paramList'] = self.paramList
        optionsDict['settingsValuesList'] = self.settingsValuesList
        optionsDict['settingsNamesList'] = self.settingsNamesList
        return optionsDict
    def reset(self):
        self.counter = 0

class gratingParamIterator(object):
    def __init__(self,paramFile,gratingType =None):
        if gratingType == None:
            gratingType = paramFile.split('_')[0]
        pathStruct = clab_utils.getPaths()
        os.chdir(pathStruct['AnalysisLocation'])
        loadStruct = io.loadmat(paramFile)
        resultStruct = loadStruct[gratingType +'ParamStruct']
        
        loadedTypes = resultStruct.dtype.names
        resultStruct = np.squeeze(resultStruct)
        
        nameIdx = [i for i,val in enumerate(loadedTypes) if val =='name'][0]
        lambdaIdx = [i for i,val in enumerate(loadedTypes) if val =='lambda'][0]
        oriIdx = [i for i,val in enumerate(loadedTypes) if val =='ori'][0]
        tfIdx = [i for i,val in enumerate(loadedTypes) if val =='tf'][0]
        macroNameIdx = [i for i,val in enumerate(loadedTypes) if val =='macroName'][0]
        levValsIdx = [i for i,val in enumerate(loadedTypes) if val =='levVals'][0]
        
        self.neuronParams= dict()
        
        #very hack-y way of dealing with the matlab-python conversion,
        #may be prone to bugs
        for entry in resultStruct:
            neuronName = str(entry[nameIdx][0])
            self.neuronParams[neuronName] = dict()
            self.neuronParams[neuronName]['tf'] = [int(entry[tfIdx][0][0])]
            self.neuronParams[neuronName]['lambda'] = [int(entry[lambdaIdx][0][0])]
            self.neuronParams[neuronName]['ori'] = [int(entry[oriIdx][0][0])]
            self.neuronParams[neuronName]['macroName'] = str(entry[macroNameIdx][0])
            self.neuronParams[neuronName]['levVals'] = [int(np.squeeze(i)) for i in entry[levValsIdx][0]]
        self.gratingType = gratingType
        self.settingsNamesList = ['lambda','ori','tf']
        self.counter = 0
        
        
    def set(self,neuronName):
        self.counter = 0
        matchName = neuronName[:9] #the paramFile holds the macro names which are len 9
        
        gratingNameList = k_drawMat.getPlexLink(matchName,self.gratingType) 
        
        if len(gratingNameList) == 0:
            self.paramList = []
            self.size = 0
        else:

            self.paramList = []
            for gratingName in gratingNameList:
                if gratingName in self.neuronParams:
                    settingsDict = self.neuronParams[gratingName]
                    if settingsDict['macroName']== 'tft':
                        settingsDict['tf'] = settingsDict['levVals'] 
                    elif settingsDict['macroName']== 'sft':
                        settingsDict['lambda'] = settingsDict['levVals'] 
                    elif settingsDict['macroName']== 'ort':
                        settingsDict['ori'] = settingsDict['levVals']
                    else:
                        break #not a ort/sft/tft grating (which can be created using cosTaperedGrating)
    
                    self.paramList.extend(list(itertools.product(settingsDict['lambda'] ,settingsDict['ori'],settingsDict['tf'])))
                
            self.size =len(self.paramList)
        
    def __iter__(self):
        return self
    def __next__(self):
        if self.counter >= self.size:
            raise StopIteration
        else:
            combSetting = self.paramList[self.counter]
            stim = k_drawMat.cosTaperedGrating(*combSetting)
            
            self.counter +=1
            return stim
    def next(self):
        return self.__next__()        
    def options(self):
        optionsDict = dict()
        optionsDict['paramList'] = self.paramList
        optionsDict['settingsNamesList'] = self.settingsNamesList
        return optionsDict
        
    def hasNext(self):
        if self.counter < self.size:
            return True
        else:
            return False
    def reset(self):
        self.counter = 0

class drawGratingIterator(object):
    def __init__(self,gratingType =None):
        self.gratingType = gratingType
        self.counter = 0
            
        
    def set(self,neuronName):
        self.counter = 0
        matchName = neuronName[:9]
        
        gratingNameList = k_drawMat.getPlexLink(matchName,self.gratingType) 
        
        if len(gratingNameList) == 0 or gratingNameList[0] == '---------':
            self.paramList = []
            self.size = 0
        else:
            if len(gratingNameList) >= 2:
                raise('multiple matching gratings for this neuron, cannot handle')
            
            for gratingName in gratingNameList:
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
        
        #if we are building the stimulus from grating param file, then 
        #we need to pass the neuronName
        stimIterator.set(neuronName)
        
        
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
    
    predictionMode = myInput('Grating prediction mode? (1 (gratingMovie set)  or 2 (grating movie matching neuron) 3 (aMovieClip Set)):\n')
    
    if predictionMode == '1':
        os.chdir(pathStruct['Grating_MoviesLocation'])
        print('Current Grating Movies Directory:')
        print(os.getcwd())

        
        print('Possible Movies: ')
        for directory in os.listdir('.'):
            if os.path.isdir(directory):
                print(directory)
        iteratorFile = myInput('Movie set?:\n') 
        iteratorString = '_movieSet' + iteratorFile
        stimIterator = gratingMovieIterator(iteratorFile)
        cropper = p_stimCropper.stimCropper(0,1,2,
                                      reshapeOrder = 'F',interp = 'bilinear',
                                      cropWinIndexStyle='MATLAB')

    elif predictionMode =='2':
        os.chdir(pathStruct['AnalysisLocation'])
        print('Current Analysis Files Directory:')
        print(os.getcwd())
        
        print('Possible Param Files: ')
        for file in os.listdir('.'):
            if 'GratingParams' in file:
                print(file)
        iteratorFile = myInput('Grating Analysis File??:\n')
        iteratorString = '_paramFile' + iteratorFile
        stimIterator = gratingParamIterator(iteratorFile)
        cropper = p_stimCropper.stimCropper(0,1,2,
                                      reshapeOrder = 'F',interp = 'bilinear',
                                      cropWinIndexStyle='MATLAB')

    elif predictionMode =='3':
        os.chdir(pathStruct['MOVIESLocation'])
        print('Current Natural Image Movies Directory:')
        print(os.getcwd())
        
        print('Possible Movies: ')
        for directory in os.listdir('.'):
            if os.path.isdir(directory):
                print(directory)
        iteratorFile = myInput('Movie set?:\n') 
        iteratorString = '_aMovieClipSet' + iteratorFile
        stimIterator = aMovieClipIterator(iteratorFile)
        cropper = p_stimCropper.stimCropper(0,1,2,
                                      reshapeOrder = 'F',interp = 'bilinear',
                                      cropWinIndexStyle='MATLAB')

    elif predictionMode =='4':
        iteratorFile = myInput('stimulus grating name??:\n')

        stimIterator = drawGratingIterator(iteratorFile)
        cropper = p_stimCropper.stimCropper(0,1,2,
                                      reshapeOrder = 'F',interp = 'bilinear',
                                      cropWinIndexStyle='MATLAB')
        iteratorString = '_mode4' + iteratorFile
        
    else:
        raise('not a valid mode')
        
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

#def OLD_predictMovieSetGratingResponse(model,modelData,cropWin,movieSet):   
#   
#    
#    #get movieSet settings
#    pathStruct = p_utils.getPaths()
#    os.chdir(pathStruct['Grating_MoviesLocation'])
#    os.chdir(movieSet)
#    
#    movieSettings = io.loadmat(movieSet +'_settings.mat')
#    dataMatrix = movieSettings['dataMatrix'].astype(np.float32)
#    
#    
#    settingsDict = dict()
#    settingsValuesList = []
#    settingsNamesList = []
#    predictedResponseList = []
#    for param in movieSettings['dataSettings'][0]:
#        paramName = str(param[0][0])
#        settingsDict[paramName] = dict()
#        paramValues =param[1][0]
#        settingsValuesList.append(paramValues)
#        settingsNamesList.append(paramName)
#        
#        for idx,value in enumerate(paramValues):
#            settingsDict[paramName][value] = idx
#    
#    paramCombinations = itertools.product(*settingsValuesList)
#    
#    for combSetting in paramCombinations:
#        
#        thisParamSetting = zip(settingsNamesList,combSetting)
#        movieName = movieSet
#        dataMatrixIndex = []
#        for paramName,paramVal in thisParamSetting:
#            movieName = '_'.join((movieName,paramName + str(paramVal)))
#            dataMatrixIndex.append(settingsDict[paramName][paramVal])
#            
#        movieName = movieName + '.mat'
#        dataMatrixIndex = tuple(dataMatrixIndex)
#        print(dataMatrixIndex)
#        
#        #get current grating movie
#        cropKern = int(modelData['options']['Downsampling_Ratio'].split()[-1])
#        
#        stim = io.loadmat(movieName)
#        stim = stim['currData'][0][0][0].astype(np.uint32)
#        
#        #!!!!!!!replace stim here!!!!!!!!!!!!11111!!!
#        #!!!!!!add 8 frames of stimulus
#        stim = stim[cropWin[2]:cropWin[3]+1,cropWin[0]:cropWin[1]+1,:]
#        stim = resize(stim,(cropKern,cropKern))
#        stim = stim.transpose((1,0,2))  
#        stim = np.transpose(np.reshape(stim,(np.square(cropKern),np.size(stim,2))))
#        
#
#        stim = runAlgorithms.setStimulus(stim,np.size(stim,0),modelData['options'],method = modelData['method'])
#        
#        
#        predictedResponse = model.predict(stim)
#        #!!!! NL 
#        #!!!!!replicate response until we get 2 seconds worth resonse
#        avgResponse = np.mean(predictedResponse)
#        dataMatrix[dataMatrixIndex] = avgResponse
#        predictedResponseList.append(predictedResponse)
#        
#            
#    
#    
#    options = dict()
#    options['settingsNames'] = settingsNamesList
#    options['settingsValues'] = settingsValuesList
#    options['paramCombinations'] = list(paramCombinations)
#    
#    return dataMatrix,options