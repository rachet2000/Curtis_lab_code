#-*- coding: utf-8 -*-
"""
Created on Tue Mar  8 12:20:40 2016

@author: phil
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
import pickle
#from models import strflab_allModelInfo  #import matlab.engine before matplotlib.pyplot and scipy
import scipy.io as io
import numpy as np

import collections
from utils import p_utils
from curtis_lab_utils import clab_utils
from utils.p_utils import myInput
from utils import k_utils
from models import k_allModelInfo
from models import a_allModelInfo


### Set data folder, where pkl/mat is stored###  
paths = clab_utils.getPaths()
dataFolder = paths['channelLocation']
#dataFolder = '/media/phil/52301BCA301BB3C7/Data/Results/resultStructFiles/'
###############################################

from curtis_lab_utils.clab_utils import setReorgModules

def readPkl():  
    ''' Input: 
            - None: will read from stdin, entire the name of a pkl file in dataFolder
        Output:
            - resultDict: dictionary stored in the pkl file
                keys: names of neurons
                values: list of all models and results for that neuron (for different hyperparameter settings)
    '''
            
    os.chdir(dataFolder)
    
    resultFile = myInput('result file :')    
    resultDict = pickle.load(open(resultFile+'.pkl','rb'))
    
    print('If this is an older pkl, please run fillMissingData(resultDict) before running other functions')
    
    return resultDict,resultFile
def fillMissingData(resultDict):
    '''- missingData: settings used to generate the data.
    older pkls will be missing these fields:
    ['method'] - the algorithm/model 
    ['options']['Frames'] - the set of frames of stimulus corresponding to a certain delay from the response,
                            This is usually list(range(8))
    Example:
        algorithm = 'kConvNet'
        frames = list(range(8))
    '''
    print('Asking for missing parameters...')
    exampleNeuron = resultDict.keys()[0]
    if 'method' not in resultDict[exampleNeuron][0]:
        method = myInput('algorithm? (e.g. kConvNet):\n')
    else:
        method = None
    if 'Frames' not in resultDict[exampleNeuron][0]['options']:
        frames = eval(myInput('frames/delays used (e.g. list(range(8))):\n'))
    else:
        frames = None

    if 'Stride' not in resultDict[exampleNeuron][0]['options']:
        stride = int(myInput('stride? (e.g. 1):\n'))
    else:
        stride = None

    for key,neuronList in resultDict.items():
        for neuronTrial in neuronList:
            if 'method' not in neuronTrial:
                if method is None:
                    raise(Exception('no method given, or stored'))
                else:
                    neuronTrial['method'] = method
            if 'Frames' not in neuronTrial['options']:
                if frames is None:
                    raise(Exception('no Frames data given, or stored'))
                else:
                    neuronTrial['options']['Frames'] = frames
            
            if 'Stride' not in neuronTrial['options']:
                if stride is None:
                    raise(Exception('no stride data given, or stored'))
                else:
                    neuronTrial['options']['Stride'] = stride
    
    return resultDict    
def saveAsMat(bestDict):
    ''' Input: 
            - bestDict (obtained from converting resultDict with getbestDict)
        Output:
            - None (will save bestDict as .mat file)
    '''
    bestDict = removeModelClassBestDict(bestDict)
    for neuronIdx in bestDict:
        neuronData = bestDict[neuronIdx]
    #remove the config data since it contains None-types which dont convert to .mat
        try:
            del neuronData['model']['config']
        except:
            print('neuron config data missing')
        neuronData = prepareDictForMat(neuronData['options'])
    
    os.chdir(dataFolder)    
    newMatFileName  = myInput('new mat file name :')
    io.savemat(newMatFileName+'.mat',bestDict)
    return

def checkHyperparam(neuronTrials):
    allOptionDicts = [x['options'] for x in neuronTrials]
    
    hyperParamDict = collections.defaultdict(set)
    for d in allOptionDicts:
        for key,value in d.items():
            if isinstance(value,collections.Hashable):
                hyperParamDict[key].add(value)
                
    combinations = 1
    for key,value in hyperParamDict.items():
        if len(value) > 1:
            print(key)
            print(value)
        combinations = combinations* len(value)
    print('number of combinations that are hashable : ' + str(combinations))
    return hyperParamDict

def viewHyperparamVAF(resultDict,identifyVal):
    topDict = dict()
    distDict = dict()
    diffDict = dict()
    for key,neuron in resultDict.items():
        bestVaf = 0
        
        vafList = []
        for neuronTrial in neuron:
            vafList.append(np.nan_to_num(neuronTrial['predVaf']))
        vafList = sorted(vafList)
        bestVaf = vafList[-1]
        secondBestVaf = vafList[-2]
            
        for neuronTrial in neuron:
            val = None
            exec('val = neuronTrial'+identifyVal)
            thisVaf = np.nan_to_num(neuronTrial['predVaf'])
            if val not in distDict:                
                distDict[val] = [thisVaf]
                diffDict[val] = [bestVaf - thisVaf]
            else:                
                distDict[val].append(thisVaf)
                diffDict[val].append(bestVaf - thisVaf)
            if neuronTrial['predVaf'] == secondBestVaf:        
                if val not in topDict:
                    topDict[val] = [bestVaf - secondBestVaf]
                else:
                    topDict[val].append(bestVaf - secondBestVaf)
            
    return topDict,distDict,diffDict

def compareWithOffHyperparam(resultDict,identifyVal,hyperParamOnVal,hyperParamOffVal):
    diffList = []
    for key,neuron in resultDict.items():
        for neuronTrial in neuron:
            exec('val = neuronTrial'+identifyVal)
            if val == hyperParamOnVal:
                onVal = neuronTrial['predVaf']
            if val == hyperParamOffVal:
                offVal = neuronTrial['predVaf']
        diffList.append(onVal - offVal)
    return diffList
def prepareDictForMat(myDict):
    for key in myDict:
        val = myDict[key]
        if val is None:
            #Remove None-types. Convert them to Strings
            myDict[key] = 'None'
        else:
            try:
                np.asarray(val)
            except ValueError:
                #Current numpy code has a bug where it will try to broadcast list
                #objects unless one of the items is empty (or object)
                if type(val) is list:
#                    myDict[key][0] = val[0].astype(np.object)
                    myDict[key].append('Null_obj')
                        
        
    return myDict
    
def getNeuronMatrix(resultDict):
    ''' To use if you want to quickly view the VAF values stored in resultDict
        Input: 
            - resultDict (from readPkl)
        Output:
            - neuronMatrix: a matrix describing the results obtained in resultDict.
            - nameList: a list indexing the neuron's name for corresponding index in neuronMatrix 
    '''
    nameList = []
    neuronMatrix = []
    neuronMatrix = np.array([])
    for neuronName in resultDict:        
        nameList.append(neuronName)
        neuronData = vafReader(resultDict[neuronName])         
        neuronMatrix = np.nan_to_num(np.dstack((neuronMatrix, neuronData)) if neuronMatrix.size else neuronData)
    return (neuronMatrix,nameList)
    
def getBestDict(resultDict,optionDict = dict()):
    ''' Input: 
            - resultDict: obtained from readPkl()
            - optionDict: dictionary, where each key must match one key in the options dictionary of each neuron Trial.
                                    values are lambda functions which correspond to True on the values of interest.
                                    i.e. if you only want to check trials  where Pool_Size = 3,
                                    optionDict['Pool_Size'] = lambda x: x ==3

        Output:
            - matDict: dictionary stored in a way that can be saved as .mat file using saveAsMat
                will only store the result that obtains the best validVaf
            
    '''    
    
    bestDict = dict()    
    neuronIdx = 1
    
    for neuronName in resultDict:
        neuronList = resultDict[neuronName]
        bestVaf = -1
        for neuronTrial in neuronList:
            
            #skip this trial if the options don't match the optionDict
            skipFlag = 0
            for key,lambdaFunc in optionDict.items():
                if not lambdaFunc(neuronTrial['options'][key]):
                    skipFlag = 1
            if skipFlag == 1:
                continue
            
            #save this trial if it has a better VAF
            validVaf = p_utils.turnNantoZero(neuronTrial['validVaf'])
            noNLValidVaf = p_utils.turnNantoZero(neuronTrial['noNLValidVaf'])
            if validVaf>bestVaf:
                neuronData = neuronTrial
                bestVaf = neuronTrial['validVaf']
            if noNLValidVaf>bestVaf:
                neuronData = neuronTrial
                bestVaf = neuronTrial['noNLValidVaf']
        neuronFieldName = 'neuron' + str(neuronIdx)
        
        
        #keep relevant estimated parameters
        neuronData = configNeuronData(neuronData)

        
        #store in matDict
        bestDict[neuronFieldName] = neuronData
        bestDict[neuronFieldName]['neuronName'] = neuronName
        neuronIdx = neuronIdx + 1
    
    return bestDict
    


def configNeuronData(neuronData):
    ''' Will configure the neuronData structure such that it can be saved as a .mat file '''
    
    if 'method' not in neuronData:
        raise Exception('no method data, use the fillMissingData function')
    else:
        algorithm = neuronData['method']
    
    

        
    if algorithm == 'kConvNet' or algorithm == 'kRadialNet':
        neuronData['alpha'] = neuronData['model']['weights'][-3]
        neuronData['p_opt'] = neuronData['model']['p_opt']
        
        
    elif algorithm == 'kOnOffConvNet':
        
        '''TODO: CHECK WHICH PRELU IS WHICH TO BE SURE''' 
        ''' must reorganize the weights because the matlab conversion will crash for some unknown reason otherwise'''
        neuronData['neg_PReLU'] = neuronData['model']['weights'][0]
        neuronData['pos_PReLU'] = neuronData['model']['weights'][1]
        neuronData['model']['weights'] = neuronData['model']['weights'][2:]
        
        neuronData['alpha'] = neuronData['model']['weights'][-3]
        neuronData['p_opt'] = neuronData['model']['p_opt']

    elif algorithm == 'kConvSplitNet':
        neuronData['p_opt'] = neuronData['model']['p_opt']
    
    elif algorithm == 'kConvGaussNet':
        neuronData['alpha'] = neuronData['model']['weights'][2]
        neuronData['p_opt'] = neuronData['model']['p_opt']

            
    
    return neuronData

def getAllTrialDict(neuronTrials):
    allDict = dict()
    trialNum = 1
    for neuronData in neuronTrials:
        allDict['neuronTrial_'+str(trialNum)] = neuronData
        trialNum = trialNum + 1
    return allDict
    
def removeModelClassResultDict(resultDict):
    for key,val in resultDict.items():
        for neuronData in val:
            del neuronData['model_class']
    return resultDict
def removeModelClassBestDict(bestDict):
    for key,neuronData in bestDict.items():
        try:
            del neuronData['model_class']
        except:
            print('neuron model_class missing at' + str(key))
    return bestDict
def checkModelClassBestDict(bestDict):
    myDict = dict()
    for key,neuronData in bestDict.items():
        if neuronData['model_class'] in myDict:
            print(key)
            print(neuronData['neuronName'])
            print(neuronData['model_class'])
            print(myDict[neuronData['model_class']]['neuronName'])
        else:
            myDict[neuronData['model_class']] = neuronData

    return bestDict
def addP_OPTResultDict(resultDict):
    for key,val in resultDict.items():
        for neuronData in val:
            neuronData['p_opt'] = neuronData['model']['p_opt']
    return resultDict
def addAlphaResultDict(resultDict,alphaIdx):
    for key,val in resultDict.items():
        for neuronData in val:
            neuronData['alpha'] = neuronData['model']['weights'][alphaIdx]
    return resultDict
def findLastNeuron(bestDict):
    #load the allNeuronStruct
    pathStruct= clab_utils.getPaths()
    AllNeuronStructLocation = os.path.sep.join((pathStruct['AllNeuronLocation'],'AllNeuronStruct.mat'))
    allNeuronStruct = io.loadmat(AllNeuronStructLocation)
    neuronStruct = np.squeeze(allNeuronStruct['neuronStruct']) 
    neuronNameList = [str(neuron[0][0]) for neuron in neuronStruct]
    bestDictSet = { neuronData['neuronName'] for neuronData in bestDict.values()}
    lastNeuronName = None
    lastNeuronIdx = None
    for idx,neuronName in enumerate(neuronNameList):
        neuronName = neuronName[:-4]
        if neuronName in bestDictSet:
            lastNeuronName = neuronName
            lastNeuronIdx = idx
    return lastNeuronIdx,lastNeuronName

#if __name__ == '__main__':
#    aa = readPkl()
#    resultDict = aa[0]
#    bestDict = getBestDict(resultDict)

#    
