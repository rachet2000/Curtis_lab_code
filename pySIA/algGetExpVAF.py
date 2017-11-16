# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 15:35:05 2017

@author: phil
"""
import k_loadModel
from utils import p_utils
from utils.p_utils import myInput
from curtis_lab_utils import clab_utils
import pyResultsToMat
import pickle
import os
from scipy import io
import numpy as np
import sys
def algOverPkl():
    pathStruct = clab_utils.getPaths()
    os.chdir(pathStruct['channelLocation'])
   
    saveName = myInput('Input pkl file stored in the channelLocation directory:\n')
    os.chdir(pathStruct['channelLocation'])

    resultDict = pickle.load(open(saveName+'.pkl','rb'))
    resultDict = pyResultsToMat.fillMissingData(resultDict)
    bestList = pyResultsToMat.getBestDict(resultDict)    
#    strfFile = myInput('Crop file?:\n')
#
#
#    iteratorFile = myInput('Movie set?:\n') 
#    iteratorString = '_aMovieClipSet' + iteratorFile
#    stimIterator = aMovieClipIterator(iteratorFile)
#    cropper = p_stimCropper.stimCropper(0,1,2,
#                                  reshapeOrder = 'F',interp = 'bilinear',
#                                  cropWinIndexStyle='MATLAB')



        
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
    resultDict = dict()
    neuronIdx = 1
    for neuronData in p_utils.myValues(bestList):
        neuronNum = 'neuron' + str(neuronIdx)
        results = predictAndGetExpVaf(neuronData,extra_defaults,transpose)
#        results = predictAndGet1NormalizedCorr(neuronData,extra_defaults,transpose)
#        results = predictAndSSE(neuronData,extra_defaults,transpose)

        resultDict[neuronNum] = results
        resultDict[neuronNum]['neuronName'] = neuronData['neuronName']
        os.chdir(pathStruct['channelLocation'])
        io.savemat(saveName + '_expVaf' +'.mat',resultDict)
#        io.savemat(saveName + '_normCorr' +'.mat',resultDict)
#        io.savemat(saveName + '_PredPower' +'.mat',resultDict)
        neuronIdx = neuronIdx+1;
    
    return

def predictAndGetExpVaf(neuronData,extra_defaults,transpose = False):
    results = dict()
    
    neuronName = neuronData['neuronName']
    stim_options = neuronData['stim_options']

    if 'strfFile' not in stim_options:
        strfFile = extra_defaults['strfFile']
    else:
        strfFile = stim_options['strfFile']
    
    
    (stim,resp,_,_) = clab_utils.getStrfData(strfFile,neuronName,stim_options)
    (estRespSet,regRespSet,predRespSet)= clab_utils.getRespSet(neuronName)
    (model,model_class,model_options) = k_loadModel.loadNeuronModel(neuronData,extra_defaults)
    
    stim = model_class.stimScaler.applyScaleTransform(stim)
    shapeFunc = model_class.shapeStimulus()
    stim = shapeFunc(stim,model_options)
    
    noNLResponse = model.predict(stim)
    NLResponse = p_utils.siaNLPredict(noNLResponse,neuronData['model']['p_opt'])
    
    modelPred = np.squeeze(NLResponse[-1875:])
    results['noiseAdjVaf'] = p_utils.estimateNoiseAdjVaf(predRespSet)
    results['modelAdjVaf'] = p_utils.estimateModelAdjVaf(modelPred,predRespSet)
    results['adjustedVaf']= (results['modelAdjVaf'] / results['noiseAdjVaf'])*100
    results['rawVaf'] = p_utils.vaf(resp[-1875:],modelPred)
    results['modelPredResp'] = modelPred
    results['neuronPredResp'] = resp[-1875:]
    return results

def predictAndGetNormalizedCorr(neuronData,extra_defaults,transpose = False):
    results = dict()
    
    neuronName = neuronData['neuronName']
    stim_options = neuronData['stim_options']

    if 'strfFile' not in stim_options:
        strfFile = extra_defaults['strfFile']
    else:
        strfFile = stim_options['strfFile']
    
    
    (stim,resp,_,_) = clab_utils.getStrfData(strfFile,neuronName,stim_options)
    (estRespSet,regRespSet,predRespSet)= clab_utils.getRespSet(neuronName)
    (model,model_class,model_options) = k_loadModel.loadNeuronModel(neuronData,extra_defaults)
    
    stim = model_class.stimScaler.applyScaleTransform(stim)
    shapeFunc = model_class.shapeStimulus()
    stim = shapeFunc(stim,model_options)
    
    noNLResponse = model.predict(stim)
    NLResponse = p_utils.siaNLPredict(noNLResponse,neuronData['model']['p_opt'])
    
    modelPred = np.squeeze(NLResponse[-1875:])
    results['noiseCorr'] = p_utils.estimateNoiseCorr(predRespSet)
    results['modelCorr'] = p_utils.estimateModelCorr(modelPred,predRespSet)
    results['normalizedCorr']= (results['modelCorr'] / results['noiseCorr'])*100
    results['rawCorr'] = np.corrcoef(resp[-1875:],modelPred)[0,1]
    results['modelPredResp'] = modelPred
    results['neuronPredResp'] = resp[-1875:]
    return results

def predictAndSSE(neuronData,extra_defaults,transpose = False):
    results = dict()
    
    neuronName = neuronData['neuronName']
    stim_options = neuronData['stim_options']

    if 'strfFile' not in stim_options:
        strfFile = extra_defaults['strfFile']
    else:
        strfFile = stim_options['strfFile']
    
    
    (stim,resp,_,_) = clab_utils.getStrfData(strfFile,neuronName,stim_options)
    (estRespSet,regRespSet,predRespSet)= clab_utils.getRespSet(neuronName)
    (model,model_class,model_options) = k_loadModel.loadNeuronModel(neuronData,extra_defaults)
    
    stim = model_class.stimScaler.applyScaleTransform(stim)
    shapeFunc = model_class.shapeStimulus()
    stim = shapeFunc(stim,model_options)
    
    noNLResponse = model.predict(stim)
    NLResponse = p_utils.siaNLPredict(noNLResponse,neuronData['model']['p_opt'])
    
    modelPred = np.squeeze(NLResponse[-1875:])
    results['neuronVar'] = p_utils.estimateNeuronVariance(predRespSet)
    results['modelSSE'] = p_utils.estimateSSE(modelPred,predRespSet)
    results['PredictivePower']= (1.0 - (results['modelSSE'] / results['neuronVar']))*100
    results['rawCorr'] = np.corrcoef(resp[-1875:],modelPred)[0,1]
    results['modelPredResp'] = modelPred
    results['neuronPredResp'] = resp[-1875:]
    return results
    
    
 
if __name__ == '__main__':
    reorgModule = myInput('Was the pickle generated before pySIA reorganization? (y/n) \n')
    if reorgModule == 'y':
        clab_utils.setReorgModules()
        
    algOverPkl()