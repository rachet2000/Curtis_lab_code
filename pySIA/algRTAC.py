# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 12:56:20 2017

@author: phil
"""
from curtis_lab_utils import clab_utils

import p_utils 
import numpy as np
from p_utils import myInput
import os
import re
from scipy import io
import itertools
import pickle
import RTACRunAlgorithms
import sys
def RTACOverAllNeuronStruct(commandLineArgs = None):
    
    pathStruct= clab_utils.getPaths()
    AllNeuronStructLocation = os.path.sep.join((pathStruct['AllNeuronLocation'],'AllNeuronStruct.mat'))
    allNeuronStruct = io.loadmat(AllNeuronStructLocation)
    neuronStruct = np.squeeze(allNeuronStruct['neuronStruct'])    
    
    if len(commandLineArgs) == 1:
        neuronList = myInput('Neuron List :')
        neuronList = re.sub(r'^"|"$', '', neuronList) 
        areaList = myInput('Area List :')
        strfFile = myInput('strf file :')
        saveName = myInput('save name :')
        flagNeuron = myInput('Flag Neuron : ')
    else:
        neuronList = commandLineArgs[1] #If you want to use regex 'or' (i.e. '|')  . please enclose neuronList
                                        # in quotation marks "", so that the | is not interpreted as a pipe
        neuronList = re.sub(r'^"|"$', '', neuronList) #remove enclosing quotation marks
        areaList = commandLineArgs[2]
        strfFile = commandLineArgs[3]
        saveName = commandLineArgs[4]
        flagNeuron = commandLineArgs[5]
        print('neuronList' + neuronList)
        print('areaList' + areaList)
        print('strfFile' + strfFile)
        print('saveName' + saveName)
        
        
    saveName = os.path.sep.join((pathStruct['channelLocation'],saveName))
    
    orderFlag = 0
    if flagNeuron == 'None' or flagNeuron == '' :
        orderFlag = 1
    

    stim_options = dict()
    stim_options['Scaling_Method'] = ('Standardize',0)
    stim_options['cropKern'] = ('Value',30)
    stim_options['Reshape_Order'] = 'F'
    stim_options['Movie_Creation'] = 'PYTHON'
    stim_options['Interpolation'] = 'bilinear'
    
    options = dict() 
    options['Batch_Size'] = [7500]
    options['N_Epochs'] = [700]
    options['Learning_Rate'] = [0.001,0.0005,0.0001]
    options['Frames'] =[list(range(8))]
    options['Reshape_Order'] = stim_options['Reshape_Order']
#    options['RTAC'] = [[1,0],[1,1],[1,2],[1,3],[1,4],[1,5]]
    options['RTAC'] = [[1,0],[1,1],[1,2]]
#    options['Momentum'] = [0.9]
    options['Dense_Init'] =['zero'] #RTAC models work best with weights initialized at zero
    options['ClipNorm'] = [1.0]
    
    neuronDict = dict()
    

    paramList = list(dict(itertools.izip(options, x)) for x in itertools.product(*p_utils.myValues(options)))
    for neuron in neuronStruct:
        neuronNameFull = str(neuron[0][0])
        neuronAreaFull = str(neuron[1][0])
        
        if re.match(flagNeuron,neuronNameFull):
            orderFlag = 1
        if (re.search(neuronList,neuronNameFull) is not None) and ( areaList == neuronAreaFull) and (orderFlag ==1):

            neuronName = neuronNameFull[:-4]

            stim,resp,stim_options['Downsampling_Ratio'],stim_options['cropWin'] = clab_utils.getStrfData(strfFile,neuronName,stim_options )
            stim_options['strfFile'] = strfFile

            neuronDict[neuronName] = []
            for optionSet in paramList:
                print(neuronNameFull)


                currResult = RTACRunAlgorithms.runkRTAC(stim,resp,optionSet)

                
                os.chdir(pathStruct['channelLocation'])
                currResult['stim_options'] = stim_options
                neuronDict[neuronName].append(currResult)
                pickle.dump(neuronDict,open(saveName + '_temp.pkl','wb')) #create temp save in case computer crashes

            pickle.dump(neuronDict,open(saveName + '.pkl','wb'))
            os.remove(saveName + '_temp.pkl') #remove temp file
    return
    
if __name__ == '__main__':
    commandLineArgs = sys.argv
    RTACOverAllNeuronStruct(commandLineArgs)