# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 15:17:26 2015

@author: phil
"""
import os
# NOTE: set tensorflow to use just the first GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from utils import p_utils
import numpy as np
from scipy import io
import sys
import re
import itertools
import pickle
#import runAlgorithms
from utils.p_utils import myInput
from curtis_lab_utils import clab_utils

def singleFile():
    stim,resp = clab_utils.readStrfMat()
    estIdx,regIdx,predIdx = clab_utils.getEstRegPred(resp)
    
    #OPTIONS#
    options = dict()
    options['N_Kern'] = 1
    options['Filter_Size'] = 11
    options['Pool_Size'] = 3
    options['L1'] = 1.5
    options['L2'] = 0.0
    #########
    
    currResult = runAlgorithms.runKConvNet(stim,resp,options,'given')
    
    return currResult
    
    
def goOverAllNeuronStruct(commandLineArgs = None):
    # define runMode for error checking
    lRunMode = list()
    lRunMode.append('sweepAvg')    #sweep averaged
    lRunMode.append('noSweepAvg')   #sweep not averaged

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
        runMode = commandLineArgs[6]
        print('neuronList: ' + neuronList)
        print('areaList: ' + areaList)
        print('strfFile: ' + strfFile)
        print('saveName: ' + saveName)
        if runMode in lRunMode:
            print('runMode:' + str(runMode))
        else:
            raise RuntimeError("ERROR: unknown runMode!")
        
        
    saveName = os.path.sep.join((pathStruct['channelLocation'],saveName))
    
    orderFlag = 0
    if flagNeuron == 'None' or flagNeuron == '':
        orderFlag = 1
    

    stim_options = dict()
    stim_options['Scaling_Method'] = ('Standardize',0)
    stim_options['cropKern'] = ('Value',30)
    stim_options['Reshape_Order'] = 'F'
    stim_options['Movie_Creation'] = 'PYTHON'
    stim_options['Interpolation'] = 'bilinear'
    
    nRepeat = 1

    options = dict() 
    
    #############################
    ### All Algorithm Options ###
    #############################
#    options['Batch_Size'] = 1875 #now the batch size is set by the algorithm
#    options['Learning_Rate'] = 0.01
#    options['N_Epochs'] = 15
    options['Frames'] =[list(range(8))]

#    options['Frames'] =[1,2,3]
#    options['L1'] = 0.05                               #TESTING
#    options['L2'] = 0       
    options['Check_On_Off'] = [False]
    options['N_Epochs'] = [700]
#    options['initial_alpha'] = 0.5
#    options['fix_alpha'] = 0.0
#    options['Momentum'] = 0.00
    options['Reshape_Order'] = stim_options['Reshape_Order']
#    cropKern = ('Downsample_Minimum',40)
    
    
    #########################
    ### All kConv Options ###
    #########################
    options['Initial_PReLU'] = [0.5]


    options['N_Kern'] = [1]
#    options['Filter_Size'] = [4,6,8,10,12,14,16,18,20,22,24,26,28]
    options['Filter_Size'] = [11]
#    options['Pool_Size'] = [2]

#    options['Pool_Size'] = [2,3]
    options['Pool_Size'] = [3]
#    options['L1'] = [1.5,8.0]
    options['L1'] = [0.0]
#    options['L1'] = [1.5]
    options['L2'] = [0.0]
    
    options['Stride'] = [1]
    options['Learning_Rate'] = [0.001]
    
    ############################
    ### Experimental Options ###
    ############################
#    options['Input_Dropout'] = [0.0,0.5]
    options['Map_Dropout'] = [0.0]
#    options['Sigma_Div'] = [1.0,2.0]
#    options['Sigma_Div'] = [1.0]
#    options['Sigma_Reg_L2'] = [0.0,1e-7,0.0001]
    
#    options['Activity_L1'] = [0.0]
    options['Activity_L1'] = [0.0]
#    options['Activity_L2'] = [0.0,1e-5,1,100]
#    options['Gaussian_Layer_Scale'] = [0.1,0.5,1.0]
    neuronDict = dict()
    

    paramList = list(dict(zip(options, x)) for x in itertools.product(*p_utils.myValues(options)))
    if neuronStruct.size == 1:
    	neuronStruct = list([neuronStruct.tolist()])  # convert 0-d array to list for the following loop
    for neuron in neuronStruct:
        neuronNameFull = str(neuron[0][0])
        neuronAreaFull = str(neuron[1][0])
        
        if re.match(flagNeuron,neuronNameFull):
            orderFlag = 1
        if not orderFlag == 1:
            print("orderFlag: " + str(orderFlag))
        if re.search(neuronList,neuronNameFull) is None:
        	print("neuronList :" + str(neuronList) + " neuronNameFull: " + str(neuronNameFull))
            #print("areaList == neuronAreaFull: " + str(areaList == neuronAreaFull))
        if (re.search(neuronList,neuronNameFull) is not None) and ( areaList == neuronAreaFull) and (orderFlag ==1):
            print("Processing neuron: " + str(neuronNameFull))
            neuronName = neuronNameFull[:-4]
            stim,resp,stim_options['Downsampling_Ratio'],stim_options['cropWin'] = clab_utils.getStrfData(strfFile,neuronName,stim_options)
            stim_options['strfFile'] = strfFile
            stim_options['neuronArea'] = neuronAreaFull
            neuronDict[neuronName] = []
            for optionSet in paramList:
                print(neuronNameFull)
                for iRep in range(nRepeat):
                    # currResult = runAlgorithms.runConvNet(stim,resp,optionSet,'given')
                    # currResult = runAlgorithms.runPhaseNeuralNet(stim,resp,optionSet,'given')
                    # currResult = runAlgorithms.runSTARegression(stim,resp,optionSet,'given')
                    # currResult = runAlgorithms.runAlphaNet(stim,resp,optionSet,'given')
                    # currResult = runAlgorithms.runAlphaConvNet(stim,resp,optionSet,'given')
                    # currResult = runAlgorithms.runPPAConvNet(stim,resp,optionSet,'given')
                    # currResult = runAlgorithms.runKConvNet(stim,resp,optionSet)
                    currResult = runAlgorithms.runKConvGaussNet(stim,resp,optionSet)
                    # currResult = runAlgorithms.runKConvGaussEXP(stim,resp,optionSet)
                    # currResult = runAlgorithms.runKConvDOGNet(stim,resp,optionSet)
                    # currResult = runAlgorithms.runKConvTwoGaussNet(stim,resp,optionSet)
                    # currResult = runAlgorithms.runKConvTwoAffineNet(stim,resp,optionSet)
                    # currResult = runAlgorithms.runKConvSplitNet(stim,resp,optionSet)
                    # currResult = runAlgorithms.runKOnOffConvNet(stim,resp,optionSet)
                    # currResult = runAlgorithms.runKRegression(stim,resp,optionSet)
                    # currResult = runAlgorithms.runKLNLSTM(stim,resp,optionSet)
                    # currResult = runAlgorithms.runKSTAintoConvNetTimeSeparable(stim,resp,optionSet)
                    # currResult = runAlgorithms.runKConvNetTimeSeparable(stim,resp,optionSet)
                    # currResult = runAlgorithms.runKRadialNet(stim,resp,optionSet)
                    # currResult = runAlgorithms.runKSTAintoConvNetTimeSeparableDOG(stim,resp,optionSet)
                    # currResult = runAlgorithms.runKRadialNetDOG(stim,resp,optionSet)
                    currResult['stim_options'] = stim_options
                    if iRep > 0:
                        currResult['iRep'] = iRep
                    neuronDict[neuronName].append(currResult)
                    pickle.dump(neuronDict,open(saveName + '_temp.pkl','wb')) #create temp save in case computer crashes
            pickle.dump(neuronDict,open(saveName + '.pkl','wb'))
            os.remove(saveName + '_temp.pkl') #remove temp file
    return


if __name__ == '__main__':
    commandLineArgs = sys.argv
    goOverAllNeuronStruct(commandLineArgs)
    

###    
#    nKernList = [1]
##    filterSizeList = [5,7,10]
##    filterSizeList = [5,11,15]
#    filterSizeList = [11]
##    poolSizeList = [3,5]
##    poolSizeList = [1,2,3]
#    poolSizeList = [2]
##    poolSizeList = [1]
##    midModeList = ['lambda x:x']
#    midModeList = ['N/A']
##    endModeList = ['lambda x:x','lambda x:T.nnet.relu(x)']
#    endModeList = ['N/A']
##    l1List = [ 0.0005,0.001,0.005,0.01] 
##    l1List = [1.5,8.0]
#    l1List = [1.5]
##    l1List = [0.0]
##    l1List = [0.001,0.1, 1.0, 10.0, 50.0,100.0] 
##    l1List = [ 8.0] 
##    l2List = [50.0]
#    l2List = [0.0]
#    stride = [1]
#    learningRate = [0.001]
#    paramList = list(itertools.product(nKernList, filterSizeList,poolSizeList,midModeList,endModeList,l1List,l2List,stride,learningRate))
##    
