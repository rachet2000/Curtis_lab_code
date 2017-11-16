# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 13:00:03 2017

@author: amol
"""
from utils.p_utils import optDef
import numpy as np

def kConvNet(given_options):
    options = dict()
        
    options['N_Kern'] = optDef('N_Kern',given_options,1)
    options['Filter_Size'] = optDef('Filter_Size',given_options,13)
    options['Stride'] = optDef('Stride',given_options,1)
    options['Initial_Filter_Weights'] = optDef('Initial_Filter_Weights',given_options,None)
    options['Initial_PReLU']= optDef('Initial_PReLU',given_options,0.5)

    options['L1'] = optDef('L1',given_options,0)
    options['L2'] = optDef('L2',given_options,0)
    options['Pool_Size'] = optDef('Pool_Size',given_options,2)
    options['Initial_Map_Weights'] = optDef('Initial_Map_Weights',given_options,None)
    options['Loss_Function']= optDef('Loss_Function',given_options,'mse')
    return options



def kConvNetDropOut(given_options):
    options = dict()
        
    options['N_Kern'] = optDef('N_Kern',given_options,1)
    options['N_Kern_2'] = optDef('N_Kern_2',given_options,1)
    options['Filter_Size'] = optDef('Filter_Size',given_options,13)
    options['Filter_Size_2'] = optDef('Filter_Size_2',given_options,8)
    options['Stride'] = optDef('Stride',given_options,1)
    options['Initial_Filter_Weights'] = optDef('Initial_Filter_Weights',given_options,None)
    options['p_dropout'] = optDef('p_dropout',given_options,0.5)
    options['p_gaussNoise'] = optDef('p_gaussNoise', given_options, 0.5)    
    options['p_dropout_dense'] = optDef('p_dropout_dense', given_options, 0.5)
    options['p_gaussNoise_dense'] = optDef('p_gaussNoise_dense',given_options, 0.5)
    options['Initial_PReLU']= optDef('Initial_PReLU',given_options,0.5)
     
    options['L1'] = optDef('L1',given_options,0)
    options['L2'] = optDef('L2',given_options,0)
    options['Pool_Size'] = optDef('Pool_Size',given_options,2)
    options['Initial_Map_Weights'] = optDef('Initial_Map_Weights',given_options,None)
    options['Loss_Function']= optDef('Loss_Function',given_options,'poisson') # mse poisson
    options['Optimizer'] = optDef('Optimizer', given_options,'Adagrad')
    return options

def kConvNetTimeSeparable(given_options):
    options = dict()
        
    options['N_Kern'] = optDef('N_Kern',given_options,1)
    options['Filter_Size'] = optDef('Filter_Size',given_options,13)
    options['Stride'] = optDef('Stride',given_options,1)
    options['Initial_Filter_Weights'] = optDef('Initial_Filter_Weights',given_options,None)
    
    options['Initial_Temporal_Weights'] = optDef('Initial_Temporal_Weights',given_options,None)
          
#        options['Initial_Temporal_Weights'] = np.ones((len(given_options['Frames'])))
    options['Initial_PReLU']= optDef('Initial_PReLU',given_options,0.5)
    options['p_dropout'] = optDef('p_dropout',given_options,0.5)
    options['L1'] = optDef('L1',given_options,0)
    options['L2'] = optDef('L2',given_options,0)
    options['Pool_Size'] = optDef('Pool_Size',given_options,2)
    options['Initial_Map_Weights'] = optDef('Initial_Map_Weights',given_options,None)
    options['Loss_Function']= optDef('Loss_Function',given_options,'mse')
    return options
