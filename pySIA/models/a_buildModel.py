# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 13:02:29 2017

@author: amol
"""

import keras
import theano
import numpy as np
from keras.layers import Dense, Dropout, Activation, Flatten,advanced_activations, Reshape
from keras.layers import Convolution2D, AveragePooling2D, Convolution1D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.regularizers import l1
from keras.utils.conv_utils import conv_output_length
from keras.layers.noise import GaussianDropout
from extra_layers import k_layers
from extra_layers import a_layers
from keras.layers.advanced_activations import PReLU
#from keras.layers.advanced_activations import ParametricSoftplus as Psoftplus
from keras.constraints import maxnorm


def buildkConvNet(options):
    numRows =  options['Input_Shape'][1]
    numCols =  options['Input_Shape'][2]
    assert numRows == numCols
    numFrames = options['Input_Shape'][0]
    #Variable Setup
    #since the lambdas were originally estimated as regularization
    #the mean weight value of the map layer, we convert it here so it is a regularizer 
    #on the sum of the weights in the map layer.
    convImageSize = conv_output_length(numRows,options['Filter_Size'],'valid',options['Stride'])
    downsampImageSize = conv_output_length(convImageSize,options['Pool_Size'],'valid',options['Pool_Size'])
    sum_L1_lambda = options['L1'] /(downsampImageSize**2)
    sum_L2_lambda = options['L2'] /(downsampImageSize**2) #not used
    regularizer = keras.regularizers.WeightRegularizer(l1=sum_L1_lambda, l2=sum_L2_lambda)
    stride = (options['Stride'],options['Stride'])
    
    #################
    #MODEL CREATION #
    #################
    
    inputLayer =keras.layers.Input(shape=options['Input_Shape'])
    model_conv1 = Convolution2D(options['N_Kern'], options['Filter_Size'], options['Filter_Size'], 
                            border_mode='valid',
                            input_shape=options['Input_Shape'],
                            init='glorot_normal',
                            weights = options['Initial_Filter_Weights'],
                            subsample = stride)(inputLayer)
                        
    model_prelu2 = k_layers.singlePReLU(weights=options['Initial_PReLU'])(model_conv1)
    model_pool3 = AveragePooling2D(pool_size=(options['Pool_Size'], options['Pool_Size']))(model_prelu2)

    model_flat4 = Flatten()(model_pool3)
    model_dense5 = Dense(1,W_regularizer=regularizer,init='glorot_normal',
                            weights = options['Initial_Map_Weights'])(model_flat4)
    output = Activation('relu')(model_dense5)
    model = keras.models.Model(input=inputLayer,output =output)
#    
    return model

def buildkConvNetDropOut(options):
    numRows =  options['Input_Shape'][1]
    numCols =  options['Input_Shape'][2]
    assert numRows == numCols
    numFrames = options['Input_Shape'][0]
    #Variable Setup
    #since the lambdas were originally estimated as regularization
    #the mean weight value of the map layer, we convert it here so it is a regularizer 
    #on the sum of the weights in the map layer.
    convImageSize = conv_output_length(numRows,options['Filter_Size'],'valid',options['Stride'])
    downsampImageSize = conv_output_length(convImageSize,options['Pool_Size'],'valid',options['Pool_Size']) * options['N_Kern']
#    L2_filter = options['L2']/(options['Filter_Size']**2)    
#    L2_map = options['L2'] /(downsampImageSize**2)
    L2_filter = options['L2']
    L2_map = options['L2']   
    L1_map = options['L1']    
    
    W_regularizer_filter = keras.regularizers.WeightRegularizer(l2 = L2_filter)    
    W_regularizer_map = keras.regularizers.WeightRegularizer(l2=L2_map)
    A_regularizer_map = keras.regularizers.ActivityRegularizer(l1=L1_map)
    stride = (options['Stride'],options['Stride'])
    
    #################
    #MODEL CREATION #
    #################
    
    inputLayer =keras.layers.Input(shape=options['Input_Shape'])
    model_input_dropout = Dropout(options['p_dropout'])(inputLayer)
#    model_input_Noise = GaussianDropout(options['p_gaussNoise'])(model_input_dropout)    
    model_input_Noise = a_layers.PoissonNoise()(model_input_dropout)    
    
    model_conv = Convolution2D(options['N_Kern'], options['Filter_Size'], options['Filter_Size'], 
                            border_mode='valid',
                            input_shape=options['Input_Shape'],
                            init='glorot_normal', W_regularizer = W_regularizer_filter, W_constraint = maxnorm(m=1, axis =[1,2,3]),
                            subsample = stride)(model_input_Noise)
    model_dropout = Dropout(options['p_dropout_dense'])(model_conv)
    model_Noise = GaussianDropout(options['p_gaussNoise_dense'])(model_dropout)
#    model_Noise = a_layers.PoissonNoise()(model_dropout)#using poisson noise here leads to error value NaN 

#    model_BatchNorm = BatchNormalization(epsilon=0.001, mode=0, axis=1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None)(model_Noise)
#    model_NL = Psoftplus(shared_axes=[2,3])(model_Noise)
#    model_NL = Activation('softplus')(model_Noise)
    
    model_NL =  PReLU(init='zero', weights=None, shared_axes=[2,3])(model_Noise)                      
#    model_NL = k_layers.singlePReLU(weights=options['Initial_PReLU'])(model_Noise)
    model_pool = AveragePooling2D(pool_size=(options['Pool_Size'], options['Pool_Size']))(model_NL)
#    model_pool = MaxPooling2D(pool_size = (options['Pool_Size'], options['Pool_Size']))(model_NL)

    # Second Convolutional layer
#    model_conv2 = Convolution2D(options['N_Kern_2'], options['Filter_Size_2'], options['Filter_Size_2'], border_mode ='valid',
#                                init = 'glorot_normal', W_regularizer = W_regularizer_filter, W_constraint = maxnorm(m=1,axis=[1,2,3]),
#                                subsample = stride, activation = 'relu')(model_pool)    
#    model_dropout2 = Dropout(options['p_dropout_dense'])(model_conv2)

    model_flat = Flatten()(model_pool)
    model_dense = Dense(1,W_regularizer=W_regularizer_map, activity_regularizer = A_regularizer_map,init='glorot_normal',
                            W_constraint = maxnorm(m=1, axis =[0]))(model_flat)
#    model_BatchNormMap = BatchNormalization(epsilon=0.001, mode=0, axis=1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None)(model_dense)
    # Additional LGN Layer
#    model_relu = Activation('relu')(model_dense)
#    model_denseDropout = Dropout(options['p_dropout'])(model_relu)
#    model_dense6 = Dense(1,init='glorot_normal',
#                         W_constraint = maxnorm(m=1, axis =[0]))(model_denseDropout)
    
    
    
    output = Psoftplus()(model_dense)
    model = keras.models.Model(input=inputLayer,output =output)
#    
    return model
        
def buildkConvNetTimeSeparable(options):
    numRows =  options['Input_Shape'][1]
    numCols =  options['Input_Shape'][2]
    assert numRows == numCols
    numFrames = len(options['Frames'])
    
    
    convImageSize = conv_output_length(numRows,options['Filter_Size'],'valid',options['Stride'])
    downsampImageSize = conv_output_length(convImageSize,options['Pool_Size'],'valid',options['Pool_Size'])
    sum_L1_lambda = options['L1'] /(downsampImageSize**2)
    sum_L2_lambda = options['L2'] /(downsampImageSize**2) #not used
    regularizer = keras.regularizers.WeightRegularizer(l1=sum_L1_lambda, l2=sum_L2_lambda)
    stride = (options['Stride'],options['Stride'])
    
    #################
    #MODEL CREATION #
    #################
    
    #create the convolutional layer
    convolutionLayer = Convolution2D(options['N_Kern'],
                                     options['Filter_Size'],
                                     options['Filter_Size'],
                                     border_mode='valid',
                                     input_shape=options['Input_Shape'],
                                     init='glorot_normal',
                                     subsample = stride)
    
    #create separate inputs for each frame to the convolutional layer
    frameInput = [None]*numFrames
    weightedFrameConv = [None]*numFrames
    for frame in range(numFrames):
        frameInput[frame] = keras.layers.Input(shape=options['Input_Shape'])
        frameConv = convolutionLayer(frameInput[frame])
        weightedFrameConv[frame]  = k_layers.singleWeight(init = 'glorot_normal')(frameConv)
    
    mergedConvLayer = keras.layers.merge(weightedFrameConv,mode='sum')    
    
    #model after the merger
    model_PReLU = k_layers.singlePReLU(weights=options['Initial_PReLU'])(mergedConvLayer)
    model_dropout = Dropout(options['p_dropout'])(model_PReLU)
    model_Pool = AveragePooling2D(pool_size=(options['Pool_Size'], options['Pool_Size']))(model_dropout)
    model_Flatten = Flatten()(model_Pool)
    model_Map = Dense(1,W_regularizer=regularizer,init='glorot_normal')(model_Flatten)
    output = Activation('relu')(model_Map)

    
    model = keras.models.Model(input=frameInput,output =output)    
    
    return model
