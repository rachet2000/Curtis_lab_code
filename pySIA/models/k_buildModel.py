# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 13:45:49 2016

@author: phil
"""
import keras
import numpy as np
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Convolution2D, AveragePooling2D, Convolution1D
from keras.models import Sequential
from keras.utils.conv_utils import conv_output_length
from extra_layers import k_layers
from keras.layers.core import ActivityRegularization
from keras.layers.advanced_activations import PReLU

def buildkRegression(options):    
    numFeatures =  options['Input_Shape'][0]
    
    sum_L1_lambda = options['L1'] /(numFeatures)
    sum_L2_lambda = options['L2'] /(numFeatures) #not used
    regularizer = keras.regularizers.WeightRegularizer(l1=sum_L1_lambda, l2=sum_L2_lambda)

    
    #################
    #MODEL CREATION #
    #################
    model = Sequential()
    model.add(Dense(1,W_regularizer=regularizer,init=options['Dense_Init'],
                            weights = options['Initial_Weights'],input_dim=numFeatures))
    model.add(Activation('relu'))
    
    return model
    

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

    preluWeight = np.array(options['Initial_PReLU'],ndmin=3)
    model_prelu2 = PReLU(weights=[preluWeight],
                         shared_axes=[1, 2, 3])(model_conv1)
    #model_prelu2 = k_layers.singlePReLU(weights=options['Initial_PReLU'])(model_conv1)
    model_pool3 = AveragePooling2D(pool_size=(options['Pool_Size'], options['Pool_Size']))(model_prelu2)

    model_flat4 = Flatten()(model_pool3)
    model_dense5 = Dense(1, W_regularizer=regularizer,init='glorot_normal',
                            weights = options['Initial_Map_Weights'])(model_flat4)
    output = Activation('relu')(model_dense5)
    activity_reg_output = ActivityRegularization(l1 =options['Activity_L1'],l2= options['Activity_L2'])(output)
    model = keras.models.Model(input=inputLayer, output =activity_reg_output)
    return model

def buildkConvGaussNet(options):
    numRows =  options['Input_Shape'][1]
    numCols =  options['Input_Shape'][2]
    assert numRows == numCols
    numFrames = options['Input_Shape'][0]
    
    convImageSize = conv_output_length(numRows,options['Filter_Size'],'valid',options['Stride'])
    downsampImageSize = conv_output_length(convImageSize,options['Pool_Size'],'valid',options['Pool_Size'])
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
    preluWeight = np.array(options['Initial_PReLU'], ndmin=3)
    model_prelu2 = PReLU(weights=[preluWeight],
                         shared_axes=[1, 2, 3])(model_conv1)
    # model_prelu2 = k_layers.singlePReLU(weights=options['Initial_PReLU'])(model_conv1)
    model_pool3 = AveragePooling2D(pool_size=(options['Pool_Size'], options['Pool_Size']))(model_prelu2)

    model_gaussian4 = k_layers.gaussian2dMapLayer(
        (downsampImageSize,downsampImageSize),
        init_mean=options['Initial_Gaussian_Mean'],
        init_sigma = options['Initial_Gaussian_Sigma'],
        init_sigma_div = options['Sigma_Div'],
        sigma_regularizer_l2 = keras.regularizers.l2(options['Sigma_Reg_L2']))(model_pool3)
        
    model_dense5 = Dense((1),weights=options['Initial_Dense_Values'])(model_gaussian4)
    output = Activation('relu')(model_dense5)
    activity_reg_output = ActivityRegularization(l1 =options['Activity_L1'],l2= options['Activity_L2'])(output)

    model = keras.models.Model(input=inputLayer,output =activity_reg_output)
    
    return model

def buildkConvDOGNet(options):
    numRows =  options['Input_Shape'][1]
    numCols =  options['Input_Shape'][2]
    assert numRows == numCols
    numFrames = options['Input_Shape'][0]

    convImageSize = conv_output_length(numRows,options['Filter_Size'],'valid',options['Stride'])
    downsampImageSize = conv_output_length(convImageSize,options['Pool_Size'],'valid',options['Pool_Size'])
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

    model_dog4 = k_layers.DOG2dMapLayer(
        (downsampImageSize,downsampImageSize),
        init_pos_mean = options['Initial_Gaussian_Pos_Mean'],
        init_pos_sigma = options['Initial_Gaussian_Pos_Sigma'],
        init_neg_mean = options['Initial_Gaussian_Neg_Mean'],
        init_neg_sigma = options['Initial_Gaussian_Neg_Sigma'],
        init_scale = options['Initial_DOG_Scale']
        )(model_pool3)
        
    model_dense5 = Dense((1),weights=options['Initial_Dense_Values'])(model_dog4)
    output = Activation('relu')(model_dense5)
    activity_reg_output = ActivityRegularization(l1 =options['Activity_L1'],l2= options['Activity_L2'])(output)

    model = keras.models.Model(input=inputLayer,output =activity_reg_output)
   
    
    return model
    
def buildkConvGaussEXP(options):
    numRows =  options['Input_Shape'][1]
    numCols =  options['Input_Shape'][2]
    assert numRows == numCols
    numFrames = options['Input_Shape'][0]
    
    convImageSize = conv_output_length(numRows,options['Filter_Size'],'valid',options['Stride'])
    downsampImageSize = conv_output_length(convImageSize,options['Pool_Size'],'valid',options['Pool_Size'])
    stride = (options['Stride'],options['Stride'])
    
    #################
    #MODEL CREATION #
    #################
    inputLayer =keras.layers.Input(shape=options['Input_Shape'])
    inputDropout = keras.layers.Dropout(p =options['Input_Dropout'])(inputLayer)
    model_conv1 = Convolution2D(options['N_Kern'], options['Filter_Size'], options['Filter_Size'], 
                            border_mode='valid',
                            input_shape=options['Input_Shape'],
                            init='glorot_normal',
                            weights = options['Initial_Filter_Weights'],
                            subsample = stride)(inputDropout)
                            
    model_prelu2 = k_layers.singlePReLU(weights=options['Initial_PReLU'])(model_conv1)
    model_pool3 = AveragePooling2D(pool_size=(options['Pool_Size'], options['Pool_Size']))(model_prelu2)

    mapDropout = keras.layers.Dropout(p =options['Map_Dropout'])(model_pool3)

    model_gaussian4 = k_layers.gaussian2dMapLayerNormalized(
        (downsampImageSize,downsampImageSize),
        init_mean=options['Initial_Gaussian_Mean'],
        init_sigma = options['Initial_Gaussian_Sigma'],
        init_sigma_div = options['Sigma_Div'],
        scale = options['Gaussian_Layer_Scale'],
        sigma_regularizer_l2 = keras.regularizers.l2(options['Sigma_Reg_L2']))(mapDropout)
        
    model_dense5 = Dense((1),weights=options['Initial_Dense_Values'])(model_gaussian4)
    output = keras.layers.advanced_activations.ParametricSoftplus()(model_dense5)
    activity_reg_output = ActivityRegularization(l1 =options['Activity_L1'],l2= options['Activity_L2'])(output)

    model = keras.models.Model(input=inputLayer,output =activity_reg_output)
    
    return model
    
def buildkConvTwoGaussNet(options):
    numRows =  options['Input_Shape'][1]
    numCols =  options['Input_Shape'][2]
    assert numRows == numCols
    numFrames = options['Input_Shape'][0]

    convImageSize = conv_output_length(numRows,options['Filter_Size'],'valid',options['Stride'])
    downsampImageSize = conv_output_length(convImageSize,options['Pool_Size'],'valid',options['Pool_Size'])
    stride = (options['Stride'],options['Stride'])
    #################
    #MODEL CREATION #
    #################
    
    inputLayer =keras.layers.Input(shape=(numFrames, numRows, numCols))
    
    posLayer1 = Convolution2D(options['N_Kern'], options['Filter_Size'], options['Filter_Size'], 
                            border_mode='valid',
                            input_shape=options['Input_Shape'],
                            init='glorot_normal',
                            weights = options['Initial_Filter_Weights_Pos'],
                            subsample = stride)(inputLayer)
    posLayer2 = k_layers.singlePReLU(weights=options['Initial_PReLU_Pos'])(posLayer1)
    posLayer3 = AveragePooling2D(pool_size=(options['Pool_Size'], options['Pool_Size']))(posLayer2)
    posLayer4 = k_layers.gaussian2dMapLayer(
        (downsampImageSize,downsampImageSize),init_mean=options['Initial_Gaussian_Pos_Mean'],
        init_sigma = options['Initial_Gaussian_Pos_Sigma'])(posLayer3)
    
    negLayer1 = Convolution2D(options['N_Kern'], options['Filter_Size'], options['Filter_Size'], 
                            border_mode='valid',
                            input_shape=options['Input_Shape'],
                            init='glorot_normal',
                            weights = options['Initial_Filter_Weights_Neg'],
                            subsample = stride)(inputLayer)
    negLayer2 = k_layers.singlePReLU(weights=options['Initial_PReLU_Neg'])(negLayer1)
    negLayer3 = AveragePooling2D(pool_size=(options['Pool_Size'], options['Pool_Size']))(negLayer2)
    negLayer4 = k_layers.gaussian2dMapLayer(
        (downsampImageSize,downsampImageSize),init_mean=options['Initial_Gaussian_Neg_Mean'],
        init_sigma = options['Initial_Gaussian_Neg_Sigma'])(negLayer3)
    
    mergedLayer = keras.layers.merge([posLayer4, negLayer4],mode='concat',concat_axis=1)
    model5 = Dense(1,weights=options['Initial_Dense_Values'])(mergedLayer)
    output = Activation('relu')(model5)
    model = keras.models.Model(input=inputLayer,output =output)
    return model

def buildkConvTwoAffineNet(options):
    
    numRows =  options['Input_Shape'][1]
    numCols =  options['Input_Shape'][2]
    assert numRows == numCols
    numFrames = options['Input_Shape'][0]


    convImageSize = conv_output_length(numRows,options['Filter_Size'],'valid',options['Stride'])
    downsampImageSize = conv_output_length(convImageSize,options['Pool_Size'],'valid',options['Pool_Size'])
    sum_L1_lambda = options['L1'] /downsampImageSize 
    sum_L2_lambda = options['L2'] /downsampImageSize #not used
    regularizer = keras.regularizers.WeightRegularizer(l1=sum_L1_lambda, l2=sum_L2_lambda)
    
    stride = (options['Stride'],options['Stride'])
    #################
    #MODEL CREATION #
    #################
    
    inputLayer =keras.layers.Input(shape=(numFrames, numRows, numCols))
    
    posLayer1 = Convolution2D(options['N_Kern'], options['Filter_Size'], options['Filter_Size'], 
                            border_mode='valid',
                            input_shape=options['Input_Shape'],
                            init='glorot_normal',
                            weights = options['Initial_Filter_Weights_Pos'],
                            subsample = stride)(inputLayer)
    posLayer2 = k_layers.singlePReLU(weights=options['Initial_PReLU_Pos'])(posLayer1)
    posLayer3 = AveragePooling2D(pool_size=(options['Pool_Size'], options['Pool_Size']))(posLayer2)
    posLayer4 = k_layers.gaussian2dMapLayer((downsampImageSize,downsampImageSize))(posLayer3)
    posLayer4 = keras.layers.Flatten()(posLayer3)
    posLayer5 = Dense(1,W_regularizer=regularizer,init='glorot_normal',
                            weights = options['Initial_Map_Weights_Pos'])(posLayer4)
    
    negLayer1 = Convolution2D(options['N_Kern'], options['Filter_Size'], options['Filter_Size'], 
                            border_mode='valid',
                            input_shape=options['Input_Shape'],
                            init='glorot_normal',
                            weights = options['Initial_Filter_Weights_Neg'],
                            subsample = stride)(inputLayer)
    negLayer2 = k_layers.singlePReLU(weights=options['Initial_PReLU_Neg'])(negLayer1)
    negLayer3 = AveragePooling2D(pool_size=(options['Pool_Size'], options['Pool_Size']))(negLayer2)
    negLayer4 = keras.layers.Flatten()(negLayer3)
    negLayer5 = Dense(1,W_regularizer=regularizer,init='glorot_normal',
                            weights = options['Initial_Map_Weights_Neg'])(negLayer4)
    
    mergedLayer = keras.layers.merge([posLayer5, negLayer5],mode='concat',concat_axis=1)
    model5 = Dense(1,weights=options['Initial_Dense_Values'])(mergedLayer)
    output = Activation('relu')(model5)
    model = keras.models.Model(input=inputLayer,output =output)
    
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
                                     weights = options['Initial_Filter_Weights'],
                                     subsample = stride)
    
    #create separate inputs for each frame to the convolutional layer
    frameInput = [None]*numFrames
    weightedFrameConv = [None]*numFrames
    for frame in range(numFrames):
        frameInput[frame] = keras.layers.Input(shape=options['Input_Shape'])
        frameConv = convolutionLayer(frameInput[frame])
        weightedFrameConv[frame]  = k_layers.singleWeight(weights=options['Initial_Temporal_Weights'][frame])(frameConv)
    
    mergedConvLayer = keras.layers.merge(weightedFrameConv,mode='sum')    
    
    #model after the merger
    model_PReLU = k_layers.singlePReLU(weights=options['Initial_PReLU'])(mergedConvLayer)
    model_Pool = AveragePooling2D(pool_size=(options['Pool_Size'], options['Pool_Size']))(model_PReLU)
    model_Flatten = Flatten()(model_Pool)
    model_Map = Dense(1,W_regularizer=regularizer,init='glorot_normal',
                            weights = options['Initial_Map_Weights'])(model_Flatten)
    output = Activation('relu')(model_Map)

    
    model = keras.models.Model(input=frameInput,output =output)    
    
    return model

def buildkConvNetTimeSeparableDOG(options):
    numRows =  options['Input_Shape'][1]
    numCols =  options['Input_Shape'][2]
    assert numRows == numCols
    numFrames = len(options['Frames'])
    
    convImageSize = conv_output_length(numRows,options['Filter_Size'],'valid',options['Stride'])
    downsampImageSize = conv_output_length(convImageSize,options['Pool_Size'],'valid',options['Pool_Size'])
    stride = (options['Stride'],options['Stride'])
    #################
    #MODEL CREATION #
    #################
    
    #create the convolutional layer
    convolutionLayer = Convolution2D(options['N_Kern'], options['Filter_Size'], options['Filter_Size'], 
                    border_mode='valid', input_shape=options['Input_Shape'],init='glorot_normal',weights = options['Initial_Filter_Weights'],subsample = stride)
    
    #create separate inputs for each frame to the convolutional layer
    frameInput = [None]*numFrames
    weightedFrameConv = [None]*numFrames
    for frame in range(numFrames):
        frameInput[frame] = keras.layers.Input(shape=options['Input_Shape'])
        frameConv = convolutionLayer(frameInput[frame])
        weightedFrameConv[frame]  = k_layers.singleWeight(weights=options['Initial_Temporal_Weights'][frame])(frameConv)
    
    mergedConvLayer = keras.layers.merge(weightedFrameConv,mode='sum')    
    
    #model after the merger
    model_PReLU = k_layers.singlePReLU(weights=options['Initial_PReLU'])(mergedConvLayer)
    model_Pool = AveragePooling2D(pool_size=(options['Pool_Size'], options['Pool_Size']))(model_PReLU)
    model_Map = k_layers.DOG2dMapLayer(
        (downsampImageSize,downsampImageSize),
        init_pos_mean = options['Initial_Gaussian_Pos_Mean'],
        init_pos_sigma = options['Initial_Gaussian_Pos_Sigma'],
        init_neg_mean = options['Initial_Gaussian_Neg_Mean'],
        init_neg_sigma = options['Initial_Gaussian_Neg_Sigma'],
        init_scale = options['Initial_DOG_Scale']
        )(model_Pool)
    model_Linout = Dense((1),weights=options['Initial_Dense_Values'])(model_Map)
    output = Activation('relu')(model_Linout)

    
    model = keras.models.Model(input=frameInput,output =output)    
    
    return model

def buildkRadialNet(options):
    numSubImages =  options['Input_Shape'][0]
    radialProfileLength =  options['Input_Shape'][1]
    convImgRows =int(np.sqrt(numSubImages))
    numFrames = len(options['Frames'])
    
    convImageSize = convImgRows
    downsampImageSize = conv_output_length(convImageSize,options['Pool_Size'],'valid',options['Pool_Size'])
    sum_L1_lambda = options['L1'] /(downsampImageSize**2) 
    sum_L2_lambda = options['L2'] /(downsampImageSize**2) #not used
    regularizer = keras.regularizers.WeightRegularizer(l1=sum_L1_lambda, l2=sum_L2_lambda)
    #################
    #MODEL CREATION #
    #################
    
    #create the convolutional layer
    convolutionLayer = Convolution1D(options['N_Kern'], 1, 
                            border_mode='valid',
                            input_shape=(numSubImages, radialProfileLength),
                            init='glorot_normal',
                            weights = options['Initial_Filter_Weights'])
    
    #create separate inputs for each frame to the convolutional layer
    frameInput = [None]*numFrames
    weightedFrameConv = [None]*numFrames
    for frame in range(numFrames):
        frameInput[frame] = keras.layers.Input(shape=(numSubImages, radialProfileLength))
        frameConv = convolutionLayer(frameInput[frame])
        weightedFrameConv[frame]  = k_layers.singleWeight(weights=options['Initial_Temporal_Weights'][frame])(frameConv)
    
    mergedConvLayer = keras.layers.merge(weightedFrameConv,mode='sum')    
    
    #model after the merger
    model_PReLU = k_layers.singlePReLU(weights=options['Initial_PReLU'])(mergedConvLayer)
    model_Reshape2d = Reshape((1,convImgRows,convImgRows))(model_PReLU)
    model_Pool = AveragePooling2D(pool_size=(options['Pool_Size'], options['Pool_Size']))(model_Reshape2d)
    model_Flatten = Flatten()(model_Pool)
    model_Map = Dense(1,init='glorot_normal',
                      W_regularizer=regularizer,
                      weights = options['Initial_Map_Weights'])(model_Flatten)
    output = Activation('relu')(model_Map)

    
    model = keras.models.Model(input=frameInput,output =output)
    return model
    
def buildkRadialNetDOG(options):

    numSubImages =  options['Input_Shape'][0]
    radialProfileLength =  options['Input_Shape'][1]
    convImgRows =int(np.sqrt(numSubImages))
    numFrames = len(options['Frames'])
    convImageSize = convImgRows
    downsampImageSize = conv_output_length(convImageSize,options['Pool_Size'],'valid',options['Pool_Size'])
    #################
    #MODEL CREATION #
    #################
    
    #create the convolutional layer
    convolutionLayer = Convolution1D(options['N_Kern'], 1, 
                    border_mode='valid', input_shape=(numSubImages, radialProfileLength),init='glorot_normal',weights = options['Initial_Filter_Weights'])
    
    #create separate inputs for each frame to the convolutional layer
    frameInput = [None]*numFrames
    weightedFrameConv = [None]*numFrames
    for frame in range(numFrames):
        frameInput[frame] = keras.layers.Input(shape=(numSubImages, radialProfileLength))
        frameConv = convolutionLayer(frameInput[frame])
        weightedFrameConv[frame]  = k_layers.singleWeight(weights=options['Initial_Temporal_Weights'][frame])(frameConv)
    
    mergedConvLayer = keras.layers.merge(weightedFrameConv,mode='sum')    
    
    #model after the merger
    model_PReLU = k_layers.singlePReLU(weights=options['Initial_PReLU'])(mergedConvLayer)
    model_Reshape2d = Reshape((1,convImgRows,convImgRows))(model_PReLU)
    model_Pool = AveragePooling2D(pool_size=(options['Pool_Size'], options['Pool_Size']))(model_Reshape2d)
    model_Map = k_layers.DOG2dMapLayer(
        (downsampImageSize,downsampImageSize),
        init_pos_mean = options['Initial_Gaussian_Pos_Mean'],
        init_pos_sigma = options['Initial_Gaussian_Pos_Sigma'],
        init_neg_mean = options['Initial_Gaussian_Neg_Mean'],
        init_neg_sigma = options['Initial_Gaussian_Neg_Sigma'],
        init_scale = options['Initial_DOG_Scale']
        )(model_Pool)
    model_Linout = Dense((1),weights=options['Initial_Dense_Values'])(model_Map)
    output = Activation('relu')(model_Linout)

   
    model = keras.models.Model(input=frameInput,output =output)

    return model
    
def buildkOnOffConvNet(options):
    numRows =  options['Input_Shape'][1]
    numCols =  options['Input_Shape'][2]
    assert numRows == numCols
    numFrames = options['Input_Shape'][0]
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
    
    modelp1 = k_layers.singleTwoSidePReLU(weights=options['Initial_TwoSided_PReLU'])(inputLayer)
    modelp2 = Convolution2D(options['N_Kern'],
                                     options['Filter_Size'],
                                     options['Filter_Size'],
                                     border_mode='valid',
                                     input_shape=options['Input_Shape'],
                                     init='glorot_normal',
                                     weights = options['Initial_Filter_Weights'],
                                     subsample = stride)(modelp1)
    modelp3 = k_layers.singlePReLU(weights=options['Initial_PReLU'])(modelp2)
    modelp4 = AveragePooling2D(pool_size=(options['Pool_Size'], options['Pool_Size']))(modelp3)
    modelp5 = Flatten()(modelp4)
    modelp6 = Dense(1,W_regularizer=regularizer,init='glorot_normal',
                            weights = options['Initial_Map_Weights'])(modelp5)
    output = Activation('relu')(modelp6)
    model = keras.models.Model(input=inputLayer,output =output)
    
    return model

def buildkConvSplitNet(options):
    numRows =  options['Input_Shape'][1]
    numCols =  options['Input_Shape'][2]
    assert numRows == numCols
    numFrames = options['Input_Shape'][0]
    
    convImageSize = conv_output_length(numRows,options['Filter_Size'],'valid',options['Stride'])
    downsampImageSize = conv_output_length(convImageSize,options['Pool_Size'],'valid',options['Pool_Size'])
    sum_L1_lambda = options['L1'] /(downsampImageSize**2) 
    sum_L2_lambda = options['L2'] /(downsampImageSize**2) #not used
    regularizer = keras.regularizers.WeightRegularizer(l1=sum_L1_lambda, l2=sum_L2_lambda)
    stride = (options['Stride'],options['Stride'])
    
    inputLayer =keras.layers.Input(shape=options['Input_Shape'])
    

    convLayer = Convolution2D(options['N_Kern'],
                                     options['Filter_Size'],
                                     options['Filter_Size'],
                                     border_mode='valid',
                                     input_shape=options['Input_Shape'],
                                     init='glorot_normal',
                                     weights = options['Initial_Filter_Weights'],
                                     subsample = stride)(inputLayer)
                                     
    posHalf = keras.layers.core.Lambda(lambda x: keras.backend.relu(x))(convLayer)
    posHalfPool = AveragePooling2D(pool_size=(options['Pool_Size'], options['Pool_Size']))(posHalf)
    posHalfPoolFlat = Flatten()(posHalfPool)
    
    negHalf = keras.layers.core.Lambda(lambda x: keras.backend.relu(-x))(convLayer)
    negHalfPool = AveragePooling2D(pool_size=(options['Pool_Size'], options['Pool_Size']))(negHalf)
    negHalfPoolFlat = Flatten()(negHalfPool)
    
    merged = keras.layers.merge([posHalfPoolFlat, negHalfPoolFlat], mode='concat')
    denseLayer = Dense(1,W_regularizer=regularizer,init='glorot_normal',
                            weights = options['Initial_Map_Weights'])(merged)
    output = Activation('relu')(denseLayer)
    model = keras.models.Model(input=inputLayer,output =output)
    return model