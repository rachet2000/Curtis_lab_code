# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 17:12:15 2015

@author: phil
"""

'''CURRENT THINGS IM PLAYING WITH! 1.MIDMODE 2. layer2 sum or concat 3. p_layers, filter layer with 0 initial weights'''

from utils import p_utils
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
import theano
from utils import t_utils
from extra_layers import p_layers
from utils.p_utils import optDef
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

def lassoWithDelay(X_train,y_train,X_test,y_test,options):

    from sklearn import linear_model
    '''IMPORTANT! changed so that lambda is no longer a list, only one value
        this will break other code'''

    lamb  = options['lambdas']

    
    
    bestModel = dict()
    bestModel['Vaf']= 0
    bestModel['Lambda'] = []
    bestModel['NLFit'] = []
    
    
    print(lamb)
    if lamb ==0:
        clf = linear_model.LinearRegression()
    else:
        clf = linear_model.Lasso(alpha=np.float64(lamb))
        
    clf.fit(X_train,y_train)
    y_train_hat = clf.predict(X_train)
    
    try:
        p_opt = p_utils.siaNLFit(y_train,y_train_hat)
    except RuntimeError:
        p_opt = [1.0,1.0]
    
    y_test_hat =  clf.predict(X_test)
    y_test_NL = p_utils.siaNLPredict(y_test_hat,p_opt)        
    
    vaf = p_utils.vaf(y_test,y_test_NL)
    

    bestModel['Vaf']= vaf
    bestModel['Lambda'] = lamb
    bestModel['NLFit'] = p_opt
    bestModel['Weights'] = np.concatenate(((clf.intercept_,),clf.coef_))
    
    return bestModel



def convNet(X_train,y_train,X_valid,y_valid,X_test,y_test,options):
    import theano
    from DeepLearningTutorials.mlp import HiddenLayer
    from deepnn.costs import LSE as LSE
    import theano.tensor as T
#    theano.config.optimizer='None'
    ####################
    #DEFAULT PARAMETERS#
    ####################
    
    batch_size = optDef('Batch_Size',options,500)    
    n_kern = optDef('N_Kern',options,1)
    learning_rate = optDef('Learning_Rate',options,0.01)
    n_epochs = optDef('N_Epochs',options,700)
    #We will only be applying L1/L2 to the mapping layer, not the filter
    L1_lambda = optDef('L1',options,0)
    L2_lambda = optDef('L2',options,0)
    filter_size = optDef('Filter_Size',options,12)
    pool_size = optDef('Pool_Size',options,2)
    midNL = eval(optDef('Mid_Activation_Mode',options,'lambda x: x'))
    endNL = eval(optDef('End_Activation_Mode',options,'lambda x: x'))

    #######################
    #THEANO VARIABLE SETUP#
    #######################
    
    rng = np.random.RandomState(23455)
    
    
    train_set_x = p_utils.load_shared_data(X_train)
    valid_set_x = p_utils.load_shared_data(X_valid)
    test_set_x= p_utils.load_shared_data(X_test)
    
    train_set_y= T.cast(p_utils.load_shared_data(y_train),'float32')
    valid_set_y= T.cast(p_utils.load_shared_data(y_valid),'float32')
    test_set_y= T.cast(p_utils.load_shared_data(y_test),'float32')
    
    
    
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_test_batches /= batch_size
    n_valid_batches /= batch_size
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.tensor3('x')   # the data is presented as rasterized images
    y = T.fvector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    ourFunctions = t_utils.tFunctions(x,y,index,batch_size,n_train_batches,n_valid_batches,n_test_batches)    
    theano_rng = RandomStreams(rng.randint(666))

    
    
    numFrames = np.shape(X_train)[1]
    winLen = int(np.sqrt(np.shape(X_train)[2]))
    
    #################
    #MODEL CREATION #
    #################
    
    print('... building the model')
    
    
    layer0_input = [None] * numFrames
    layer0 = [None] * numFrames
    
    
    #####COMMENTING THIS OUT#####
    for frame in (list(range(numFrames))):
        layer0_input[frame] =  x[:,frame,:].reshape((batch_size, 1, winLen, winLen))
        
        

    
        layer0[frame] =p_layers.p_convPoolLayer(
            rng,
            input=layer0_input[frame],
            image_shape=(batch_size, 1,  winLen, winLen),
            filter_shape=(n_kern, 1, filter_size, filter_size),
            poolsize=(pool_size, pool_size),
            mid_activation = midNL,
            end_activation = endNL

            ,downsamp_mode = 'average'
        )
#        layer2_input = layer2_input + layer0[frame].output.flatten(2)
        
    #if maps for each time lags are seperated    
#    layer2_input = T.concatenate([layer0[frame].output.flatten(2) for frame in range(numFrames)] ,axis=1)
#    mapFrames = numFrames
    #if maps for each time lag are combined
    layer2_input = T.sum([layer0[frame].output.flatten(2) for frame in range(numFrames)] ,axis=0)  
    mapFrames = 1
    #####COMMENTING THIS OUT#####
    
    mapSize = (winLen -filter_size + 1)/ pool_size
    layer2 = p_layers.HiddenLayer(
        rng,
        theano_rng,
        input=layer2_input,
        n_in=n_kern * mapSize* mapSize*mapFrames ,
        n_out = 1,
        activation=lambda x: T.nnet.relu(x, alpha=0)
#        activation=lambda x: T.nnet.softplus(x)
    )
    
    ################################ 
    #MODEL FITTING ALGORITHM SET UP#
    ################################

    params =  layer2.params
    for frame in range(numFrames):
        params = params+ layer0[frame].params
        
    L1 = T.mean(abs(layer2.W))
    L2 = T.mean(layer2.W ** 2)
    cost = LSE(layer2.output,y) + L1*L1_lambda + L2*L2_lambda
     
    grads = T.grad(cost, params)
    
    
    updates = [(param_i, param_i - learning_rate * grad_i)
                for param_i, grad_i in zip(params, grads)]



    train_model = ourFunctions.createUpdateFunc(cost,updates,train_set_x,train_set_y)
    
    validate_model = ourFunctions.createXYFunc(LSE(layer2.output,y),valid_set_x,valid_set_y)
    test_model = ourFunctions.createXYFunc(LSE(layer2.output,y),test_set_x,test_set_y)

    get_train_output = ourFunctions.createXYFunc([layer2.output,y],train_set_x,train_set_y)
    get_validation_output = ourFunctions.createXYFunc([layer2.output,y],valid_set_x,valid_set_y)
    get_test_output = ourFunctions.createXYFunc([layer2.output,y],test_set_x,test_set_y)
#    get_test_grads = [ourFunctions.createXYFunc([grad],test_set_x,test_set_y) for grad in grads ] 
#    get_test_costs = ourFunctions.createXYFunc(cost,train_set_x,train_set_y)


    #########################################
    #CREATE VALIDATION AND TESTING FUNCTIONS#    
    #########################################


    def validation():
        #function that runs every validation, and gives validation score
        validation_losses = [validate_model(i) for i
                     in range(n_valid_batches)]
        this_validation_loss = np.mean(validation_losses)
        return this_validation_loss
    def testVaf():

        resultDict = dict()
        #function that runs every test and gives test score (in this case, prediction vaf)

        train_predictions = [get_train_output(i) for i in range(n_train_batches)]       
        train_predictions=t_utils.unpackTheanoOutput(train_predictions) 
        
        validation_predictions = [get_validation_output(i) for i in range(n_valid_batches)]       
        validation_predictions=t_utils.unpackTheanoOutput(validation_predictions)
        
        test_predictions = [get_test_output(i) for i in range(n_test_batches)]       
        test_predictions=t_utils.unpackTheanoOutput(test_predictions)
        
        noNLValidVaf = p_utils.vaf(np.squeeze(validation_predictions[0]),validation_predictions[1])
        print('valid vaf no NL: ' + str(noNLValidVaf))
        noNLPredVaf = p_utils.vaf(np.squeeze(test_predictions[0]),test_predictions[1])
        print('pred vaf no NL: ' + str(noNLPredVaf))
        
        
        p_opt = p_utils.siaNLFit(train_predictions[1],np.squeeze(train_predictions[0]))

            
        y_valid_NL = p_utils.siaNLPredict(np.squeeze(validation_predictions[0]),p_opt)
        validVaf = p_utils.vaf(y_valid_NL,validation_predictions[1])
        print('valid vaf NL fit: ' + str(validVaf))
        y_test_NL = p_utils.siaNLPredict(np.squeeze(test_predictions[0]),p_opt)
        predVaf = p_utils.vaf(y_test_NL,test_predictions[1])
        print('pred vaf NL fit: ' + str(predVaf))
        
        
        p_optREV = p_utils.siaNLFit(np.squeeze(train_predictions[0]),train_predictions[1])       

        
        y_valid_NLREV = p_utils.siaNLPredict(validation_predictions[1],p_optREV)
        validVafREV = p_utils.vaf(np.squeeze(validation_predictions[0]),y_valid_NLREV)
        print('valid vafREV NL fit: ' + str(validVafREV))
        y_test_NLREV = p_utils.siaNLPredict((test_predictions[1]),p_optREV)
        predVafREV = p_utils.vaf(np.squeeze(test_predictions[0]),y_test_NLREV)
        print('pred vafREV NL fit: ' + str(predVafREV))
        
        
        resultDict['noNLValidVaf'] =noNLValidVaf
        resultDict['noNLPredVaf'] = noNLPredVaf
        resultDict['validVaf'] = validVaf
        resultDict['predVaf'] =predVaf 
        resultDict['p_opt'] = p_opt
        
        return resultDict
        
    ################
    #MODEL TRAINING#
    ################
    print('... training')    
    trainOptions = dict()
    trainOptions['patience'] = 10*(n_train_batches+100)
    trainOptions['patience_increase'] = 1.2   
    trainOptions['improvement_threshold'] = 1
    trainOptions['n_epochs'] = n_epochs
    
    #Training!
    results = ourFunctions.patienceTrain(train_model,validation,testVaf,trainOptions)     
        

    #done training, give results
    print('Optimization complete.')

    print('vaf: ' + str(results['predVaf']))      

    
    bestModel = dict()
    bestModel['validVaf'] = results['validVaf']
    bestModel['predVaf'] = results['predVaf']
    bestModel['noNLValidVaf'] =results['noNLValidVaf']
    bestModel['noNLPredVaf'] = results['noNLPredVaf']
    bestModel['options'] = options.copy()
    bestModel['filterWeights'] = [layer0[xx].W.get_value() for xx in range(len(layer0))]
    bestModel['mapWeights']=layer2.W.get_value()
    bestModel['p_opt'] = results['p_opt']
#    bestModel['filterWeights'] = [np.squeeze(layer0[xx].W.get_value()) for xx in range(len(layer0))]
#    bestModel['mapWeights']=[ np.reshape(xx,(9,9)).T for xx in np.split(np.squeeze(layer2.W.get_value()),16)]
    
    #plot
#    p_utils.plotMapWeights(bestModel['mapWeights'],n_kern,mapFrames)
#    p_utils.plotFilterWeights(bestModel['filterWeights'],n_kern)
    #release theano variables

#    train_set_x.set_value([[]]) 
#    valid_set_x.set_value([[]]) 
#    test_set_x.set_value([[]]) 
#    
#    train_set_y.set_value([[]]) 
#    valid_set_y.set_value([[]]) 
#    test_set_y.set_value([[]]) 
    
    return bestModel
    
def alphaConvNet(X_train,y_train,X_valid,y_valid,X_test,y_test,options):

    from DeepLearningTutorials.mlp import HiddenLayer
    from deepnn.costs import RMSE as RMSE
    from deepnn.costs import LSE as LSE
    from t_utils import RMSLE as RMSLE
    
    import theano.tensor as T
    
    theano.config.optimizer='fast_compile'
    theano.config.exception_verbosity='high'
    theano.config.compute_test_value = 'warn'
    
    ####################
    #DEFAULT PARAMETERS#
    ####################
    
    batch_size = optDef('Batch_Size',options,500)    
    n_kern = optDef('N_Kern',options,1)
    learning_rate = optDef('Learning_Rate',options,0.01)
    n_epochs = optDef('N_Epochs',options,700)
    #We will only be applying L1/L2 to the mapping layer, not the filter
    L1_lambda = optDef('L1',options,0)
    L2_lambda = optDef('L2',options,0)
    filter_size = optDef('Filter_Size',options,12)
    pool_size = optDef('Pool_Size',options,2)
    momentum = optDef('Momentum',options,0.0)
    initial_alpha = optDef('initial_alpha',options,0.5)
    fix_alpha = optDef('fix_alpha',options,1)
    #######################
    #THEANO VARIABLE SETUP#
    #######################
    
    rng = np.random.RandomState(23455)
    
    
    train_set_x = p_utils.load_shared_data(X_train)
    valid_set_x = p_utils.load_shared_data(X_valid)
    test_set_x= p_utils.load_shared_data(X_test)
    
    train_set_y= T.cast(p_utils.load_shared_data(y_train),'float32')
    valid_set_y= T.cast(p_utils.load_shared_data(y_valid),'float32')
    test_set_y= T.cast(p_utils.load_shared_data(y_test),'float32')
    
    
    
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_test_batches /= batch_size
    n_valid_batches /= batch_size
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.tensor3('x')   # the data is presented as rasterized images
    y = T.fvector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    
    x.tag.test_value = X_test.astype('float32')[0:batch_size]
    y.tag.test_value = y_test.astype('float32')[0:batch_size]
    
    ourFunctions = t_utils.tFunctions(x,y,index,batch_size,n_train_batches,n_valid_batches,n_test_batches)    
    theano_rng = RandomStreams(rng.randint(666))

    
    
    numFrames = np.shape(X_train)[1]
    winLen = int(np.sqrt(np.shape(X_train)[2]))
    
    #################
    #MODEL CREATION #
    #################
    
    print('... building the model')
    
    
    layer0_input = [None] * numFrames
    layer0 = [None] * numFrames
    
    alpha = theano.shared(np.asarray(initial_alpha,dtype=theano.config.floatX),borrow=True)
    print('initial alpha: ' + str(alpha.get_value()))
    for frame in (list(range(numFrames))):
        layer0_input[frame] =  x[:,frame,:].reshape((batch_size, 1, winLen, winLen))
        
    
        layer0[frame] =p_layers.p_alphaConvPoolLayer(
            rng,theano_rng,
            input=layer0_input[frame],
            image_shape=(batch_size, 1,  winLen, winLen),
            filter_shape=(n_kern, 1, filter_size, filter_size),
            poolsize=(pool_size, pool_size),
            alpha = alpha,
            
                
            downsamp_mode = 'average'
        )

        
    

    layer0_output_midNL= T.sum([layer0[frame].output_midNL.flatten(2) for frame in range(numFrames)] ,axis=0)  
    layer2_input = layer0_output_midNL
    mapFrames = 1
    

    mapSize = (winLen -filter_size + 1)/ pool_size
    n_in = n_kern * mapSize* mapSize*mapFrames

    layer2 = p_layers.HiddenLayer(
        rng,
        theano_rng,
        input=layer2_input,
        n_in= n_in,
        n_out = 1,

        activation=lambda x: T.nnet.relu(x, alpha=0)

    )
    


    
    
    #NODownSampled Net#
       
    layer0_output_noDS= T.sum([layer0[frame].conv_out_midNL.flatten(2) for frame in range(numFrames)] ,axis=0)  

    layer3_input = layer0_output_noDS
    mapFrames = 1
    

    mapSizeNoDS = (winLen -filter_size + 1)
    n_in_noDS = n_kern * mapSizeNoDS* mapSizeNoDS*mapFrames

    layer3 = p_layers.HiddenLayer(
        rng,
        theano_rng,
        input=layer3_input,
        n_in= n_in_noDS,
        n_out = 1,
#        W=W,
        activation=lambda x: T.nnet.relu(x, alpha=0)
#        activation=lambda x: T.nnet.softplus(x)
    )
    


    
    ################################ 
    #MODEL FITTING ALGORITHM SET UP#
    ################################
        
  
    model_output =  layer2.output
    L1 = T.mean(abs(layer2.W))
    L2 = T.mean(layer2.W ** 2)
    
    
    if fix_alpha:
        print('fixing alpha')
        params =  layer2.params
    else:
        print('alpha not fixed')
        params =  layer2.params +[alpha] 
    
    
    
    for frame in range(numFrames):
        params = params+ layer0[frame].params

    cost = LSE(model_output,y) + L1*(L1_lambda) + L2*(L2_lambda)
        
   
    
    
    if momentum:  
        updates, (global_learning_rate, global_momentum), incs = t_utils.compute_updates_grads(cost,params,learning_rate=learning_rate,masks=None,momentum=momentum)
        print('using momentum')       
    else:  
        grads = T.grad(cost, params)
        updates = [(param_i, param_i - learning_rate * grad_i)
                for param_i, grad_i in zip(params, grads)]
        print('not using momentum')
    
    train_model = ourFunctions.createUpdateFunc(cost,updates,train_set_x,train_set_y)
    
    validate_model = ourFunctions.createXYFunc(LSE(model_output,y),valid_set_x,valid_set_y)

    get_train_output = ourFunctions.createXYFunc([model_output,y],train_set_x,train_set_y)
    get_validation_output = ourFunctions.createXYFunc([model_output,y],valid_set_x,valid_set_y)
    get_test_output = ourFunctions.createXYFunc([model_output,y],test_set_x,test_set_y)
    
#    get_train_output = ourFunctions.createXYFunc([layer2.dot_output,y],train_set_x,train_set_y)
#    get_validation_output = ourFunctions.createXYFunc([layer2.dot_output,y],valid_set_x,valid_set_y)
#    get_test_output = ourFunctions.createXYFunc([layer2.dot_output,y],test_set_x,test_set_y)
#    
#    get_train_dot = ourFunctions.createXYFunc([layer2.dot_output,y],train_set_x,train_set_y)
#    get_train_lin = ourFunctions.createXYFunc([layer2.lin_output,y],train_set_x,train_set_y)
    
    

    
    


    #########################################
    #CREATE VALIDATION AND TESTING FUNCTIONS#    
    #########################################


    def validation():
        #function that runs every validation, and gives validation score
#            for layer in layer0:
#                layer.set_test_mode()
            
        validation_losses = [validate_model(i) for i
                     in range(n_valid_batches)]
        this_validation_loss = np.mean(validation_losses)
        
#            for layer in layer0:
#                layer.set_train_mode()
        return this_validation_loss
    def testVaf():
#            for layer in layer0:
#                layer.set_test_mode()
        resultDict = dict()
        #function that runs every test and gives test score (in this case, prediction vaf)

        train_predictions = [get_train_output(i) for i in range(n_train_batches)]       
        train_predictions=t_utils.unpackTheanoOutput(train_predictions) 
        
        validation_predictions = [get_validation_output(i) for i in range(n_valid_batches)]       
        validation_predictions=t_utils.unpackTheanoOutput(validation_predictions)
        
        test_predictions = [get_test_output(i) for i in range(n_test_batches)]       
        test_predictions=t_utils.unpackTheanoOutput(test_predictions)
        
        noNLValidVaf = p_utils.vaf(np.squeeze(validation_predictions[0]),validation_predictions[1])
        print('valid vaf no NL: ' + str(noNLValidVaf))
        noNLPredVaf = p_utils.vaf(np.squeeze(test_predictions[0]),test_predictions[1])
        print('pred vaf no NL: ' + str(noNLPredVaf))
        
        
        p_opt = p_utils.siaNLFit(train_predictions[1],np.squeeze(train_predictions[0]))

            
        y_valid_NL = p_utils.siaNLPredict(np.squeeze(validation_predictions[0]),p_opt)
        validVaf = p_utils.vaf(y_valid_NL,validation_predictions[1])
        print('valid vaf NL fit: ' + str(validVaf))
        y_test_NL = p_utils.siaNLPredict(np.squeeze(test_predictions[0]),p_opt)
        predVaf = p_utils.vaf(y_test_NL,test_predictions[1])
        print('pred vaf NL fit: ' + str(predVaf))
        
        
        p_optREV = p_utils.siaNLFit(np.squeeze(train_predictions[0]),train_predictions[1])       

        
        y_valid_NLREV = p_utils.siaNLPredict(validation_predictions[1],p_optREV)
        validVafREV = p_utils.vaf(np.squeeze(validation_predictions[0]),y_valid_NLREV)
        print('valid vafREV NL fit: ' + str(validVafREV))
        y_test_NLREV = p_utils.siaNLPredict((test_predictions[1]),p_optREV)
        predVafREV = p_utils.vaf(np.squeeze(test_predictions[0]),y_test_NLREV)
        print('pred vafREV NL fit: ' + str(predVafREV))
        

        
        
        
        for layer in layer0:
            layer.save_bestParams()
        layer2.save_bestParams()
        layer3.save_bestParams()
        print('alpha: ' + str(alpha.get_value()))
        
        resultDict['noNLValidVaf'] =noNLValidVaf
        resultDict['noNLPredVaf'] = noNLPredVaf
        resultDict['validVaf'] = validVaf
        resultDict['predVaf'] =predVaf 
        resultDict['validVafREV'] = validVafREV
        resultDict['predVafREV'] =predVafREV 
        resultDict['p_opt'] = p_opt
        resultDict['alpha'] = alpha.get_value()
        resultDict['test_predictions'] = test_predictions
        
#            for layer in layer0:
#                layer.set_train_mode()
        return resultDict
            
    ################
    #MODEL TRAINING#
    ################
    print('... training')    
    trainOptions = dict()
    trainOptions['patience'] = 12*(n_train_batches+100)
    trainOptions['patience_increase'] = 1.2  
    trainOptions['improvement_threshold'] = 1
    trainOptions['n_epochs'] = n_epochs
    


    resultDict = ourFunctions.patienceTrain(train_model,validation,testVaf,trainOptions)
    
    
    ################
    #MODEL Clean up#
    ################
    for layer in layer0:
        layer.load_bestParams()
        
    layer2.load_bestParams()
    layer3.load_bestParams()
    print('first iteration alpha: ' + str(alpha.get_value()))
    alpha.set_value(resultDict['alpha'] )
    

    #done training, give results
    print('Optimization complete.')
#    aa = layer2.W.get_value()
#    bb = p_utils.upsample(aa,4,kern_size = 27,downsamp=2)
    print('noNLValidVaf '+ str(resultDict['noNLValidVaf'] ))
    print('noNLPredVaf ' + str(resultDict['noNLPredVaf'] ))
    print('validVaf ' + str(resultDict['validVaf']))
    print('predVaf ' + str(resultDict['predVaf'] ))
    print('p_opt ' + str(resultDict['p_opt']))
    print(alpha.get_value())
    
    print('Printing hyperparameters ...')
    print('batch_size: ' +str(batch_size))
    print('n_kern: ' +str(n_kern))
    print('learning_rate: ' + str(learning_rate)) 
    print('n_epochs: ' +str(n_epochs))
    print('L1_lambda: ' +str(L1_lambda)) 
    print('L2_lambda: ' +str(L2_lambda)) 
    print('filter_size: ' +str(filter_size)) 
    print('pool_size: ' +str(pool_size))
    print('momentum: ' +str(momentum)) 
    
    bestModel = dict()
    bestModel['validVaf'] = resultDict['validVaf']
    bestModel['predVaf'] = resultDict['predVaf']
    bestModel['noNLValidVaf'] =resultDict['noNLValidVaf']
    bestModel['noNLPredVaf'] = resultDict['noNLPredVaf']
    bestModel['alpha'] = resultDict['alpha']
    bestModel['validVafREV']= resultDict['validVafREV'] 
    bestModel['predVafREV']= resultDict['predVafREV'] 
    bestModel['options'] = options.copy()
    bestModel['filterWeights'] = [layer0[xx].W.get_value() for xx in range(len(layer0))]
    bestModel['mapWeights']=layer2.W.get_value()
    bestModel['p_opt'] = resultDict['p_opt']
#    bestModel['filterWeights'] = [np.squeeze(layer0[xx].W.get_value()) for xx in range(len(layer0))]
#    bestModel['mapWeights']=[ np.reshape(xx,(9,9)).T for xx in np.split(np.squeeze(layer2.W.get_value()),16)]
    
    #plot
#    p_utils.plotMapWeights(bestModel['mapWeights'],n_kern,mapFrames)
#    p_utils.plotFilterWeights(bestModel['filterWeights'],n_kern)
    #release theano variables

#    train_set_x.set_value([[]]) 
#    valid_set_x.set_value([[]]) 
#    test_set_x.set_value([[]]) 
#    
#    train_set_y.set_value([[]]) 
#    valid_set_y.set_value([[]]) 
#    test_set_y.set_value([[]]) 
    
    return bestModel
        
def alphaNet(X_train,y_train,X_valid,y_valid,X_test,y_test,options):

    from deepnn.costs import LSE as LSE
#    import theano.tensor as T
#    from DeepLearningTutorials.mlp import HiddenLayer
    ####################
    #DEFAULT PARAMETERS#
    ####################
    
    batch_size = optDef('Batch_Size',options,7500)    
    n_kern = optDef('N_Kern',options,1)
    learning_rate = optDef('Learning_Rate',options,0.001)
    n_epochs = optDef('N_Epochs',options,7000)



    #######################
    #THEANO VARIABLE SETUP#
    #######################
    
    rng = np.random.RandomState(666)
    
    
    train_set_x = p_utils.load_shared_data(X_train)
    valid_set_x = p_utils.load_shared_data(X_valid)
    test_set_x= p_utils.load_shared_data(X_test)
    
    train_set_y= T.cast(p_utils.load_shared_data(y_train),'float32')
    valid_set_y= T.cast(p_utils.load_shared_data(y_valid),'float32')
    test_set_y= T.cast(p_utils.load_shared_data(y_test),'float32')
    
    
    
    n_train_examples = train_set_x.get_value(borrow=True).shape[0]
    n_valid_examples = valid_set_x.get_value(borrow=True).shape[0]
    n_test_examples = test_set_x.get_value(borrow=True).shape[0]
    
    
    n_train_batches = 1
    n_valid_batches = 1
    n_test_batches = 1

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.fvector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    theano_rng = RandomStreams(rng.randint(666))
    ourFunctions = t_utils.tFunctions(x,y,index,batch_size,n_train_batches,n_valid_batches,n_test_batches)    

    
    


    stimLen = int(np.shape(X_train)[1])
    #################
    #MODEL CREATION #
    #################
    
    print('... building the model')
    
#    W_values = np.asarray(np.zeros((stimLen,1)),dtype=theano.config.floatX)
#    W_values = np.asarray(rng.normal(0,.0000001,(stimLen,1)),dtype=theano.config.floatX)
    W_values = np.asarray(rng.uniform(
    low=-np.sqrt(6e-15 / (stimLen + 1)),
    high=np.sqrt(6e-15 / (stimLen + 1)),
    size=(stimLen, 1)), dtype=theano.config.floatX)
    W = theano.shared(value=W_values, name='W', borrow=True)

    alpha = theano.shared(np.asarray(0.0,dtype=theano.config.floatX),borrow=True)

    layer0 =p_layers.HiddenLayer(
        rng,
        theano_rng,
        input=x,
        n_in = stimLen,
        n_out = 1,
        W=W,
        activation = lambda x: T.nnet.relu(x,alpha = alpha)

    )
    finalOutput = layer0.output
    
    ################################ 
    #MODEL FITTING ALGORITHM SET UP#
    ################################

    params =  layer0.params + [alpha]

        
        

    cost = LSE(finalOutput,y) 
     
    grads = T.grad(cost, params)
    
    
    updates = [(param_i, param_i - learning_rate * grad_i)
                for param_i, grad_i in zip(params, grads)]



    train_model = ourFunctions.createUpdateFunc(cost,updates,train_set_x,train_set_y)
    
    validate_model = ourFunctions.createXYFuncWithBatch(LSE(finalOutput,y),valid_set_x,valid_set_y,n_valid_examples)
    test_model = ourFunctions.createXYFuncWithBatch(LSE(finalOutput,y),test_set_x,test_set_y,n_test_examples)

    get_train_output = ourFunctions.createXYFunc([finalOutput,y],train_set_x,train_set_y)
    get_validation_output = ourFunctions.createXYFuncWithBatch([finalOutput,y],valid_set_x,valid_set_y,n_valid_examples)
    get_test_output = ourFunctions.createXYFuncWithBatch([finalOutput,y],test_set_x,test_set_y,n_test_examples)

    

    #########################################
    #CREATE VALIDATION AND TESTING FUNCTIONS#    
    #########################################


    def validation():
        #function that runs every validation, and gives validation score
        validation_losses = [validate_model(i) for i
                     in range(n_valid_batches)]
        this_validation_loss = np.mean(validation_losses)
        return this_validation_loss
    def testVaf():

        resultDict = dict()
        #function that runs every test and gives test score (in this case, prediction vaf)

        train_predictions = [get_train_output(i) for i in range(n_train_batches)]       
        train_predictions=t_utils.unpackTheanoOutput(train_predictions) 
        
        validation_predictions = [get_validation_output(i) for i in range(n_valid_batches)]       
        validation_predictions=t_utils.unpackTheanoOutput(validation_predictions)
        
        test_predictions = [get_test_output(i) for i in range(n_test_batches)]       
        test_predictions=t_utils.unpackTheanoOutput(test_predictions)
        
        noNLValidVaf = p_utils.vaf(np.squeeze(validation_predictions[0]),validation_predictions[1])
        print('valid vaf no NL: ' + str(noNLValidVaf))
        noNLPredVaf = p_utils.vaf(np.squeeze(test_predictions[0]),test_predictions[1])
        print('pred vaf no NL: ' + str(noNLPredVaf))
        
        
        p_opt = p_utils.siaNLFit(train_predictions[1],np.squeeze(train_predictions[0]))

            
        y_valid_NL = p_utils.siaNLPredict(np.squeeze(validation_predictions[0]),p_opt)
        validVaf = p_utils.vaf(y_valid_NL,validation_predictions[1])
        print('valid vaf NL fit: ' + str(validVaf))
        y_test_NL = p_utils.siaNLPredict(np.squeeze(test_predictions[0]),p_opt)
        predVaf = p_utils.vaf(y_test_NL,test_predictions[1])
        print('pred vaf NL fit: ' + str(predVaf))
        
        
        p_optREV = p_utils.siaNLFit(np.squeeze(train_predictions[0]),train_predictions[1])       

        
        y_valid_NLREV = p_utils.siaNLPredict(validation_predictions[1],p_optREV)
        validVafREV = p_utils.vaf(np.squeeze(validation_predictions[0]),y_valid_NLREV)
        print('valid vafREV NL fit: ' + str(validVafREV))
        y_test_NLREV = p_utils.siaNLPredict((test_predictions[1]),p_optREV)
        predVafREV = p_utils.vaf(np.squeeze(test_predictions[0]),y_test_NLREV)
        print('pred vafREV NL fit: ' + str(predVafREV))
        
        
        resultDict['noNLValidVaf'] =noNLValidVaf
        resultDict['noNLPredVaf'] = noNLPredVaf
        resultDict['validVaf'] = validVaf
        resultDict['predVaf'] =predVaf 
        resultDict['p_opt'] = p_opt
        
        return resultDict
        
    ################
    #MODEL TRAINING#
    ################
    print('... training')    
    trainOptions = dict()
    trainOptions['patience'] =3*(n_train_batches+100)
    trainOptions['patience_increase'] = 1.2   
    trainOptions['improvement_threshold'] = 1
    trainOptions['n_epochs'] = n_epochs
    
    #Training!
    resultDict = ourFunctions.patienceTrain(train_model,validation,testVaf,trainOptions)     
#    predVaf,p_opt = ourFunctions.patienceTrain(train_NL,validation,testVaf,trainOptions)     

    #done training, give results
    print('Optimization complete.')

        
    print('noNLValidVaf '+ str(resultDict['noNLValidVaf'] ))
    print('noNLPredVaf ' + str(resultDict['noNLPredVaf'] ))
    print('validVaf ' + str(resultDict['validVaf']))
    print('predVaf ' + str(resultDict['predVaf'] ))
    print('p_opt ' + str(resultDict['p_opt']))
    
#    bestModel = dict()
#    bestModel['validVaf'] = validVaf
#    bestModel['predVaf'] = predVaf
#    bestModel['options'] = options
#    bestModel['filterWeights'] = [layer0[xx].W.get_value() for xx in range(len(layer0))]
#    bestModel['mapWeights']=layer2.W.get_value()
#    bestModel['NLFit'] = p_opt
#    bestModel['filterWeights'] = [np.squeeze(layer0[xx].W.get_value()) for xx in range(len(layer0))]
#    bestModel['mapWeights']=[ np.reshape(xx,(9,9)).T for xx in np.split(np.squeeze(layer2.W.get_value()),16)]
    
    #plot
    print(alpha.get_value())
    p_utils.plotMapWeights(bestModel['mapWeights'],n_kern,mapFrames)
    p_utils.plotFilterWeights(bestModel['filterWeights'],n_kern)
    #release theano variables

    train_set_x.set_value([[]]) 
    valid_set_x.set_value([[]]) 
    test_set_x.set_value([[]]) 
    
    train_set_y.set_value([[]]) 
    valid_set_y.set_value([[]]) 
    test_set_y.set_value([[]]) 
    
    return bestModel   
    
def staRegression(X_train,y_train,X_valid,y_valid,X_test,y_test,options):

    from deepnn.costs import LSE as LSE
#    import theano.tensor as T
#    from DeepLearningTutorials.mlp import HiddenLayer
    ####################
    #DEFAULT PARAMETERS#
    ####################
    
    batch_size = optDef('Batch_Size',options,7500)    
    n_kern = optDef('N_Kern',options,1)
    learning_rate = optDef('Learning_Rate',options,0.001)
    n_epochs = optDef('N_Epochs',options,7000)



    #######################
    #THEANO VARIABLE SETUP#
    #######################
    
    rng = np.random.RandomState(666)
    
    
    train_set_x = p_utils.load_shared_data(X_train)
    valid_set_x = p_utils.load_shared_data(X_valid)
    test_set_x= p_utils.load_shared_data(X_test)
    
    train_set_y= T.cast(p_utils.load_shared_data(y_train),'float32')
    valid_set_y= T.cast(p_utils.load_shared_data(y_valid),'float32')
    test_set_y= T.cast(p_utils.load_shared_data(y_test),'float32')
    
    
    
    n_train_examples = train_set_x.get_value(borrow=True).shape[0]
    n_valid_examples = valid_set_x.get_value(borrow=True).shape[0]
    n_test_examples = test_set_x.get_value(borrow=True).shape[0]
    
    
    n_train_batches = 1
    n_valid_batches = 1
    n_test_batches = 1

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.fvector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    theano_rng = RandomStreams(rng.randint(666))
    ourFunctions = t_utils.tFunctions(x,y,index,batch_size,n_train_batches,n_valid_batches,n_test_batches)    

    
    


    stimLen = int(np.shape(X_train)[1])
    #################
    #MODEL CREATION #
    #################
    
    print('... building the model')
    
#    W_values = np.asarray(np.zeros((stimLen,1)),dtype=theano.config.floatX)
#    W_values = np.asarray(rng.normal(0,.0000001,(stimLen,1)),dtype=theano.config.floatX)
    W_values = np.asarray(rng.uniform(
    low=-np.sqrt(6e-15 / (stimLen + 1)),
    high=np.sqrt(6e-15 / (stimLen + 1)),
    size=(stimLen, 1)), dtype=theano.config.floatX)
    W = theano.shared(value=W_values, name='W', borrow=True)

    layer0 =p_layers.HiddenLayer(
        rng,
        theano_rng,
        input=x,
        n_in = stimLen,
        n_out = 1,
        W=W,
        activation = lambda x: T.nnet.relu(x)

    )
    finalOutput = layer0.output
    
    ################################ 
    #MODEL FITTING ALGORITHM SET UP#
    ################################

    params =  layer0.params

        
        

    cost = LSE(finalOutput,y) 
     
    grads = T.grad(cost, params)
    
    
    updates = [(param_i, param_i - learning_rate * grad_i)
                for param_i, grad_i in zip(params, grads)]



    train_model = ourFunctions.createUpdateFunc(cost,updates,train_set_x,train_set_y)
    
    validate_model = ourFunctions.createXYFuncWithBatch(LSE(finalOutput,y),valid_set_x,valid_set_y,n_valid_examples)
    test_model = ourFunctions.createXYFuncWithBatch(LSE(finalOutput,y),test_set_x,test_set_y,n_test_examples)

    get_train_output = ourFunctions.createXYFunc([finalOutput,y],train_set_x,train_set_y)
    get_validation_output = ourFunctions.createXYFuncWithBatch([finalOutput,y],valid_set_x,valid_set_y,n_valid_examples)
    get_test_output = ourFunctions.createXYFuncWithBatch([finalOutput,y],test_set_x,test_set_y,n_test_examples)

    

    #########################################
    #CREATE VALIDATION AND TESTING FUNCTIONS#    
    #########################################


    def validation():
        #function that runs every validation, and gives validation score
        validation_losses = [validate_model(i) for i
                     in range(n_valid_batches)]
        this_validation_loss = np.mean(validation_losses)
        return this_validation_loss
    def testVaf():

        resultDict = dict()
        #function that runs every test and gives test score (in this case, prediction vaf)

        train_predictions = [get_train_output(i) for i in range(n_train_batches)]       
        train_predictions=t_utils.unpackTheanoOutput(train_predictions) 
        
        validation_predictions = [get_validation_output(i) for i in range(n_valid_batches)]       
        validation_predictions=t_utils.unpackTheanoOutput(validation_predictions)
        
        test_predictions = [get_test_output(i) for i in range(n_test_batches)]       
        test_predictions=t_utils.unpackTheanoOutput(test_predictions)
        
        noNLValidVaf = p_utils.vaf(np.squeeze(validation_predictions[0]),validation_predictions[1])
        print('valid vaf no NL: ' + str(noNLValidVaf))
        noNLPredVaf = p_utils.vaf(np.squeeze(test_predictions[0]),test_predictions[1])
        print('pred vaf no NL: ' + str(noNLPredVaf))
        
        
        p_opt = p_utils.siaNLFit(train_predictions[1],np.squeeze(train_predictions[0]))

            
        y_valid_NL = p_utils.siaNLPredict(np.squeeze(validation_predictions[0]),p_opt)
        validVaf = p_utils.vaf(y_valid_NL,validation_predictions[1])
        print('valid vaf NL fit: ' + str(validVaf))
        y_test_NL = p_utils.siaNLPredict(np.squeeze(test_predictions[0]),p_opt)
        predVaf = p_utils.vaf(y_test_NL,test_predictions[1])
        print('pred vaf NL fit: ' + str(predVaf))
        
        
        p_optREV = p_utils.siaNLFit(np.squeeze(train_predictions[0]),train_predictions[1])       

        
        y_valid_NLREV = p_utils.siaNLPredict(validation_predictions[1],p_optREV)
        validVafREV = p_utils.vaf(np.squeeze(validation_predictions[0]),y_valid_NLREV)
        print('valid vafREV NL fit: ' + str(validVafREV))
        y_test_NLREV = p_utils.siaNLPredict((test_predictions[1]),p_optREV)
        predVafREV = p_utils.vaf(np.squeeze(test_predictions[0]),y_test_NLREV)
        print('pred vafREV NL fit: ' + str(predVafREV))
        
        
        resultDict['noNLValidVaf'] =noNLValidVaf
        resultDict['noNLPredVaf'] = noNLPredVaf
        resultDict['validVaf'] = validVaf
        resultDict['predVaf'] =predVaf 
        resultDict['p_opt'] = p_opt
        
        return resultDict
        
    ################
    #MODEL TRAINING#
    ################
    print('... training')    
    trainOptions = dict()
    trainOptions['patience'] =3*(n_train_batches+100)
    trainOptions['patience_increase'] = 1.2   
    trainOptions['improvement_threshold'] = 1
    trainOptions['n_epochs'] = n_epochs
    
    #Training!
    resultDict = ourFunctions.patienceTrain(train_model,validation,testVaf,trainOptions)     
#    predVaf,p_opt = ourFunctions.patienceTrain(train_NL,validation,testVaf,trainOptions)     

    #done training, give results
    print('Optimization complete.')

        
    print('noNLValidVaf '+ str(resultDict['noNLValidVaf'] ))
    print('noNLPredVaf ' + str(resultDict['noNLPredVaf'] ))
    print('validVaf ' + str(resultDict['validVaf']))
    print('predVaf ' + str(resultDict['predVaf'] ))
    print('p_opt ' + str(resultDict['p_opt']))
    
#    bestModel = dict()
#    bestModel['validVaf'] = validVaf
#    bestModel['predVaf'] = predVaf
#    bestModel['options'] = options
#    bestModel['filterWeights'] = [layer0[xx].W.get_value() for xx in range(len(layer0))]
#    bestModel['mapWeights']=layer2.W.get_value()
#    bestModel['NLFit'] = p_opt
#    bestModel['filterWeights'] = [np.squeeze(layer0[xx].W.get_value()) for xx in range(len(layer0))]
#    bestModel['mapWeights']=[ np.reshape(xx,(9,9)).T for xx in np.split(np.squeeze(layer2.W.get_value()),16)]
    
    #plot
    p_utils.plotMapWeights(bestModel['mapWeights'],n_kern,mapFrames)
    p_utils.plotFilterWeights(bestModel['filterWeights'],n_kern)
    #release theano variables

    train_set_x.set_value([[]]) 
    valid_set_x.set_value([[]]) 
    test_set_x.set_value([[]]) 
    
    train_set_y.set_value([[]]) 
    valid_set_y.set_value([[]]) 
    test_set_y.set_value([[]]) 
    
    return bestModel


#    
#    
#def convNetKeras(X_train,y_train,X_valid,y_valid,X_test,y_test,options):
#    
#    
#    
#    return
def PPAConvNet(X_train,y_train,X_valid,y_valid,X_test,y_test,options):

    from DeepLearningTutorials.mlp import HiddenLayer
    from deepnn.costs import RMSE as RMSE
    from deepnn.costs import LSE as LSE
    from .t_utils import RMSLE as RMSLE
    
    import theano.tensor as T
    
    theano.config.optimizer='fast_compile'
    theano.config.exception_verbosity='high'
    theano.config.compute_test_value = 'warn'
    
    ####################
    #DEFAULT PARAMETERS#
    ####################
    
    batch_size = optDef('Batch_Size',options,500)    
    n_kern = optDef('N_Kern',options,1)
    learning_rate = optDef('Learning_Rate',options,0.01)
    n_epochs = optDef('N_Epochs',options,700)
    #We will only be applying L1/L2 to the mapping layer, not the filter
    L1_lambda = optDef('L1',options,0)
    L2_lambda = optDef('L2',options,0)
    filter_size = optDef('Filter_Size',options,12)
    pool_size = optDef('Pool_Size',options,2)
    momentum = optDef('Momentum',options,0.0)
    intial_alpha = optDef('initial_alpha',options,0.5)
    #######################
    #THEANO VARIABLE SETUP#
    #######################
    
    rng = np.random.RandomState(23455)
    
    
    train_set_x = p_utils.load_shared_data(X_train)
    valid_set_x = p_utils.load_shared_data(X_valid)
    test_set_x= p_utils.load_shared_data(X_test)
    
    train_set_y= T.cast(p_utils.load_shared_data(y_train),'float32')
    valid_set_y= T.cast(p_utils.load_shared_data(y_valid),'float32')
    test_set_y= T.cast(p_utils.load_shared_data(y_test),'float32')
    
    
    
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_test_batches /= batch_size
    n_valid_batches /= batch_size
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.tensor3('x')   # the data is presented as rasterized images
    y = T.fvector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    
    x.tag.test_value = X_test.astype('float32')[0:batch_size]
    y.tag.test_value = y_test.astype('float32')[0:batch_size]
    
    ourFunctions = t_utils.tFunctions(x,y,index,batch_size,n_train_batches,n_valid_batches,n_test_batches)    
    theano_rng = RandomStreams(rng.randint(666))

    
    
    numFrames = np.shape(X_train)[1]
    winLen = int(np.sqrt(np.shape(X_train)[2]))
    
    #################
    #MODEL CREATION #
    #################
    
    print('... building the model')
    
    
    layer0_input = [None] * numFrames
    layer0 = [None] * numFrames
    
    
#    push_pull = theano.shared(np.asarray(1.0,dtype=theano.config.floatX),borrow=True)
    push = theano.shared(np.asarray(1.0,dtype=theano.config.floatX),borrow=True)
    pull = theano.shared(np.asarray(1.0,dtype=theano.config.floatX),borrow=True)

    alpha = theano.shared(np.asarray(initial_alpha,dtype=theano.config.floatX),borrow=True)

    print('initial alpha: ' + str(alpha.get_value()))
    for frame in (list(range(numFrames))):
        initial_frame = x[:,frame,:]
        pushPullFrame = T.switch(initial_frame > 0,push* initial_frame, pull * initial_frame) 
#        pushPullFrame = T.nnet.relu(initial_frame,alpha = push_pull)
        layer0_input[frame] =  pushPullFrame.reshape((batch_size, 1, winLen, winLen))
        
    
        layer0[frame] =p_layers.p_alphaConvPoolLayer(
            rng,theano_rng,
            input=layer0_input[frame],
            image_shape=(batch_size, 1,  winLen, winLen),
            filter_shape=(n_kern, 1, filter_size, filter_size),
            poolsize=(pool_size, pool_size),
            alpha = alpha,
            
                
            downsamp_mode = 'average'
        )

        
    
    #DownSampled Net#
      

#    layer0_output_endNL = T.sum([layer0[frame].output_endNL.flatten(2) for frame in range(numFrames)] ,axis=0)  
    layer0_output_midNL= T.sum([layer0[frame].output_midNL.flatten(2) for frame in range(numFrames)] ,axis=0)  
#    layer2_input = layer0_output_endNL
    layer2_input = layer0_output_midNL
    mapFrames = 1
    

    mapSize = (winLen -filter_size + 1)/ pool_size
    n_in = n_kern * mapSize* mapSize*mapFrames

    layer2 = p_layers.HiddenLayer(
        rng,
        theano_rng,
        input=layer2_input,
        n_in= n_in,
        n_out = 1,
#        W=W,
        activation=lambda x: T.nnet.relu(x, alpha=0)
#        activation=lambda x: T.nnet.softplus(x)
    )
    


    
    
    #NODownSampled Net#
       
    layer0_output_noDS= T.sum([layer0[frame].conv_out_midNL.flatten(2) for frame in range(numFrames)] ,axis=0)  

    layer3_input = layer0_output_noDS
    mapFrames = 1
    

    mapSizeNoDS = (winLen -filter_size + 1)
    n_in_noDS = n_kern * mapSizeNoDS* mapSizeNoDS*mapFrames

    layer3 = p_layers.HiddenLayer(
        rng,
        theano_rng,
        input=layer3_input,
        n_in= n_in_noDS,
        n_out = 1,
#        W=W,
        activation=lambda x: T.nnet.relu(x, alpha=0)
#        activation=lambda x: T.nnet.softplus(x)
    )
    


    
    ################################ 
    #MODEL FITTING ALGORITHM SET UP#
    ################################
        
    for phase in range(1):
#    for phase in range(2):
        if phase == 0:
            model_output =  layer2.output
            L1 = T.mean(abs(layer2.W))
            L2 = T.mean(layer2.W ** 2)
            params =  layer2.params +[alpha] +[push]+[pull]
            for frame in range(numFrames):
                params = params+ layer0[frame].params

            cost = LSE(model_output,y) + L1*(L1_lambda) + L2*(L2_lambda)
            
        elif phase ==1:
            layer3.b.set_value(layer2.b.get_value())
            layer3.best_b.set_value(layer2.b.get_value())

            upSampledW = p_utils.upsample(layer2.W.get_value(),filter_size,kern_size = mapSizeNoDS,downsamp=pool_size)
            upSampledW = upSampledW/np.square(pool_size)
            upSampledW = upSampledW.astype(theano.config.floatX)
            layer3.W.set_value(upSampledW)
            layer3.best_W.set_value(upSampledW)
            
            model_output =  layer3.output
            L1 = T.mean(abs(layer3.W))
            L2 = T.mean(layer3.W ** 2)
#            L1 = 0
#            L2 =0
            params =  layer3.params +[alpha] +[push]+[pull]
            cost = LSE(model_output,y) #+ L1*(L1_lambda) + L2*(L2_lambda)
#            learning_rate = learning_rate/10
        

    
            
#        cost = LSE(model_output,y) + L1*(L1_lambda) + L2*(L2_lambda)
#    #    cost = RMSE(model_output,y) + L1*L1_lambda + L2*L2_lambda
#    #    cost = RMSLE(model_output,y) + L1*L1_lambda + L2*L2_lambda
#        
        useMomentum = 0
        if useMomentum:  
            updates, (global_learning_rate, global_momentum), incs = t_utils.compute_updates_grads(cost,params,learning_rate=learning_rate,masks=None,momentum=momentum)
            print('using momentum')
        else:  
            grads = T.grad(cost, params)
            updates = [(param_i, param_i - learning_rate * grad_i)
                    for param_i, grad_i in zip(params, grads)]
            print('not using momentum')
        
        train_model = ourFunctions.createUpdateFunc(cost,updates,train_set_x,train_set_y)
        
        validate_model = ourFunctions.createXYFunc(LSE(model_output,y),valid_set_x,valid_set_y)
    
        get_train_output = ourFunctions.createXYFunc([model_output,y],train_set_x,train_set_y)
        get_validation_output = ourFunctions.createXYFunc([model_output,y],valid_set_x,valid_set_y)
        get_test_output = ourFunctions.createXYFunc([model_output,y],test_set_x,test_set_y)
        
    #    get_train_output = ourFunctions.createXYFunc([layer2.dot_output,y],train_set_x,train_set_y)
    #    get_validation_output = ourFunctions.createXYFunc([layer2.dot_output,y],valid_set_x,valid_set_y)
    #    get_test_output = ourFunctions.createXYFunc([layer2.dot_output,y],test_set_x,test_set_y)
    #    
    #    get_train_dot = ourFunctions.createXYFunc([layer2.dot_output,y],train_set_x,train_set_y)
    #    get_train_lin = ourFunctions.createXYFunc([layer2.lin_output,y],train_set_x,train_set_y)
        
        
    
        
        
    
    
        #########################################
        #CREATE VALIDATION AND TESTING FUNCTIONS#    
        #########################################
    
    
        def validation():
            #function that runs every validation, and gives validation score
#            for layer in layer0:
#                layer.set_test_mode()
                
            validation_losses = [validate_model(i) for i
                         in range(n_valid_batches)]
            this_validation_loss = np.mean(validation_losses)
            
#            for layer in layer0:
#                layer.set_train_mode()
            return this_validation_loss
        def testVaf():
#            for layer in layer0:
#                layer.set_test_mode()
            resultDict = dict()
            #function that runs every test and gives test score (in this case, prediction vaf)
    
            train_predictions = [get_train_output(i) for i in range(n_train_batches)]       
            train_predictions=t_utils.unpackTheanoOutput(train_predictions) 
            
            validation_predictions = [get_validation_output(i) for i in range(n_valid_batches)]       
            validation_predictions=t_utils.unpackTheanoOutput(validation_predictions)
            
            test_predictions = [get_test_output(i) for i in range(n_test_batches)]       
            test_predictions=t_utils.unpackTheanoOutput(test_predictions)
            
            noNLValidVaf = p_utils.vaf(np.squeeze(validation_predictions[0]),validation_predictions[1])
            print('valid vaf no NL: ' + str(noNLValidVaf))
            noNLPredVaf = p_utils.vaf(np.squeeze(test_predictions[0]),test_predictions[1])
            print('pred vaf no NL: ' + str(noNLPredVaf))
            
            
            p_opt = p_utils.siaNLFit(train_predictions[1],np.squeeze(train_predictions[0]))
    
                
            y_valid_NL = p_utils.siaNLPredict(np.squeeze(validation_predictions[0]),p_opt)
            validVaf = p_utils.vaf(y_valid_NL,validation_predictions[1])
            print('valid vaf NL fit: ' + str(validVaf))
            y_test_NL = p_utils.siaNLPredict(np.squeeze(test_predictions[0]),p_opt)
            predVaf = p_utils.vaf(y_test_NL,test_predictions[1])
            print('pred vaf NL fit: ' + str(predVaf))
            
            
            p_optREV = p_utils.siaNLFit(np.squeeze(train_predictions[0]),train_predictions[1])       
    
            
            y_valid_NLREV = p_utils.siaNLPredict(validation_predictions[1],p_optREV)
            validVafREV = p_utils.vaf(np.squeeze(validation_predictions[0]),y_valid_NLREV)
            print('valid vafREV NL fit: ' + str(validVafREV))
            y_test_NLREV = p_utils.siaNLPredict((test_predictions[1]),p_optREV)
            predVafREV = p_utils.vaf(np.squeeze(test_predictions[0]),y_test_NLREV)
            print('pred vafREV NL fit: ' + str(predVafREV))
            
    
            
            
            
            for layer in layer0:
                layer.save_bestParams()
            layer2.save_bestParams()
            layer3.save_bestParams()
            print('alpha: ' + str(alpha.get_value()))
            
            resultDict['noNLValidVaf'] =noNLValidVaf
            resultDict['noNLPredVaf'] = noNLPredVaf
            resultDict['validVaf'] = validVaf
            resultDict['predVaf'] =predVaf 
            resultDict['validVafREV'] = validVafREV
            resultDict['predVafREV'] =predVafREV 
            resultDict['p_opt'] = p_opt
            resultDict['alpha'] = alpha.get_value()
            resultDict['test_predictions'] = test_predictions
            resultDict['push'] = push.get_value()
            resultDict['pull'] = pull.get_value()
#            for layer in layer0:
#                layer.set_train_mode()
            return resultDict
                
        ################
        #MODEL TRAINING#
        ################
        print('... training')    
        trainOptions = dict()
        trainOptions['patience'] = 12*(n_train_batches+100)
        trainOptions['patience_increase'] = 1.4  
        trainOptions['improvement_threshold'] = 1
        trainOptions['n_epochs'] = n_epochs
        


        resultDict = ourFunctions.patienceTrain(train_model,validation,testVaf,trainOptions)
        
        
        ################
        #MODEL Clean up#
        ################
        for layer in layer0:
            layer.load_bestParams()
            
        layer2.load_bestParams()
        layer3.load_bestParams()
        print('first iteration alpha: ' + str(alpha.get_value()))
        alpha.set_value(resultDict['alpha'] )
        push.set_value(resultDict['push'] )
        pull.set_value(resultDict['pull'] )

    #done training, give results
    print('Optimization complete.')
#    aa = layer2.W.get_value()
#    bb = p_utils.upsample(aa,4,kern_size = 27,downsamp=2)
    print('noNLValidVaf '+ str(resultDict['noNLValidVaf'] ))
    print('noNLPredVaf ' + str(resultDict['noNLPredVaf'] ))
    print('validVaf ' + str(resultDict['validVaf']))
    print('predVaf ' + str(resultDict['predVaf'] ))
    print('p_opt ' + str(resultDict['p_opt']))
    print(alpha.get_value())
    print(push.get_value())
    print(pull.get_value())
    
    print('Printing hyperparameters ...')
    print('batch_size: ' +str(batch_size))
    print('n_kern: ' +str(n_kern))
    print('learning_rate: ' + str(learning_rate)) 
    print('n_epochs: ' +str(n_epochs))
    print('L1_lambda: ' +str(L1_lambda)) 
    print('L2_lambda: ' +str(L2_lambda)) 
    print('filter_size: ' +str(filter_size)) 
    print('pool_size: ' +str(pool_size))
    print('momentum: ' +str(momentum)) 
    
    bestModel = dict()
    bestModel['validVaf'] = resultDict['validVaf']
    bestModel['predVaf'] = resultDict['predVaf']
    bestModel['noNLValidVaf'] =resultDict['noNLValidVaf']
    bestModel['noNLPredVaf'] = resultDict['noNLPredVaf']
    bestModel['alpha'] = resultDict['alpha']
    bestModel['validVafREV']= resultDict['validVafREV'] 
    bestModel['predVafREV']= resultDict['predVafREV'] 
    bestModel['options'] = options.copy()
    bestModel['filterWeights'] = [layer0[xx].W.get_value() for xx in range(len(layer0))]
    bestModel['mapWeights']=layer2.W.get_value()
    bestModel['p_opt'] = resultDict['p_opt']
    bestModel['push'] = resultDict['push']
    bestModel['pull'] = resultDict['pull']
#    bestModel['filterWeights'] = [np.squeeze(layer0[xx].W.get_value()) for xx in range(len(layer0))]
#    bestModel['mapWeights']=[ np.reshape(xx,(9,9)).T for xx in np.split(np.squeeze(layer2.W.get_value()),16)]
    
    #plot
#    p_utils.plotMapWeights(bestModel['mapWeights'],n_kern,mapFrames)
#    p_utils.plotFilterWeights(bestModel['filterWeights'],n_kern)
    #release theano variables

#    train_set_x.set_value([[]]) 
#    valid_set_x.set_value([[]]) 
#    test_set_x.set_value([[]]) 
#    
#    train_set_y.set_value([[]]) 
#    valid_set_y.set_value([[]]) 
#    test_set_y.set_value([[]]) 
    
    return bestModel
    
    
