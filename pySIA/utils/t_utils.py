# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 17:33:32 2015

@author: phil
"""

import numpy as np
import theano
import theano.tensor as T
from utils.p_utils import optDef

def unpackTheanoOutput(output):
    #shape is hardcoded, would not recommend using this function for too many things
    # assumes rows corresponds to different instances
    sepList = list(zip(*output))
    fullList = []
    for aList in sepList:
        newShape = list(aList[0].shape)
        newShape[0] = newShape[0]*len(aList)
        newList = np.zeros((newShape))
        start = 0
        for batch in aList:
            newList[start:start+batch.shape[0]] = batch
            start = start + batch.shape[0]
        fullList.append(newList)
    return fullList
def RMSLE(x,y):
    #Root Mean Squared Logarithmic Error
    return T.sqrt(T.mean((T.log(x.T +1)-T.log(y +1))**2))            
class tFunctions(object):
    #spits out dictionary containing
    def __init__(self,x,y,index,batch_size,n_train_batches,n_valid_batches,n_test_batches):
        self.x = x
        self.y = y
        self.index = index
        self.batch_size = batch_size
        self.n_train_batches = n_train_batches
        self.n_valid_batches = n_valid_batches
        self.n_test_batches = n_test_batches
         
    
    def createXFunc(self,output,dataSet_X):
        theanoFunction = theano.function(
            [self.index],
            output,
            givens={
                self.x: dataSet_X[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            }
        )
        return theanoFunction
        
  
    def createXYFunc(self,output,dataSet_X,dataSet_y):
        theanoFunction = theano.function(
            [self.index],
            output,
            givens={
                self.x: dataSet_X[self.index * self.batch_size: (self.index + 1) * self.batch_size],
                self.y: dataSet_y[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            }
        )
        return theanoFunction
    def createXYFuncWithBatch(self,output,dataSet_X,dataSet_y,batch_size):
        theanoFunction = theano.function(
            [self.index],
            output,
            givens={
                self.x: dataSet_X[self.index * batch_size: (self.index + 1) * batch_size],
                self.y: dataSet_y[self.index * batch_size: (self.index + 1) * batch_size]
            }
        )
        return theanoFunction
    def createUpdateFunc(self,output,updates,dataSet_X,dataSet_y):
        theanoFunction = theano.function(
            [self.index],
            output,
            updates=updates,
            givens={
                self.x: dataSet_X[self.index * self.batch_size: (self.index + 1) * self.batch_size],
                self.y: dataSet_y[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            }
        )
        return theanoFunction
        
    def patienceTrain(self,training_function,validation_function,test_function,options):
        #training function will train the model on every iteration
        #validation function  returns a validation score (needs to be a number)
        #testing function returns a testing score (can be anything, the best validated score will be returned)
        patience = optDef('patience',options,3000)    
        patience_increase = optDef('patience_increase',options,1.2) 
        improvement_threshold = optDef('improvement_threshold',options,1) 
        n_epochs = optDef('n_epochs',options,500) 


        validation_frequency = min(self.n_train_batches, patience / 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch
    
        best_validation_loss = np.inf
        best_iter = 0
        test_score = 0.

    
        epoch = 0
        done_looping = False
        
        resultDict = dict()
        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in range(self.n_train_batches):
    
                iter = (epoch - 1) * self.n_train_batches + minibatch_index
    
                if iter % 100 == 0:
                    print('training @ iter = ', iter)
                cost_ij = training_function(minibatch_index)
                
                if (iter + 1) % validation_frequency == 0:
    
                    # compute zero-one loss on validation set
                    
                    this_validation_loss = validation_function()
                    print(('epoch %i, minibatch %i/%i, validation LSE %f ' %
                          (epoch, minibatch_index + 1, self.n_train_batches,
                           this_validation_loss)))
    
                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
    
                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                            patience = max(patience, iter * patience_increase)
    
                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter
                        
                        this_test_score = test_function()

                if patience <= iter:
                    done_looping = True
                    break
            
            print('patience: ' + str(patience))
                
                
                
                
                
                
        return this_test_score
        
def compute_updates_grads(cost,params,learning_rate=.1,masks=None,momentum=.95):
		

		updates = []
		incs = []

		gparams = T.grad(cost, params)

		global_learning_rate = theano.shared(theano._asarray(learning_rate, dtype=theano.config.floatX),name='learning rate',borrow=True)
		global_momentum = theano.shared(theano._asarray(momentum, dtype=theano.config.floatX),name='learning rate',borrow=True)

		for param in params:			
			inc = theano.shared(value=np.array(param.get_value() * 0,dtype='float32'),borrow=True)
			inc.name = 'inc_'+ str(param.name)
			inc.constrainable = False
			incs.append(inc)
		
		for param, gparam, inc in zip(params, gparams,incs):			

			update = inc * global_momentum - global_learning_rate * gparam
			updates.append((inc,  update))

			param.constrainable = True

			updates.append((param, param + update))


		return updates, (global_learning_rate, global_momentum), incs
#def vaf    
