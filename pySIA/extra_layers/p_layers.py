# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 16:02:23 2015

@author: phil
"""
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv,conv3d2d
from theano.sandbox.cuda import dnn
from theano.sandbox.neighbours import images2neibs
class p_convPoolLayer(object):
    """built off of leNetConvPoolLayer, see convolutional_mpl.py """
    
    def __init__(self, rng, input, filter_shape, image_shape, 
                 poolsize=(2, 2),downsamp_mode='average',
                 mid_activation = lambda x: x,
                 end_activation = lambda x: x,
#                 end_activation = lambda x: T.nnet.relu(x, alpha=0),
#                 end_activation = lambda x: T.nnet.softplus(x),
#                 end_activation = lambda x: T.tanh(x),
                 ):
        
        
        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(poolsize))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        
#        initialFilter = rng.uniform(low=-W_bound, high=W_bound, size=filter_shape)
#        initialFilter = rng.normal(loc=0, scale=W_bound, size=filter_shape)
        initialFilter = rng.uniform(low=-0, high=0, size=filter_shape)
#        initialFilter[:,:,filter_shape[2]/2,filter_shape[3]/2] = W_bound
        
        self.W = theano.shared(np.asarray(initialFilter,dtype=theano.config.floatX),borrow=True)


        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
        
        self.best_b = theano.shared(value=self.b.get_value(),name='best_b',borrow=True)
        self.best_W = theano.shared(value=self.W.get_value(),name='best_W',borrow=True)


        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )
        conv_out = mid_activation(conv_out)
        # downsample each feature map individually, using maxpooling
        if downsamp_mode == 'average':
#            raise Exception('average pooling not implemented yet')
            pooled_out = pool.max_pool_2d(
                input=conv_out,
                ds=poolsize,
                ignore_border=True,
                mode = 'average_inc_pad'
            )

        else:
            #default to max pooling
            pooled_out = pool.max_pool_2d(
                input=conv_out,
                ds=poolsize,
                ignore_border=True,
                mode = 'max'
            )
#        pooled_out = dnn.dnn_pool(
#            img=conv_out,
#            ws=poolsize,
#            mode = downsamp_mode
#        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        lin_output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        self.output = end_activation(lin_output)

        # store parameters of this layer
        self.params = [self.W, self.b]
        self.bestParams = [self.best_W, self.best_b]
        # keep track of model input
        self.input = input
        
        def load_bestParams(self):
            for param,bestParam in zip(self.params,self.bestParams):
              param.set_value(bestParam.get_value())
              
        def save_bestParams(self):
            for param,bestParam in zip(self.params,self.bestParams):
              bestParam.set_value(param.get_value())
        
class p_alphaConvPoolLayer(object):
    """built off of leNetConvPoolLayer, see convolutional_mpl.py """
    
    def __init__(self, rng,theano_rng, input, filter_shape, image_shape, 
                 poolsize=(2, 2),downsamp_mode='average',
                alpha = None,W = None,b = None,#alpha_noise =None


                 ):
        #both mid and end activation will be relu, with learnable alphas
        
        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(poolsize))
                   
                   
                   
        if W is None:
        # initialize weights with random weights
            W_bound = np.sqrt(6e-1 / (fan_in + fan_out))
            
            initialFilter = rng.uniform(low=-W_bound, high=W_bound, size=filter_shape)
    #        initialFilter = rng.normal(loc=0, scale=W_bound, size=filter_shape)
    #        initialFilter = rng.uniform(low=-0, high=0, size=filter_shape)
#            initialFilter[:,:,filter_shape[2]/2,filter_shape[3]/2] = W_bound
            
            W = theano.shared(np.asarray(initialFilter,dtype=theano.config.floatX),borrow=True)
    
        if b is None:
                # the bias is a 1D tensor -- one bias per output feature map
            b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, borrow=True)
        
        if alpha is None:
            alpha = theano.shared(np.asarray(0.0,dtype=theano.config.floatX),borrow=True)
        
#        if alpha_noise is None:
#            alpha_noise = theano.shared(np.asarray(0.0,dtype=theano.config.floatX),borrow=True)
        
        self.W = W
        self.b = b   
        self.alpha = alpha
        
#        self.alpha_noise = alpha_noise
#        self.init_alpha_noise = self.alpha_noise.get_value()
        
        self.best_b = theano.shared(value=self.b.get_value(),name='best_b',borrow=True)
        self.best_W = theano.shared(value=self.W.get_value(),name='best_W',borrow=True)
#        self.best_alpha =  theano.shared(value=self.alpha.get_value(),name='best_alpha',borrow=True)
#        unitNoiseGenerator = theano_rng.normal(size=(1,),avg=0.0,std=1.0)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )
        
        
        #We now split the output into two paths, with the nonlinearity before/after downsampling
        
        self.mid_activation_endNL = lambda x : x
#        self.mid_activation_midNL = lambda x: T.nnet.relu(x,alpha = self.alpha+(self.alpha_noise *unitNoiseGenerator))
        self.mid_activation_midNL = lambda x: T.nnet.relu(x,alpha = self.alpha)

#        conv_out_endNL = self.mid_activation_endNL(conv_out)
#        conv_out_midNL = self.mid_activation_midNL(conv_out)
#        
        self.conv_out_endNL = self.mid_activation_endNL(conv_out+ self.b.dimshuffle('x', 0, 'x', 'x'))
        self.conv_out_midNL = self.mid_activation_midNL(conv_out+ self.b.dimshuffle('x', 0, 'x', 'x'))
        # downsample each feature map individually, using maxpooling
        
        
        
        
        if downsamp_mode == 'average':
#            raise Exception('average pooling not implemented yet')
            pooled_out_endNL = pool.max_pool_2d(
                input=self.conv_out_endNL,
                ds=poolsize,
                ignore_border=True,
                mode = 'average_inc_pad'
            )
            pooled_out_midNL = pool.max_pool_2d(
                input=self.conv_out_midNL,
                ds=poolsize,
                ignore_border=True,
                mode = 'average_inc_pad'
            )

        else:
            #default to max pooling
            pooled_out_endNL = pool.max_pool_2d(
                input=self.conv_out_endNL,
                ds=poolsize,
                ignore_border=True,
                mode = 'max'
            )
            pooled_out_midNL = pool.max_pool_2d(
                input=self.conv_out_midNL,
                ds=poolsize,
                ignore_border=True,
                mode = 'max'
            )
#        pooled_out_endNL = conv_out_endNL
#        pooled_out_midNL = conv_out_midNL
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
#        lin_output_endNL = pooled_out_endNL + self.b.dimshuffle('x', 0, 'x', 'x')
#        lin_output_midNL = pooled_out_midNL + self.b.dimshuffle('x', 0, 'x', 'x')
        lin_output_endNL = pooled_out_endNL 
        lin_output_midNL = pooled_out_midNL 

        self.end_activation_endNL = lambda x: T.nnet.relu(x,alpha = self.alpha)
        self.end_activation_midNL = lambda x: x

        self.output_endNL = self.end_activation_endNL(lin_output_endNL)
        self.output_midNL = self.end_activation_midNL(lin_output_midNL)

        # store parameters of this layer
#        self.params = [self.W, self.b,self.alpha]
#        self.bestParams = [self.best_W, self.best_b,self.best_alpha]
        self.params = [self.W, self.b]
        self.bestParams = [self.best_W, self.best_b]
#        self.noise_src = [self.alpha_noise]
#        self.init_noise_vals = [self.init_alpha_noise]
        # keep track of model input
        self.input = input
        
    def load_bestParams(self):
        for param,bestParam in zip(self.params,self.bestParams):
          param.set_value(bestParam.get_value())
          
    def save_bestParams(self):
        for param,bestParam in zip(self.params,self.bestParams):
          bestParam.set_value(param.get_value())
#    def swap_activations(self):
#        temp_activation = self.end_activation
#        self.end_activation = self.mid_activation
#        self.mid_activation = temp_activation

#    def set_test_mode(self):
#        for noise_source in self.noise_src:
#            noise_source.set_value(0.0)
#    def set_train_mode(self):
#        for noise_source,init_val in zip(self.noise_src,self.init_noise_vals):
#            noise_source.set_value(init_val)
        
class p_convPoolLayer3d(object):
    """built off of leNetConvPoolLayer, see convolutional_mpl.py """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2),downsamp_mode='max',activation = lambda x: T.tanh(x)):
        '''downsamp mode not implemented, should be implemented with cuda.dnn'''

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(poolsize))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv3d2d.conv3d(
            signals=input,
            filters=self.W,
            filters_shape=filter_shape,
            signals_shape =image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = pool.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        lin_output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        self.output = activation(lin_output)

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input
        
class p_tentLayer(object):
    def __init__(self,input,numTents,inputRange,fitIntercept = True):
        self.input = input
        #scale tents such that each point on the number line is covered by only 2 tents
        self.scale = float(inputRange)/float(numTents)
        #sets up the centers for every tent function
        space = np.linspace(-inputRange,inputRange,num=2*numTents+1)
        self.space = space
        
        #create model parameters
        W_values = np.asarray(space,dtype=theano.config.floatX)
        self.W = theano.shared(W_values,borrow=True)
        
        b_values = np.zeros(1, dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
        
        
        #best_params
        self.best_b = theano.shared(value=self.b.get_value(),name='best_b',borrow=True)
        self.best_W = theano.shared(value=self.W.get_value(),name='best_W',borrow=True)

        
        
        
        #convert the tensor3 input into a vector
        self.originalShape = T.shape(input)
        self.vectorInput = T.flatten(input)  
        
        #theano version of creating the tent basis
        self.extendedMatrix=T.stack([T.maximum(0,1 - T.abs_((self.vectorInput - spot)/self.scale)) for spot in space],axis=1)
        
        #output is linear in the tent basis space
        self.lin_output = T.dot(self.extendedMatrix,self.W)
        self.vectorOutput = self.lin_output + self.b.dimshuffle('x', 0, 'x', 'x')
        self.output = T.reshape(self.vectorOutput,self.originalShape)        

        if fitIntercept:
            self.params = [self.W,self.b]
            self.bestParams =[self.best_W,self.best_b]
        else:
            self.params = [self.W]
            self.bestParams =[self.best_W]
    def load_bestParams(self):
        for param,bestParam in zip(self.params,self.bestParams):
            param.set_value(bestParam.get_value())
              
    def save_bestParams(self):
        for param,bestParam in zip(self.params,self.bestParams):
            bestParam.set_value(param.get_value())
            
            
            
class HiddenLayer(object):
	def __init__(self, rng, theano_rng, input, n_in, n_out, W=None, b=None,dropout=0,activation=T.tanh):

		self.original_dropout = dropout

		if W is None:
                  W_values = np.asarray(rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
#                  W_values = np.asarray(rng.uniform(
#                    low=-np.sqrt(6e-4 / (n_in + n_out)),
#                    high=np.sqrt(6e-4 / (n_in + n_out)),
#                    size=(n_in, n_out)), dtype=theano.config.floatX)
#                  W_values = np.asarray(np.ones(
#                    shape=(n_in, n_out)), dtype=theano.config.floatX)
                  W = theano.shared(value=W_values, name='W', borrow=True)

		if b is None:
                  b_values = np.zeros((n_out,), dtype=theano.config.floatX)
                  b = theano.shared(value=b_values, name='b', borrow=True)

		self.W = W
		self.b = b
  
          #best_params
		self.best_b = theano.shared(value=b.get_value(),name='best_b',borrow=True)
		self.best_W = theano.shared(value=W.get_value(),name='best_W',borrow=True)

		self.rng = rng
		self.theano_rng = theano_rng
		self.n_in = n_in
		self.n_out = n_out		
		self.activation = activation
		self.input = input
		self.dropout = 	dropout	

		self.p_drop = theano.shared(theano._asarray(dropout, dtype=theano.config.floatX),borrow=True)
		self.activation_scale = theano.shared(theano._asarray(1, dtype=theano.config.floatX),borrow=True)		

		noise = theano_rng.normal(size=(self.b.shape), avg=0.0, std=.01)
		self.noise_scale = theano.shared(theano._asarray(1, dtype=theano.config.floatX),borrow=True)	

	        
		self.lin_output = T.dot(input, self.W)   + self.b			
		
		self.output = activation(self.lin_output)
		
		mask  = self.theano_rng.binomial(size = (self.output.shape) ,p=self.p_drop,dtype='float32')
		if self.dropout > 0:
			self.output = self.output * mask * self.activation_scale

		# parameters of the model		
		self.params = [self.W, self.b]
		self.bestParams = [self.best_W, self.best_b]
	def set_dropout(self,scale):
		self.p_drop.set_value(value=scale)

	def set_activation_scale(self,scale):
		self.activation_scale.set_value(value=scale)

	def set_test_mode(self):

		self.set_dropout(1)
		self.noise_scale.set_value(0)

		activation_scale = 1 if self.original_dropout == 0 else self.original_dropout		
		self.set_activation_scale(activation_scale)

	def set_train_mode(self):
		self.set_dropout(self.original_dropout)				
		self.set_activation_scale(1)
		self.noise_scale.set_value(1)

	def set_noise_stddev(self,noise):
		self.noise_stddev.set_value(value=noise+1e-6)

	def refresh(self):
		W_values = np.asarray(self.rng.normal(0,.01,(self.n_in,self.n_out)),dtype=theano.config.floatX)

		self.W.set_value(value=W_values)

		b_values = np.zeros((self.n_out,), dtype=theano.config.floatX)

		self.b.set_value(value=b_values)
	def load_bestParams(self):
          for param,bestParam in zip(self.params,self.bestParams):
              param.set_value(bestParam.get_value())
              
	def save_bestParams(self):
          for param,bestParam in zip(self.params,self.bestParams):
              bestParam.set_value(param.get_value())
          


			           
     
        
def _dropout_from_layer(rng, layer, p):
    """p is the probablity of dropping a unit
    """
    srng = theano.tensor.shared_randomstreams.RandomStreams(
            rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output

class DropoutLayer(HiddenLayer):
    def __init__(self, rng, input, dropout_rate):

        self.input = input
        self.maskOutput = _dropout_from_layer(rng, self.input, p=dropout_rate)
        self.output = input*dropout_rate #scale by the dropout rate

        
        
            
class OrderDropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, theano_rng, input, n_in, n_out,trueInput,noisyInput, W=None, b=None,dropout=0,activation=T.tanh,ordrop=0):
        HiddenLayer.__init__(self, rng, theano_rng, input, n_in, n_out, W=None, b=None,dropout=0,activation=T.tanh)
        

        lin_trueOutput = T.dot(trueInput, self.W) + self.b	
        self.trueOutput = activation(lin_trueOutput)
        

        
        
        self.original_ordrop = ordrop
        self.ordrop = ordrop
        self.unitSweep =theano.shared(theano._asarray(0, dtype=theano.config.floatX),borrow = True)

        
        self.probs = [self.ordrop*((1-self.ordrop)**i) for i in range(self.n_out)]
        self.geopvals = theano.shared(theano._asarray(self.probs, dtype=theano.config.floatX),borrow=True)
        if self.ordrop > 0:
            #set up geometric mask
            
            geomArray = self.theano_rng.multinomial(size= self.output.mean(axis=1).shape,n=1,pvals=self.geopvals,dtype='float32')
            geomValue =T.outer((geomArray.T.nonzero()[0]),T.ones(self.n_out))
            fill = T.ones(self.output.mean(axis=1).shape)
            indexArray = theano.shared(theano._asarray(list(range(self.n_out)),dtype = theano.config.floatX))
            indices = T.outer(indexArray,fill).T
            self.geomMask = T.ge(geomValue+self.unitSweep,indices)
            self.output = self.output * self.geomMask 
        

    