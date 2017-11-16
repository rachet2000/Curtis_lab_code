# -*- coding: utf-8 -*-
"""
Created on Tue May 24 17:39:39 2016

@author: phil

special layers for keras
"""

from keras import initializers, regularizers
from keras.engine import Layer
from keras import backend as K
import numpy as np

if K._backend == 'theano':
    import theano
    import theano.tensor as T
    theano.config.floatX='float32'
    matrix_inverse = T.nlinalg.matrix_inverse
    tensordot = T.tensordot
    matrix_determinant = T.nlinalg.det
elif K._backend == 'tensorflow':
    import tensorflow as tf
    matrix_inverse = tf.matrix_inverse
    tensordot = tf.tensordot
    matrix_determinant = tf.matrix_determinant


# class singlePReLU(Layer):
#     # PReLU modified to have a single alpha for a whole layer
#     # since i'm deviating from the Keras backend, this may cause errors
#     # If errors occur, try replacing this singlePReLU with the Keras PReLU
#     # and see if errors still occur
#     # alpha is constrained to be < 1
#
#     '''Parametric Rectified Linear Unit:
#     `f(x) = alphas * x for x < 0`,
#     `f(x) = x for x >= 0`,
#     where `alphas` is a learned array with the same shape as x.
#
#     # Input shape
#         Arbitrary. Use the keyword argument `input_shape`
#         (tuple of integers, does not include the samples axis)
#         when using this layer as the first layer in a model.
#
#     # Output shape
#         Same shape as the input.
#
#     # Arguments
#         init: initialization function for the weights.
#         weights: initial weights, as a list of a single numpy array.
#
#     # References
#         - [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](http://arxiv.org/pdf/1502.01852v1.pdf)
#     '''
#     def __init__(self, init='zero', weights=None, **kwargs):
#         self.supports_masking = True
#         self.init = initializers.get(init)
#         self.initial_weights = weights
#         super(singlePReLU, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         self.alpha = self.init((),
#                                 name='{}_alphas'.format(self.name))
#         self.trainable_weights = [self.alpha]
#
#         if self.initial_weights is not None:
#             K.set_value(self.alpha, self.initial_weights)
#             del self.initial_weights
#
#     def call(self, x, mask=None):
#         self.alpha = (K.cast(self.alpha >= 1., K.floatx())) + self.alpha*(K.cast(self.alpha < 1., K.floatx()))
#
#         pos = K.relu(x)
#         neg = self.alpha * (x - abs(x)) * 0.5
#         return pos + neg
#
#     def get_config(self):
#         config = {'init': self.init.__name__}
#         base_config = super(singlePReLU, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))


class DOG2dMapLayer(Layer):
    ''' assumes 2-d  input
        2d difference of gaussians layer
    '''
    def __init__(self,input_dim,init = 'zero', init_pos_mean= None,init_pos_sigma = None,
                 init_neg_mean= None,init_neg_sigma = None,
                 init_scale = None,**kwargs):
                 
        self.input_dim = input_dim
        
        xSpace = np.linspace(0,input_dim[0]-1,input_dim[0])
        ySpace = np.linspace(0,input_dim[1]-1,input_dim[1])
        spaceMatrix = np.asarray((np.meshgrid(xSpace,ySpace)))
        self.spaceVector = spaceMatrix.reshape((2,input_dim[0]*input_dim[1]))
        
        self.init = initializers.get(init)
        
        if init_pos_mean is None:
            init_pos_mean = np.asarray([input_dim[0]/2.,input_dim[1]/2])
          
        if init_pos_sigma is None:
            init_pos_sigma =np.asarray([np.asarray(input_dim[0]),np.asarray(0.0),np.asarray(input_dim[1])])
            
        if init_neg_mean is None:
            init_neg_mean = np.asarray([input_dim[0]/2.,input_dim[1]/2])          
        if init_neg_sigma is None:
            init_neg_sigma =np.asarray([np.asarray(input_dim[0]),np.asarray(0.0),np.asarray(input_dim[1])])
        if init_scale is None:
            init_scale =0.75

        self.init_pos_mean = init_pos_mean.astype('float32')
        self.init_pos_sigma = init_pos_sigma.astype('float32')
        self.init_neg_mean = init_neg_mean.astype('float32')
        self.init_neg_sigma = init_neg_sigma.astype('float32')
        self.init_scale = np.asarray(init_scale).astype('float32')

        
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim)
        super(DOG2dMapLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.pos_mean = self.init((2,),
                                name='{}_posmean'.format(self.name))
        self.pos_sigma = self.init((3,),
                              name='{}_possigma'.format(self.name))

                              
        self.neg_mean = self.init((2,),
                                name='{}_negmean'.format(self.name))
        self.neg_sigma = self.init((3,),
                              name='{}_negsigma'.format(self.name))
                              
        self.scale = self.init((),
                                name='{}_scale'.format(self.name))                      

        self.trainable_weights = [self.pos_mean,  self.pos_sigma, self.neg_mean, self.neg_sigma, self.scale]

        if self.init_pos_mean is not None:
            K.set_value(self.pos_mean, self.init_pos_mean)
            del self.init_pos_mean
        if self.init_pos_sigma is not None:
            K.set_value(self.pos_sigma, self.init_pos_sigma)
            del self.init_pos_sigma

        
        if self.init_neg_mean is not None:
            K.set_value(self.neg_mean,self.init_neg_mean)
            del self.init_neg_mean
        if self.init_neg_sigma is not None:
            K.set_value(self.neg_sigma,self.init_neg_sigma)
            del self.init_neg_sigma
            
        K.set_value(self.scale,self.init_scale)
        del self.init_scale

    def calc_kernel(self,x,mean,sigma):
        tolerance = np.asarray(0.01) #use tolerance from stopping the matrix from being un-invertible
        T.set_subtensor(sigma[1] ,T.sgn(sigma[1])*T.min((T.sqrt(sigma[0]*sigma[2])-tolerance,T.abs_(sigma[1]))))

        inner = (self.spaceVector - mean.dimshuffle(0,'x'))

        cov = T.stack([[sigma[0],sigma[1]],[sigma[1],sigma[2]]])
        inverseCov = T.nlinalg.matrix_inverse(cov)
        firstProd =  T.tensordot(inner.T,inverseCov,axes=1)
        malahDistance = T.sum(firstProd*inner.T,axis =1)
        gaussianDistance = T.exp((-1./2.)*malahDistance)
        detCov = T.nlinalg.det(cov)
        denom = 1./(2*np.pi*T.sqrt(detCov))
        gdKernel = T.dot(x,denom*gaussianDistance)
        
        return gdKernel
        
    def call(self,x, mask=None):
        x = K.reshape(x, (K.shape(x)[0], K.shape(x)[-2]*K.shape(x)[-1]))
        
        
        posKernel = self.calc_kernel(x,self.pos_mean,self.pos_sigma)
        negKernel = self.calc_kernel(x,self.neg_mean,self.neg_sigma)
        DOGKernel = self.scale*posKernel - (1.0-self.scale)*negKernel
        
        
        return DOGKernel.dimshuffle(0,'x').astype('float32')
        


    def get_output_shape_for(self, input_shape):
        return (input_shape[0],1)    
        
    def get_config(self):
        config = {'init': self.init.__name__}
        base_config = super(DOG2dMapLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class gaussian2dMapLayer(Layer):
    """
        assumes 2-d  input
    """
    def __init__(self, input_dim, init='zero',
                 init_mean=None,
                 init_sigma=None,
                 init_sigma_div=1.0,
                 sigma_regularizer_l2=None,
                 **kwargs):
        assert input_dim[0] == input_dim[1], "Input must be square"
        self.input_dim = input_dim
        self.inv_scale = input_dim[0]
        xSpace = np.linspace(0, input_dim[0]-1, input_dim[0])
        ySpace = np.linspace(0, input_dim[1]-1, input_dim[1])
        spaceMatrix = np.asarray((np.meshgrid(xSpace,ySpace)))
        self.spaceVector = spaceMatrix.reshape((2,input_dim[0]*input_dim[1]))
        
        self.init = initializers.get(init)
        
        if init_mean is None:
            init_mean = np.asarray([input_dim[0]/2.,input_dim[1]/2])
          
        if init_sigma is None:
            init_sigma =np.asarray([np.asarray(input_dim[0])/init_sigma_div,
                                         np.asarray(0.0)/init_sigma_div,
                                         np.asarray(input_dim[1])/init_sigma_div])
        
        self.init_mean = init_mean.astype('float32')
        self.init_sigma = init_sigma.astype('float32')

        self.sigma_regularizer_l2 = regularizers.get(sigma_regularizer_l2)
        self.tolerance = np.asarray(0.01)  # use tolerance from stopping the matrix from being un-invertible
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim)
        super(gaussian2dMapLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.mean = self.add_weight((2,),
                                    initializer=self.init,
                                    name='{}_mean'.format(self.name))

        self.sigma = self.add_weight((3,),
                                     initializer=self.init,
                                     name='{}_sigma'.format(self.name))


        self.trainable_weights = [self.mean,  self.sigma]

        if self.init_mean is not None:
            K.set_value(self.mean, self.init_mean)
            del self.init_mean
        if self.init_sigma is not None:
            K.set_value(self.sigma, self.init_sigma)
            del self.init_sigma
            
        self.regularizers = []
        
        if self.sigma_regularizer_l2:
            #self.sigma_regularizer_l2.set_param(self.sigma)
            self.sigma_regularizer_l2(self.sigma)
            self.regularizers.append(self.sigma_regularizer_l2)
        self.built = True


    def call(self,x, mask=None):
        x = K.reshape(x, (K.shape(x)[0],K.shape(x)[-2]*K.shape(x)[-1]))
        #
        # covar = K.sign(self.sigma[1]) * K.switch(
        #     K.sqrt(self.sigma[0] * self.sigma[2]) - self.tolerance > K.abs(self.sigma[1]),
        #     K.abs(self.sigma[1]),
        #     K.sqrt(self.sigma[0] * self.sigma[2]) - self.tolerance)
        #
        # # Below is just the calculations for a Gaussian
        # inner = (self.spaceVector - self.inv_scale * K.expand_dims(self.mean))
        #
        # cov = self.inv_scale * K.stack([[self.sigma[0], covar], [covar, self.sigma[2]]])
        # inverseCov = matrix_inverse(cov)
        # firstProd = tensordot(K.transpose(inner), inverseCov, axes=1)
        # malahDistance = K.sum(firstProd * K.transpose(inner), axis=1)
        # gaussianDistance = K.exp((-1. / 2.) * malahDistance)
        # detCov = matrix_determinant(cov)
        # denom = 1. / (2 * np.pi * K.sqrt(detCov))
        # #        gdKernel = K.dot(x,denom*gaussianDistance)
        # gdKernel = tensordot(x, denom * gaussianDistance, axes=1)
        # return K.expand_dims(gdKernel)


        # tolerance = np.asarray(0.01) #use tolerance from stopping the matrix from being un-invertible
        # T.set_subtensor(self.sigma[1], T.sgn(self.sigma[1])*T.min((T.sqrt(self.sigma[0]*self.sigma[2])-tolerance, T.abs_(self.sigma[1]))))
        # inner = (self.spaceVector - self.mean.dimshuffle(0,'x'))
        # cov = T.stack([[self.sigma[0], self.sigma[1]], [self.sigma[1], self.sigma[2]]])
        # inverseCov = T.nlinalg.matrix_inverse(cov)
        # firstProd = T.tensordot(inner.T, inverseCov, axes=1)
        # malahDistance = T.sum(firstProd*inner.T, axis=1)
        # gaussianDistance = T.exp((-1./2.)*malahDistance)
        # detCov = T.nlinalg.det(cov)
        # denom = 1./(2*np.pi*T.sqrt(detCov))
        # gdKernel = T.dot(x, denom*gaussianDistance)
        # return gdKernel.dimshuffle(0, 'x').astype('float32')


        covar = K.sign(self.sigma[1]) * K.switch(
            K.sqrt(self.sigma[0] * self.sigma[2]) - self.tolerance > K.abs(self.sigma[1]),
            K.abs(self.sigma[1]),
            K.sqrt(self.sigma[0] * self.sigma[2]) - self.tolerance)
        inner = (self.spaceVector - K.expand_dims(self.mean))
        cov = K.stack([[self.sigma[0], covar], [covar, self.sigma[2]]])
        inverseCov = matrix_inverse(cov)
        firstProd = tensordot(K.transpose(inner), inverseCov, axes=1)
        malahDistance = K.sum(firstProd * K.transpose(inner), axis=1)
        gaussianDistance = K.exp((-1. / 2.) * malahDistance)
        detCov = matrix_determinant(cov)
        denom = 1. / (2 * np.pi * K.sqrt(detCov))
        gdKernel = tensordot(x, denom * gaussianDistance, axes=1)
        return K.expand_dims(gdKernel)


    def get_output_shape_for(self, input_shape):
        return (input_shape[0],1)
    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)
    def get_config(self):
        #config = {'init': self.init.__name__}
        base_config = super(gaussian2dMapLayer, self).get_config()
        #return dict(list(base_config.items()) + list(config.items()))
        return dict(list(base_config.items()))


class gaussian2dMapLayerNormalized(Layer):
    ''' assumes 2-d  input
        Only Allows Square 
    '''
    def __init__(self,input_dim,init = 'zero',
                                 init_mean= None,
                                 init_sigma = None,
                                 scale = 1.0,
                                 init_sigma_div =1.0,
                                 sigma_regularizer_l2 = None,
                                 
                                 **kwargs):
        assert input_dim[0] == input_dim[1],"Input must be square"

        self.input_dim = input_dim
        self.scale = np.float(scale)
        self.inv_scale = self.input_dim[0]/self.scale
        
        xSpace = np.linspace(0,input_dim[0]-1,input_dim[0])
        ySpace = np.linspace(0,input_dim[1]-1,input_dim[1])
        spaceMatrix = np.asarray((np.meshgrid(xSpace,ySpace)))
        self.spaceVector = spaceMatrix.reshape((2,input_dim[0]*input_dim[1]))
        
        self.init = initializers.get(init)
        
        if init_mean is None:
            half_mean = (1/2.)*self.scale
            init_mean = np.asarray([half_mean,half_mean])
          
        if init_sigma is None:
            one_sig = (np.asarray(1.0)/init_sigma_div)*self.scale
            init_sigma =np.asarray([one_sig,
                                         np.asarray(0.0),
                                         one_sig])

        self.init_mean = init_mean.astype('float32')
        self.init_sigma = init_sigma.astype('float32')

        self.sigma_regularizer_l2 = regularizers.get(sigma_regularizer_l2)
        
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim)
        super(gaussian2dMapLayerNormalized, self).__init__(**kwargs)

    def build(self, input_shape):
        self.mean = self.init((2,),
                                name='{}_mean'.format(self.name))

        self.sigma = self.init((3,),
                              name='{}_sigma'.format(self.name))


        self.trainable_weights = [self.mean,  self.sigma]

        if self.init_mean is not None:
            K.set_value(self.mean,self.init_mean)
            del self.init_mean
        if self.init_sigma is not None:
            K.set_value(self.sigma,self.init_sigma)
            del self.init_sigma
            
        self.regularizers = []
        
        if self.sigma_regularizer_l2:
            self.sigma_regularizer_l2.set_param(self.sigma)
            self.regularizers.append(self.sigma_regularizer_l2)



    def call(self,x, mask=None):
        x = K.reshape(x,(x.shape[0],x.shape[-2]*x.shape[-1]))
        tolerance = np.asarray(0.01) #use tolerance from stopping the matrix from being un-invertible
        T.set_subtensor(self.sigma[1] ,T.sgn(self.sigma[1])*T.min((T.sqrt(self.sigma[0]*self.sigma[2])-tolerance,T.abs_(self.sigma[1]))))

        inner = (self.spaceVector - self.inv_scale*self.mean.dimshuffle(0,'x'))

        cov = self.inv_scale*T.stack([[self.sigma[0],self.sigma[1]],[self.sigma[1],self.sigma[2]]])
        inverseCov = T.nlinalg.matrix_inverse(cov)
        firstProd =  T.tensordot(inner.T,inverseCov,axes=1)
        malahDistance = T.sum(firstProd*inner.T,axis =1)
        gaussianDistance = T.exp((-1./2.)*malahDistance)
        detCov = T.nlinalg.det(cov)
        denom = 1./(2*np.pi*T.sqrt(detCov))
        gdKernel = T.dot(x,denom*gaussianDistance)
        return gdKernel.dimshuffle(0,'x').astype('float32')
        


    def get_output_shape_for(self, input_shape):
        return (input_shape[0],1)    
        
    def get_config(self):
        config = {'init': self.init.__name__}
        base_config = super(gaussian2dMapLayerNormalized, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class gaussian1dMapLayer(Layer):
    '''Maps a 1d gaussian over the input vector, thus this is a precursor
        gaussian2dMap Layer. 
        This is not mapping of a gaussian over the real number line.
        Each index of the input vector is assumed to be a point along the real number line.
        The mean corresponds to a value along the input vector
        assumes the input vector is spaced [0,1,2,3,4,5,6,...,n]
        
        '''
    def __init__(self,input_dim,init = 'zero', init_mean= None,init_sigma = None,**kwargs):
                 
        self.input_dim = input_dim     
        self.spaceVector = np.asarray(range(input_dim))
        self.init = initializers.get(init)
        if init_mean is None:
            self.init_mean = np.asarray(input_dim/2.0)
        if init_sigma is None:
            self.init_sigma = np.asarray(np.sqrt(input_dim))
        
        self.init_mean = self.init_mean.astype('float32') 
        self.init_sigma = self.init_sigma.astype('float32') 
        
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(gaussian1dMapLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.mean = self.init((),
                                name='{}_mean'.format(self.name))
        self.sigma = self.init((),
                              name='{}_sigma'.format(self.name))
                              
        self.trainable_weights = [self.mean,self.sigma]

        if self.init_mean is not None:
            K.set_value(self.mean,self.init_mean)
            del self.init_mean
        if self.init_sigma is not None:
            K.set_value(self.sigma,self.init_sigma)
            del self.init_sigma


    def call(self,x, mask=None):

        inner =  (self.spaceVector- self.mean)
        beta = (2.*T.sqr(self.sigma))
        gaussianOutput = T.exp( -(1./beta) * T.sqr(inner)) 
        expOutput = T.dot(x,gaussianOutput).dimshuffle(0,'x')
        return  (1./(T.sqrt(beta*np.pi)))*expOutput

        
    def get_config(self):
        config = {'init': self.init.__name__}
        base_config = super(gaussian1dMapLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
class singleTwoSidePReLU(Layer):

    
    '''Parametric Rectified Linear Unit:
    `f(x) = alphas * x for x < 0`,
    `f(x) = x for x >= 0`,
    where `alphas` is a learned array with the same shape as x.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as the input.

    # Arguments
        init: initialization function for the weights.
        weights: initial weights, as a list of a single numpy array.

    # References
        - [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](http://arxiv.org/pdf/1502.01852v1.pdf)
    '''
    def __init__(self, init='zero', weights=None, **kwargs):
        self.supports_masking = True
        self.init = initializers.get(init)
        self.initial_weights = weights
        super(singleTwoSidePReLU, self).__init__(**kwargs)

    def build(self, input_shape):
        self.left_alpha = self.init((),
                                name='{}left_alpha'.format(self.name))
        self.right_alpha = self.init((),
                                name='{}right_alpha'.format(self.name))
        self.trainable_weights = [self.left_alpha,self.right_alpha]

        if self.initial_weights is not None:
            K.set_value(self.left_alpha,self.initial_weights[0])
            K.set_value(self.right_alpha,self.initial_weights[1])
            del self.initial_weights

    def call(self, x, mask=None):
        pos = self.right_alpha*K.relu(x)
        neg = self.left_alpha * (x - abs(x)) * 0.5
        return pos + neg

    def get_config(self):
        config = {'init': self.init.__name__}
        base_config = super(singleTwoSidePReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))        
        
        
        
class singleWeight(Layer):
    ''' just a single weight to multiply all input values,
        input and output are the same shape'''
    
    def __init__(self, init='uniform', weights=None, **kwargs):
        self.supports_masking = True
        self.init = initializers.get(init)
        self.initial_weights = weights
        super(singleWeight, self).__init__(**kwargs)

    def build(self, input_shape):
        self.coef = self.init((),
                                name='{}_coef'.format(self.name))
        self.trainable_weights = [self.coef]

        if self.initial_weights is not None:
            K.set_value(self.coef,self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        val = self.coef * (x) 
        return val

    def get_config(self):
        config = {'init': self.init.__name__}
        base_config = super(singleWeight, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


        
##############################
## Working but not updated ###
##############################      

class gaussian2dMapLayerCorr(Layer):
    ''' assumes 2-d  input
    explicitly defines the variances and the correlation coefficient
    
    '''
    def __init__(self,input_dim,init = 'zero', init_meanX= None,init_meanY= None,init_sigmaX = None,init_sigmaY = None,init_corr = None,**kwargs):
                 
        self.input_dim = input_dim
        
        xSpace = np.linspace(0,input_dim[0]-1,input_dim[0])
        ySpace = np.linspace(0,input_dim[1]-1,input_dim[1])
        spaceMatrix = np.asarray((np.meshgrid(xSpace,ySpace)))
        self.spaceVector = spaceMatrix.reshape((2,input_dim[0]*input_dim[1]))
        
        self.init = initializers.get(init)
        
        if init_meanX is None:
            self.init_meanX = np.asarray(input_dim[0]/2.)
        if init_meanY is None:
            self.init_meanY = np.asarray(input_dim[1]/2.)            
        if init_sigmaX is None:
            self.init_sigmaX = np.asarray(input_dim[0])
        if init_sigmaY is None:
            self.init_sigmaY = np.asarray(input_dim[1])
        if init_corr is None:
            self.init_corr = np.asarray(0.0)
        
        self.init_meanX = self.init_meanX.astype('float32')
        self.init_meanY = self.init_meanY.astype('float32') 
        self.init_sigmaX = self.init_sigmaX.astype('float32')
        self.init_sigmaY = self.init_sigmaY.astype('float32')
        self.init_corr = self.init_corr.astype('float32')
        
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim)
        super(gaussian2dMapLayerCorr, self).__init__(**kwargs)

    def build(self, input_shape):
        self.meanX = self.init((),
                                name='{}_meanX'.format(self.name))
        self.meanY = self.init((),
                        name='{}_meanY'.format(self.name))
        self.sigmaX = self.init((),
                              name='{}_sigmaX'.format(self.name))
        self.sigmaY = self.init((),
                              name='{}_sigmaY'.format(self.name))
        self.corr = self.init((),
                              name='{}_corr'.format(self.name))

        self.trainable_weights = [self.meanX,self.meanY,  self.sigmaX, self.sigmaY, self.corr]

        if self.init_meanX is not None:
            K.set_value(self.meanX,self.init_meanX)
            del self.init_meanX
        if self.init_meanY is not None:
            K.set_value(self.meanY,self.init_meanY)
            del self.init_meanY
        if self.init_sigmaX is not None:
            K.set_value(self.sigmaX,self.init_sigmaX)
            del self.init_sigmaX
        if self.init_sigmaY is not None:
            K.set_value(self.sigmaY,self.init_sigmaY)
            del self.init_sigmaY
        if self.init_corr is not None:
            K.set_value(self.corr,self.init_corr)
            del self.init_corr



    def call(self,x, mask=None):
        
        x = K.reshape(x,(x.shape[0],x.shape[-2]*x.shape[-1]))
        innerX = (self.spaceVector[0] - self.meanX)
        innerY = (self.spaceVector[1] - self.meanY)
        self.corr = T.min((T.max((-0.99,self.corr)),0.99)) #stop correlation from going outside [-1,1]
        corrStuff = (T.sqr(innerX)/T.sqr(self.sigmaX)) + (T.sqr(innerY)/T.sqr(self.sigmaY)) - (2.*self.corr*innerX*innerY)/(self.sigmaX*self.sigmaY)
        expStuff = (-1./(2.*(1-T.sqr(self.corr))))
        malahDistance = T.exp(expStuff*corrStuff)
        denom = 1./(2*np.pi*self.sigmaX*self.sigmaY*T.sqrt(1-T.sqr(self.corr)))
        gaussKernel = denom*malahDistance
        return T.dot(x,gaussKernel).dimshuffle(0,'x').astype('float32')

    def get_output_shape_for(self, input_shape):
        return (input_shape[0],1)    
        
    def get_config(self):
        config = {'init': self.init.__name__}
        base_config = super(gaussian2dMapLayerCorr, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))