# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 13:23:20 2017

@author: amol

Keras layers written by Amol
"""
from __future__ import absolute_import
from keras.engine import Layer
from keras import backend as K


class PoissonNoise(Layer):
    ''' Apply to the input a multiplicative Poisson noise. It's a modification of 
    GaussianDropout in keras.
    
    As it is a regularization layer, it is only active at training time.
    
    # Arguments
        p: float, drop probability
    
    # Input shape
        Arbitrary. 
        
    # Output shape
        Same shape as input.
        
    # References
        [Mike Oliver's PhD Thesis Chapter 3.4.6]
    '''
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(PoissonNoise,self).__init__(**kwargs)
    
    def call(self, x, mask = None):
        noise_x = x + K.sqrt(x) * K.random_normal(shape = K.shape(x), mean = 0, std= 1)
                               
        return K.in_train_phase(noise_x, x)
        
        
    def get_config(self):
        
        base_config = super(PoissonNoise, self).get_config()
        return dict(list(base_config.items()))
        