# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 14:04:50 2016

@author: phil
"""

''' Method for matching old model names with their corresponding model class'''

import k_allModelInfo
def matchModel(model_name):
    if model_name == 'kRegression':
        model_class = k_allModelInfo.kRegression()
    elif model_name == 'kConvNet':
        model_class = k_allModelInfo.kConvNet()
    elif model_name == 'kConvGaussNet':
        model_class = k_allModelInfo.kConvGaussNet()
    elif model_name == 'kConvDOGNet':
        model_class = k_allModelInfo.kConvDOGNet()
    elif model_name == 'kConvTwoGaussNet':
        model_class = k_allModelInfo.kConvTwoGaussNet()
    elif model_name == 'kConvTwoAffineNet':
        model_class = k_allModelInfo.kConvTwoAffineNet()
    elif model_name == 'kConvNetTimeSeparable':
        model_class = k_allModelInfo.kConvNetTimeSeparable()
    elif model_name == 'kConvNetTimeSeparableDOG':
        model_class = k_allModelInfo.kConvNetTimeSeparableDOG()
    elif model_name == 'kRadialNet':
        model_class = k_allModelInfo.kRadialNet()
    elif model_name == 'kRadialNetDOG':
        model_class = k_allModelInfo.kRadialNetDOG()
    else:
        raise('model unknown')

    
    return model_class
    