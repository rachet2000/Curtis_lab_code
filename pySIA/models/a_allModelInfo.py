# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 12:54:43 2017

@author: amol
"""

from models import k_allModelInfo 
from models import k_shapeStimulus
from models import a_defaultOptions
from models import a_buildModel

class kConvNet(k_allModelInfo.k_Model):
    model_name = 'kConvNet'

    def __init__(self):
        super(kConvNet,self).__init__()
        return
    def defaultOptions(self,options = dict()):
        return a_defaultOptions.kConvNet(options)
    def shapeStimulus(self):
        return k_shapeStimulus.kConvNetStyle
    def buildModel(self,options):
        return a_buildModel.buildkConvNet(options)


class kConvNetDropout(k_allModelInfo.k_Model):
    model_name = 'kConvNetDropout'

    def __init__(self):
        super(kConvNetDropout,self).__init__()
        return
    def defaultOptions(self,options = dict()):
        return a_defaultOptions.kConvNetDropOut(options)
    def shapeStimulus(self):
        return k_shapeStimulus.kConvNetStyle
    def buildModel(self,options):
        return a_buildModel.buildkConvNetDropOut(options)

class kConvNetTimeSeparable(k_allModelInfo.k_Model):
    model_name = 'kConvNetTimeSeparable'
    
    def __init__(self):
        super(kConvNetTimeSeparable,self).__init__()
        return
    def defaultOptions(self,options = dict()):
        return a_defaultOptions.kConvNetTimeSeparable(options)
    def shapeStimulus(self):
        def timeSep(func):
            def wrapper(X,options):
                return k_shapeStimulus.timeSeparateData(func(X,options))
            return wrapper
        return timeSep(k_shapeStimulus.kConvNetStyle)
    def buildModel(self,options):
        return a_buildModel.buildkConvNetTimeSeparable(options)
    def _getInputShape(self,X):
        return (X[0].shape[1:])