# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 15:16:10 2017

@author: phil
"""

from models import strflab_shapeStimulus
from models import k_buildModel
from models import k_defaultOptions
from models import k_allModelInfo

class kRTAC(k_allModelInfo.k_Model):
    model_name = 'kRTAC'

    def __init__(self):
        super(kRTAC,self).__init__()
        return
    def defaultOptions(self,options = dict()):
        return k_defaultOptions.kRegression(options)
    def shapeStimulus(self):
        return strflab_shapeStimulus.kRTACStyle
    def buildModel(self,options):
        #uses LN model, but we change the input stimulus
        return k_buildModel.buildkRegression(options)
