# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 14:49:08 2016

@author: phil

Module for information on all models. For a model to work in pySIA, 
it should match the abstract classes here.
Of course, you can build your own abstract classes outside of the these functions,
so long as they have the following functions (to work with the rest of the code)

Models need:

    Stimulus shaping function (shapeStimulus)
        Takes the XYT stim and turns it into something keras usable.
        This shape must be equal to the shape keras uses to build the model.
    
        (Input shape calculator (_getInputShape) is also needed, usually the input shape is the stimulus shape
        but with the 0th axis (nb_samples) removed. If this is untrue for the model (e.g. time-separable
        models), then you will need to give the correct _getInputShape function)
    
    Default parameters function (defaultOptions)
        Sets the default options for the model for any missing options. This should cover
        all options for the building function
    
    Building function (buildModel)
        Builds the model from a set of options
    
The k_Model class also includes methods for:

buildStimScaler (_buildCalcAndScale is similar but also applies the transform as well)
    This is a sub class that will scale the stimulus and store the parameters used for scaling.
    Similar to the sci-kit learn preprocessing classes
    
    As of right now (DEC232016), the stimScaler class uses the options
    options['Scaling_Method'] = ('Scaling_Method',axis)


"""
from models import k_buildModel
from models import k_defaultOptions
from models import k_shapeStimulus

class k_Model(object):
    model_name = 'abstractModel'
    input_shape = None
    stimScaler = None
    def __init__(self):
        return
    def defaultOptions(self,options = dict()):
        '''Fills in the options dictionary with default values if any are missing '''
        pass
    def shapeStimulus(self):
        '''Returns the function to shape the stimulus for this model '''
        pass 
    def buildModel(self,options):
        '''Builds a model given options, the options must have all the required parameters'''
        pass
    def buildStimScaler(self,options):
        ''' Builds a scaling object, this is the same for all models'''
        self.stimScaler = k_shapeStimulus.scaleStimulusClass(options)
        return
    def _buildCalcAndScale(self,options,stim):
        ''' Builds a scaling object, calculates the parameters for the scaler and applies the
            transform on the stim.'''
        self.stimScaler = k_shapeStimulus.scaleStimulusClass(options)
        self.stimScaler.calcScaleParams(stim)
        return self.stimScaler.applyScaleTransform(stim)
    def _getInputShape(self,X):
        ''' Default Keras shape has (nb_samples,featuredim_1,... featuredim_n
            Simply return the shape without nb_samples'''
        
        return (X.shape[1:])
    
    
class kRegression(k_Model):
    model_name = 'kRegression'

    def __init__(self):
        super(kRegression,self).__init__()
        return
    def defaultOptions(self,options = dict()):
        return k_defaultOptions.kRegression(options)
    def shapeStimulus(self):
        return k_shapeStimulus.kRegressionStyle
    def buildModel(self,options):
        return k_buildModel.buildkRegression(options)

   
class kConvNet(k_Model):
    model_name = 'kConvNet'

    def __init__(self):
        super(kConvNet,self).__init__()
        return
    def defaultOptions(self,options = dict()):
        return k_defaultOptions.kConvNet(options)
    def shapeStimulus(self):
        return k_shapeStimulus.kConvNetStyle
    def buildModel(self,options):
        return k_buildModel.buildkConvNet(options)

class kConvGaussNet(k_Model):
    model_name = 'kConvGaussNet'
    def __init__(self):
        super(kConvGaussNet,self).__init__()
        return
    def defaultOptions(self,options = dict()):
        return k_defaultOptions.kConvGaussNet(options)
    def shapeStimulus(self):
        return k_shapeStimulus.kConvNetStyle
    def buildModel(self,options):
        return k_buildModel.buildkConvGaussNet(options)
        
class kConvDOGNet(k_Model):
    model_name = 'kConvDOGNet'
    def __init__(self):
        super(kConvDOGNet,self).__init__()
        return
    def defaultOptions(self,options = dict()):
        return k_defaultOptions.kConvDOGNet(options)
    def shapeStimulus(self):
        return k_shapeStimulus.kConvNetStyle
    def buildModel(self,options):
        return k_buildModel.buildkConvDOGNet(options)

class kConvGaussEXP(k_Model):
    model_name = 'kConvGaussEXP'
    def __init__(self):
        super(kConvGaussEXP,self).__init__()
        return
    def defaultOptions(self,options = dict()):
        return k_defaultOptions.kConvGaussEXP(options)
    def shapeStimulus(self):
        return k_shapeStimulus.kConvNetStyle
    def buildModel(self,options):
        return k_buildModel.buildkConvGaussEXP(options)

class kConvTwoGaussNet(k_Model):
    model_name = 'kConvTwoGaussNet'

    def __init__(self):
        super(kConvTwoGaussNet,self).__init__()
        return
    def defaultOptions(self,options = dict()):
        return k_defaultOptions.kConvTwoGaussNet(options)
    def shapeStimulus(self):
        return k_shapeStimulus.kConvNetStyle
    def buildModel(self,options):
        return k_buildModel.buildkConvTwoGaussNet(options)

class kConvTwoAffineNet(k_Model):
    model_name = 'kConvTwoAffineNet'

    def __init__(self):
        super(kConvTwoAffineNet,self).__init__()
        return
    def defaultOptions(self,options = dict()):
        return k_defaultOptions.kConvTwoAffineNet(options)
    def shapeStimulus(self):
        return k_shapeStimulus.kConvNetStyle
    def buildModel(self,options):
        return k_buildModel.buildkConvTwoAffineNet(options)
        
class kConvNetTimeSeparable(k_Model):
    model_name = 'kConvNetTimeSeparable'

    def __init__(self):
        super(kConvNetTimeSeparable,self).__init__()
        return
    def defaultOptions(self,options = dict()):
        return k_defaultOptions.kConvNetTimeSeparable(options)
    def shapeStimulus(self):
        def timeSep(func):
            def wrapper(X,options):
                return k_shapeStimulus.timeSeparateData(func(X,options))
            return wrapper
        return timeSep(k_shapeStimulus.kConvNetStyle)
        
    def buildModel(self,options):
        return k_buildModel.buildkConvNetTimeSeparable(options)
    def _getInputShape(self,X):
        return (X[0].shape[1:])
        
class kConvNetTimeSeparableDOG(k_Model):
    model_name = 'kConvNetTimeSeparableDOG'

    def __init__(self):
        super(kConvNetTimeSeparableDOG,self).__init__()
        return
    def defaultOptions(self,options = dict()):
        return k_defaultOptions.kConvNetTimeSeparableDOG(options)
    def shapeStimulus(self):
        def timeSep(func):
            def wrapper(X,options):
                return k_shapeStimulus.timeSeparateData(func(X,options))
            return wrapper
        return timeSep(k_shapeStimulus.kConvNetStyle)
        
    def buildModel(self,options):
        return k_buildModel.buildkConvNetTimeSeparableDOG(options)
    def _getInputShape(self,X):
        return (X[0].shape[1:])

class kRadialNet(k_Model):
    model_name = 'kRadialNet'

    def __init__(self):
        super(kRadialNet,self).__init__()
        return
    def defaultOptions(self,options = dict()):
        return k_defaultOptions.kRadialNet(options)
    def shapeStimulus(self):
        def timeSep(func):
            def wrapper(X,options):
                return k_shapeStimulus.timeSeparateDataNoExpand(func(X,options))
            return wrapper
        return timeSep(k_shapeStimulus.kRadialStyle)
        
    def buildModel(self,options):
        return k_buildModel.buildkRadialNet(options)
    def _getInputShape(self,X):
        return (X[0].shape[1:])
        
class kRadialNetDOG(k_Model):
    model_name = 'kRadialNetDOG'

    def __init__(self):
        super(kRadialNetDOG,self).__init__()
        return
    def defaultOptions(self,options = dict()):
        return k_defaultOptions.kRadialNetDOG(options)
    def shapeStimulus(self):
        def timeSep(func):
            def wrapper(X,options):
                return k_shapeStimulus.timeSeparateDataNoExpand(func(X,options))
            return wrapper
        return timeSep(k_shapeStimulus.kRadialStyle)
        
    def buildModel(self,options):
        return k_buildModel.buildkRadialNetDOG(options)
    def _getInputShape(self,X):
        return (X[0].shape[1:])
        
class kOnOffConvNet(k_Model):
    model_name = 'kOnOffConvNet'

    def __init__(self):
        super(kOnOffConvNet,self).__init__()
        return
    def defaultOptions(self,options = dict()):
        return k_defaultOptions.kOnOffConvNet(options)
    def shapeStimulus(self):
        return k_shapeStimulus.kConvNetStyle
        
    def buildModel(self,options):
        return k_buildModel.buildkOnOffConvNet(options)
class kConvSplitNet(k_Model):
    model_name = 'kConvSplitNet'

    def __init__(self):
        super(kConvSplitNet,self).__init__()
        return
    def defaultOptions(self,options = dict()):
        return k_defaultOptions.kConvSplitNet(options)
    def shapeStimulus(self):
        return k_shapeStimulus.kConvNetStyle
        
    def buildModel(self,options):
        return k_buildModel.buildkConvSplitNet(options)

