# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 13:26:42 2017

@author: phil
"""
import re
import os
import h5py
import numpy as np
from scipy import io

pySIALocation =  str(os.path.abspath(os.path.dirname(__file__)))
clabLocationSep = pySIALocation.split(os.path.sep)
pySIALocation =  os.path.sep.join(clabLocationSep[:-1])
'''TODO: add padding function'''
from curtis_lab_utils import p_stimCropper
from utils.p_utils import optDef
import sys
def getEstRegPred(resp):
    #assumes that trial lengths are all 375
    estIdx = int((2.0/3.0)*len(resp))

    regIdx = int((5.0/6.0)*len(resp))
    predIdx = len(resp)
    return estIdx,regIdx,predIdx
    
def splitDataSet(X,y):
    ''' splits data set into estimation set, regularization set and prediction set'''
    
    estIdx,regIdx,predIdx = getEstRegPred(y)
    X_train = X[:estIdx]
    X_reg = X[estIdx:regIdx]
    X_test = X[regIdx:predIdx]
    y_train = y[:estIdx]
    y_reg =y[estIdx:regIdx]
    y_test = y[regIdx:predIdx]
    return X_train,X_reg,X_test,y_train,y_reg,y_test

def getPaths():

    codeLocationSep = pySIALocation.split(os.path.sep)
    pyCodeLocation =  os.path.sep.join(codeLocationSep[:-1])
    codeLocationSep = pyCodeLocation.split(os.path.sep)
    CODELocation = os.path.sep.join(codeLocationSep[:-1])
    SIALocation = os.path.sep.join(codeLocationSep[:-2])
    DATAlocation = os.path.sep.join((SIALocation,'DATA'))
    STRF_MOVIESlocation = os.path.sep.join((SIALocation,'STRF_MOVIES'))
    channelLocation = os.path.sep.join((SIALocation,'channelScripts'))
    AllNeuronLocation = os.path.sep.join((channelLocation,'AllNeurons'))
    Grating_MoviesLocation = os.path.sep.join((SIALocation,'Grating_Movies'))
    AnalysisLocation = os.path.sep.join((channelLocation,'Analysis'))
    tempLocation = os.path.sep.join((CODELocation,'temp'))    
    
    os.chdir(SIALocation)
    print('pySIALocation ' +  pySIALocation)
    print('current directory ' +os.getcwd())

    os.chdir(os.path.abspath('MOVIES'))
    MOVIESLocation = os.getcwd()
    
    pathStruct = dict()
    pathStruct['pySIALocation'] = pySIALocation
    pathStruct['pyCodeLocation'] = pyCodeLocation
    pathStruct['CODELocation'] = CODELocation
    pathStruct['SIALocation'] = SIALocation
    pathStruct['DATAlocation'] = DATAlocation
    pathStruct['channelLocation'] = channelLocation
    pathStruct['AllNeuronLocation'] = AllNeuronLocation
    pathStruct['STRF_MOVIESlocation'] = STRF_MOVIESlocation
    pathStruct['MOVIESLocation'] = MOVIESLocation
    pathStruct['Grating_MoviesLocation'] = Grating_MoviesLocation
    pathStruct['AnalysisLocation'] = AnalysisLocation
    pathStruct['tempLocation'] = tempLocation
    os.chdir(pySIALocation)
    return pathStruct




def getCropWin(cropFileName):
    try:
        with h5py.File(cropFileName,'r') as f:
            cropWin = np.asarray(f['cropWin'])

    except:
        f = io.loadmat(cropFileName)
        cropWin = f['cropWin'][0]
        
    if np.sum(cropWin) == 0:
        cropWin = np.asarray([1,480,1,480])
    # convert float to int and make format ready for later process
    if np.array_equal(cropWin.astype(np.int), cropWin):
        return np.squeeze(cropWin.astype(np.int))
    else:
        raise RuntimeError("ERROR: cropWin is not int values!")

def getStrfData(strfFile,dataFile,stim_options):
    
    pathStruct = getPaths()
    
    stim_options['Movie_Creation']  = optDef('Movie_Creation',stim_options,'PYTHON')
    #IMPORT THE RESPONSE
    dataFileName = dataFile
    dataFolder = os.path.sep.join((pathStruct['DATAlocation'],dataFileName))
    dataFileStrf = dataFolder + os.path.sep + dataFileName + '_resp.mat'
    
    
    resp =np.transpose(np.array(io.loadmat(dataFileStrf)['response']))
    resp = np.squeeze(resp)
    
    #IMPORT THE STIM       
    channelStruct = io.loadmat(pathStruct['AllNeuronLocation'] + os.path.sep + dataFile +'.mat')
    moviePath = str(channelStruct['stim']['mv'][0][0][0]['pathName'][0][0]) #getting the movie name from the awful matlab-python conversion
    movieName = re.split("\\\\|/",moviePath)[-1] #need to account for // or \\||/
 

    
    if 'Crop' in strfFile:

        print(movieName)        
       
        cropFileName = dataFolder + os.path.sep + dataFile+'_'+ strfFile +'.mat'        
        print('starting cropping')
        #will need to import create the full 480x480 movie, then crop and downsample
        cropWin = getCropWin(cropFileName)

        if stim_options['Movie_Creation'] == 'MATLAB':
            print('MATLAB movie Creation disabled')
#            print('Using MATLAB to create the movie')
#            winLen = len(range(cropWin[2],cropWin[3]+1))
#            assert winLen == len(range(cropWin[0],cropWin[1]+1))
#            cropKern = selectCropKern(stim_options['cropKern'],winLen)
#            downsampRatio = str(winLen) + ' to ' + str(cropKern) 
#            stim = matEngine.createStrfMovie(movieName,cropKern,cropWin)
            
        elif stim_options['Movie_Creation'] == 'PYTHON':
            print('Using PYTHON to create the movie')
            stim,downsampRatio = generateStimFromStrfMat(movieName,cropWin,stim_options)
    else:
        #just import the necessary strfFile
        movieFolder = pathStruct['STRF_MOVIESlocation'] + os.path.sep +movieName
        os.chdir(movieFolder)   
        
        movieFile = movieName + '_' + strfFile + '.mat'
        
        with h5py.File(movieFile,'r') as f:
            stim = np.array(f['stimMovieAll'])
            winLen = np.sqrt(np.size(stim,1))
        

        downsampRatio = '480 to ' + str(winLen.astype(np.uint32))
    
        cropWin = [1,480,1,480]
      

    return stim,resp,downsampRatio,cropWin
    
def getRespSet(neuronName):
    #IMPORT THE RESPONSE
    pathStruct = getPaths()
    dataFileName = neuronName
    dataFolder = os.path.sep.join((pathStruct['DATAlocation'],dataFileName))
    dataFileStrf = dataFolder + os.path.sep + dataFileName + '_estSetResp.mat'
    estRespSet =np.array(io.loadmat(dataFileStrf)['est_resp'])
    dataFileStrf = dataFolder + os.path.sep + dataFileName + '_regSetResp.mat'
    regRespSet =np.array(io.loadmat(dataFileStrf)['reg_resp'])
    dataFileStrf = dataFolder + os.path.sep + dataFileName + '_predSetResp.mat'
    predRespSet =np.array(io.loadmat(dataFileStrf)['pred_resp'])
    
    return estRespSet,regRespSet,predRespSet

def generateStimFromStrfMat(movieName,cropWin,stim_options= None):
    '''A python based method to read the SIA Movies
       !!! This method uses the skimage resize function, which is not that good
       Better to use matEngine.createStrfMovie 
    '''    
    
    
    '''Set Movie Folder Here'''
    pathStruct = getPaths()
    allMoviesFolder = pathStruct['MOVIESLocation']
    print('Current movie folder: ' + allMoviesFolder)
    
    ''' default Setting''' 
    downsampSetting = optDef('cropKern',stim_options,('Value',30))
    reshapeOrder = optDef('Reshape_Order',stim_options,'F')
    interpMethod = optDef('Interpolation',stim_options,'bilinear')

    
    movieFolder = allMoviesFolder+ os.path.sep +movieName
    print(movieName)  
        
    os.chdir(movieFolder) 
    
    #Set up cropper for MOVIE CLIP STIMULI: x_axis = 0,y_axis = 1,t_axis = 2
    myCropper = p_stimCropper.stimCropper(0,1,2,
                                          reshapeOrder = reshapeOrder,
                                          cropWinIndexStyle='MATLAB',
                                          interp = interpMethod)
    #set the cropWin
    myCropper.setCropWin(cropWin)
    #set the chosen cropKern
    myCropper.selectAndSetCropKern(downsampSetting)
    downsampRatio = myCropper.cropString()

    #run throught movies and crop them
    for i in np.arange(30) +1:
        print(i)
        if i <= 20:
            movieNumber = "%02d" %i
            movieFileName = movieName +'_0000_' +movieNumber+ '.mat'
        elif (i >20) and (i <=25):                
            movieNumber = "%02d" %(i-20)
            movieFileName = movieName +'_reg_0000_' +movieNumber+ '.mat'
            
        elif i >25:
            movieNumber = "%02d" %(i-25)
            movieFileName = movieName +'_pred_0000_' +movieNumber+ '.mat'

        currMovie =io.loadmat(movieFileName)['mvMovie']
        
        if i == 1:
            numFrames = np.size(currMovie,2)
            stim = np.zeros((30*numFrames,np.square(myCropper.cropKern))) 
        

        reshapedMovie = myCropper.shapeForModel(currMovie)
        stim[numFrames*(i-1):numFrames*i,:] = reshapedMovie
        

    return stim,downsampRatio

def readStrfMatGUI():
    #get strf mat file through GUI
    try:
        from Tkinter import Tk
        from tkFileDialog import askopenfilename
    except:
        from tkinter import Tk
        from tkinter.filedialog import askopenfilename
    
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
    
    #cropping window
    try:
        cropWin = eval(str(input('Cropping Window? (ex: [0,480,0,480]) : ')))
        cropWin = np.asarray(cropWin)
    except:
        print('Using default cropping window [0,480,0,480]')
        cropWin = 0
    #
        
    #read the mat file        
    return readStrfMat(filename,cropWin)
    
def readStrfMat(filename,cropWin = 0):
    #default crop
    if np.sum(cropWin) ==0:
        cropWin = np.asarray([1,480,1,480])  
    print(cropWin)
   

    
    channelStruct = io.loadmat(filename)
    
    #IMPORT THE STIM       
    moviePath = str(channelStruct['stim']['mv'][0][0][0]['pathName'][0][0]) #getting the movie name from the awful matlab-python conversion
    movieName = re.split("\\\\|/",moviePath)[-1] #need to account for // or \\||/
    stim,downsampRatio = generateStimFromStrfMat(movieName,cropWin)
    
    
    #IMPORT THE RESP
    respName = filename[:-4] + '_resp.mat'
    resp = np.transpose(np.array(io.loadmat(respName)['response']))
    resp = np.squeeze(resp)
    #
  
       
    

    return stim,resp,downsampRatio
def setReorgModules():
    ''' function required to run before using the rest of these functions
        if the pickles were generated before the pySIA reorganization
    '''    
    
    from models import a_allModelInfo,k_allModelInfo,k_shapeStimulus
    sys.modules['a_allModelInfo'] = a_allModelInfo
    sys.modules['k_allModelInfo'] = k_allModelInfo
    sys.modules['k_shapeStimulus'] = k_shapeStimulus
    return  