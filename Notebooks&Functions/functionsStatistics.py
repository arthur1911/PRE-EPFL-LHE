# Librairies
import numpy as np
import pandas as pd
import os
from os.path import join
import cv2
from juliacall import Main as jl


def wettedArea(image):
    '''
    

    Parameters
    ----------
    image : array
        Image of the experiment.

    Returns
    -------
    w : unisgned float
        Proportion of water.

    '''
    totalPixels = np.size(image)
    wetPixels = np.sum(image == 0)
    w = wetPixels / totalPixels
    return w

def calculateResiduals(data, windowSize):
    '''
    

    Parameters
    ----------
    data : list
        List containing the wetted area of the time serie.
    windowSize : unsigned int
        Size of the window of the moving average.

    Returns
    -------
    residuals : list
        Returns (the wetted area - the moving average) (w0 in hoffimann's paper).

    '''
    
    # Compute the moving average
    filteredData = pd.Series(data).rolling(window=windowSize, min_periods=1).mean().values
    
    # Do the difference
    residuals = data - filteredData
    return residuals

def calculateVariogram(residuals, timeLag):
    '''
    

    Parameters
    ----------
    residuals : list
        w0 in hoffimann's paper.
    timeLag : unsigned float
        timelag.

    Returns
    -------
    variogram : list
        Computes the variogram just as Hoffimann.

    '''
    # Initialization
    variogram = 0
    count = 0

    for i in range(len(residuals) - timeLag):
        diff = residuals[i] - residuals[i + timeLag]
        variogram += diff ** 2

        count += 1
    
    variogram = variogram / (2 * count)
    return variogram

def variogram(folderPath, timelag):
    '''
    

    Parameters
    ----------
    folderPath : string
        Path of the folder which contains the time serie.
    timelag : list
        List of the timelags.

    Returns
    -------
    variogram : list
        Computes the variogram just as Hoffimann.
    w0 : list
        The residuals of the wetted area.
    w : list
        The wetted area.

    '''
    # Get the images in the file
    allFiles = os.listdir(folderPath)
    images = [file for file in allFiles]
    N = len(images)
    w = [0 for i in range(N)]

    # Calculate the wetted area
    for i in range(N):
        nom_image = join(folderPath, images[i])
        image = cv2.imread(nom_image)
        w[i] = wettedArea(image)
        
    # Calculate the residuals and the variogram
    w0 = calculateResiduals(w, 20)
    variogram = [calculateVariogram(w0, int(timelag[i])) for i in range(len(timelag))]
    return np.array(variogram).astype(np.float32)

def consecutiveMHD(folderPath):
      '''
      

      Parameters
      ----------
      folderPath : string
          Path of the folder which contains the time serie.

      Returns
      -------
      MHD : list
          list of the MHD between two consecutive images.

      '''
      # Get the number of images
      allFiles = os.listdir(folderPath)
      images = [file for file in allFiles]
      N = len(images)
      
      # Initialization of the list
      MHD = [0 for i in range(N-1)]
      
      jl.seval('using Images')
      jl.seval('using ImageDistances')
      jl.seval('using ImageMagick')
      jl.seval('using Distances')
      
      jl.seval('function edge(img, threshold=.45); grads = imgradients(img, KernelFactors.ando3); mag = hypot.(grads...); mag .> threshold; end')

      for i in range(N-1):
          nom_image1 = join(folderPath, images[i])
          image1 = cv2.imread(nom_image1, cv2.IMREAD_GRAYSCALE)
          _, image1 = cv2.threshold(image1, 128, 1, cv2.THRESH_BINARY)
          jl.img1 = image1
          edge1 = jl.seval('edge(img1)')
          edge1 = np.array(edge1)
          edge1 = edge1.reshape(np.shape(image1))
          nom_image2 = join(folderPath, images[i+1])
          image2 = cv2.imread(nom_image2, cv2.IMREAD_GRAYSCALE)
          _, image2 = cv2.threshold(image2, 128, 1, cv2.THRESH_BINARY)
          jl.img2 = image2
          edge2 = jl.seval('edge(img2)')
          edge2 = np.array(edge2)
          edge2 = edge2.reshape(np.shape(image1))
          jl.image1 = edge1
          jl.image2 = edge2
          MHD[i] = jl.seval('ImageDistances.modified_hausdorff(image1, image2)')
      return np.array(MHD).astype(np.float32)  

def returnStatistics(MHDordered):
    '''
    

    Parameters
    ----------
    MHDordered : list
        List of the MHD ordered from the lowest to the biggest value.

    Returns
    -------
    returnPeriod : list
        Return the return period.
    returnLevel : list
        Return the return level.

    '''
    # Initialization
    t = 0
    stepmax = np.size(MHDordered)
    returnPeriod = np.zeros(np.shape(MHDordered))
    returnLevel = np.zeros(np.shape(MHDordered))
    
    # Computes the return level and period for each time step
    while t < stepmax:
        returnPeriod[t] = 1/(1-(t+1)/(stepmax+1))
        returnLevel[t] = MHDordered[t]
        t += 1
    return np.array(returnPeriod).astype(np.float32), np.array(returnLevel).astype(np.float32)

def calculate_extreme_value_index(MHD):
    hausdorff_distances = np.sort(MHD)[::-1]
    hausdorff_distances = hausdorff_distances[2:]
    N = len(hausdorff_distances)
    extreme_value_indices = []
    for k in range(1, N):
        # Calculate mean excess
        mean_excess = np.mean(hausdorff_distances[:k] - hausdorff_distances[k])
        extreme_value_indices.append(mean_excess)
    return extreme_value_indices

def get_folder_names(folderTimesSeries):
    for folder in os.listdir(folderTimesSeries):
        yield folder

