# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 11:12:01 2023

@author: EPFL-LHE
"""
# Librairies
import numpy as np
import random
from PIL import Image
import os
from juliacall import Main as jl
import pandas as pd
import cvxpy as cp


def pngToGslib(imagePath):
    '''
    

    Parameters
    ----------
    imagePath : string
        Creates a .gslib file based on the image in argument. It saves the file in the same folder as the image

    Returns
    -------
    None.

    '''
    # Read the image and convert it in gray level
    img = Image.open(imagePath).convert('L')

    # Convert the image into an array
    data = np.array(img)

    # Normalize the values
    data = data / 255.0

    # Create a dataframe for the values
    df = pd.DataFrame(data.flatten(), columns=['Z'])
    
    root, ext = os.path.splitext(imagePath)
    gslibPath = root + ".gslib"

    # Write the gslib file
    with open(gslibPath, 'w') as f:
        f.write("# This file was generated by la Ch'team\n")
        f.write(f"{data.shape[1]} {data.shape[0]}\n")
        f.write("0.000000 0.000000\n")
        f.write("1.000000 1.000000\n")
        f.write("Z\n")

    # Save the gslib file
    df.to_csv(gslibPath, sep=' ', index=False, mode='a', header=False)

def imageQuilting(TIname, TIsize, blur, overlap, tilesize):
    '''
    

    Parameters
    ----------
    TIname : string
        Name of the training image that MUST BE in the folder written in the geostatsimages.jl
    TIsize : array 
        Size of the training image.
    blur : odd int
        Size of the kernel of the Gaussian blur.
    overlap : couple
        Size of the overlap as a fraction of the tilesize.
    tilesize : couple
        Size of the tile in pixels.

    Returns
    -------
    output : array
        Generated image based on the programm from Hoffimann 2017 in Julia

    '''
    # Importing all the usefull modules
    jl.seval('using CUDA')
    jl.seval("using GeoStats")
    jl.seval("using GeoStatsPlots")
    jl.seval("using ImageQuilting")
    jl.seval("using GeoStatsImages")
    jl.seval("using Plots")
    jl.seval("using ImageFiltering")
    jl.seval("using ImageView")
    
    # Sending the argument in julia
    jl.TIname = TIname
    jl.blur = blur
    jl.overlap = overlap
    jl.tilesize = tilesize
    
    # Copying the main code of Hoffimann 
    jl.seval('trainimg = geostatsimage(TIname)')
    jl.seval('function forward(data); img = asarray(data, :Z); krn = KernelFactors.IIRGaussian([blur,blur]); fwd = imfilter(img, krn); georef((fwd=fwd,), domain(data));  end')
    jl.seval('dataTI = forward(trainimg)')
    jl.seval('problem = SimulationProblem(domain(trainimg), :Z => Float64, 1)')
    jl.seval('solver = IQ(:Z => (trainimg = trainimg, overlap = overlap, tilesize = tilesize, soft = (dataTI,dataTI), tol = 0.01, path = :raster))')
    jl.seval('ensemble = solve(problem, solver)')
    
    # Exporting the result in Python and reshape it
    m = jl.seval('ensemble.reals[1][1]')
    a = np.array(m)
    output = a.reshape(TIsize)
    
    # Free the GPU memory (if not, memory full)
    jl.seval('GC.gc()')
    jl.seval('CUDA.reclaim()')
    return output

def transition_probabilities(D, sigma, pi):
    '''
    

    Parameters
    ----------
    D : array
        Matrix of distances between centroids.
    sigma : unsigned float
        Dispersion parameter.
    pi : list
        Distribution of the modes.

    Returns
    -------
    P_optimal : Array
        Matrix of probabilities of transition between modes.

    '''
    E = 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-0.5 * D**2 / sigma**2)
    K = np.diag(1 / np.sum(E, axis=1))
    Q = K @ E  # Use '@' for matrix multiplication
    
    # Dimension of the matrix
    n = Q.shape[0]
    
    # Define the optimization variable
    P = cp.Variable((n, n))
    
    # Define the objective function (Kullback-Leibler divergence)
    kl_div = cp.sum(cp.kl_div(P, Q))
    
    # Define the stationarity constraint
    constraints = [pi @ P == pi]  # Use '@' for matrix multiplication
    
    # Ensure the elements of P are positive
    constraints += [P >= 0]
    
    # Ensure that the sum of the components of each line is one
    constraints += [ P @ np.ones((n,1)) == np.ones((n,1))]
    
    # Define and solve the optimization problem
    prob = cp.Problem(cp.Minimize(kl_div), constraints)
    
    success = False
    while not success:
        try:
            prob.solve()
            success = True
        except cp.error.SolverError:
            print("Solver failed, retrying")
            sigma = random.uniform(1, 100)
            E = 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-0.5 * D**2 / sigma**2)
            K = np.diag(1 / np.sum(E, axis=1))
            Q = K @ E  # Use '@' for matrix multiplication
            n = Q.shape[0]
            P = cp.Variable((n, n))
            kl_div = cp.sum(cp.kl_div(P, Q))
            constraints = [pi @ P == pi]  # Use '@' for matrix multiplication
            constraints += [P >= 0]
            constraints += [ P @ np.ones((n,1)) == np.ones((n,1))]
            prob = cp.Problem(cp.Minimize(kl_div), constraints)

    
    # The optimal solution
    P_optimal = P.value
    
    return P_optimal


def markov_chain(P, stepmax, lambda_, pi):
    '''
    

    Parameters
    ----------
    P : array
        Matrix of probabilities of transition between modes
    stepmax : unsigned int
        Number of steps of the Markov Chain (400 in the Hoffimann paper).
    lambda_ : unsigned float
        Parameter that rules the Poisson law.
    pi : list
        Distribution of the modes.

    Returns
    -------
    statelist : list
        List of all the modes generated by the Markov chain.

    '''
    # Markov chain initialization
    step = 1 
    state = 0
    r = random.random()
    somme0 = pi[state]
    while somme0<r:
        state = state + 1
        somme0 = somme0 + pi[state]
    statelist = np.zeros(stepmax)
    statelist[0] = state
    random.seed()

    # Markov chain route
    while step < stepmax:
        # Determine how many times to repeat the current state based on th Poisson law
        repetitions = int(np.ceil(-np.log(random.random()) / lambda_))
        for _ in range(repetitions):
            if step >= stepmax:
                break
            statelist[step] = state
            step += 1
        
        # If there is a change of mode, it is based on the transition probabilities matrix
        r = random.random()
        nextstate = 0
        somme = P[state,nextstate]
        while r > somme:
            nextstate += 1
            somme += P[state,nextstate]
        state = nextstate

    return statelist
