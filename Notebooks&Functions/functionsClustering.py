# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 14:48:46 2023

@author: EPFL-LHE
"""

import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
from os.path import join
import os
import cv2
from juliacall import Main as jl

def GetCentroids(path):
    '''
    

    Parameters
    ----------
    path : string
        Path where the centroids are saved.

    Returns
    -------
    centroids : List of arrays
        List of all centroids converted in arrays.

    '''
    # Get all the centroids in the folder
    allFiles = os.listdir(path)
    centroid_files = [file for file in allFiles if file.endswith(".png")]
    
    # Allocating the memory for the list of arrays "centroids"
    m = len(centroid_files)
    nom_centroid = join(path, centroid_files[0])
    centroid1 = cv2.imread(nom_centroid, cv2.IMREAD_GRAYSCALE)
    centroids = [np.zeros_like(centroid1) for _ in range(m)]

    # Put the centroids in the list "centroids"
    for i in range(m):
        nom_centroid = join(path, centroid_files[i])
        # Read the image in the folder, convert in gray level and put it in the list
        image = cv2.imread(nom_centroid, cv2.IMREAD_GRAYSCALE)
        _, image = cv2.threshold(image, 128, 1, cv2.THRESH_BINARY)
        centroids[i] = image
        
    
    return centroids

def MHDMatrixJulia(folder):
    '''
    

    Parameters
    ----------
    folder : string
        Path where the centroids are saved.

    Returns
    -------
    D : array
        Matrix of MHD between each pair of pictures contained in the folder.

    '''
    jl.seval('using Images')
    jl.seval('using ImageDistances')
    jl.seval('using ImageMagick')
    jl.seval('using Distances')

    img_files = sorted([f for f in os.listdir(folder) if f.endswith('.png')])  

    jl.img_files = img_files
    jl.folder = folder
    
    jl.seval('function edge(img, threshold=.45); grads = imgradients(img, KernelFactors.ando3); mag = hypot.(grads...); mag .> threshold; end')

    centroids = GetCentroids(folder)
    m = len(centroids)
    D = np.zeros((m, m))
    for i in range(m):
        print(f"Ligne {i} sur {m}")
        jl.imgi = centroids[i]
        edgei = jl.seval('edge(imgi)')
        edgei = np.array(edgei)
        edgei = edgei.reshape(np.shape(centroids[0]))
        for j in range(m):
            jl.imgj = centroids[j]
            edgej = jl.seval('edge(imgj)')
            edgej = np.array(edgej)
            edgej = edgej.reshape(np.shape(centroids[0]))
            jl.image1 = edgei
            jl.image2 = edgej
            D[i,j] = jl.seval('ImageDistances.modified_hausdorff(image1, image2)')
    return D

def DBSCAN_sihouette(mhd_matrix, min_samples_list, epsilon_list, min_points):
    '''
    

    Parameters
    ----------
    mhd_matrix : array
        Matrix of MHD between each pair of experimental pictures.
    min_samples_list : list
        List of values of min_samples to try.
    epsilon_list : list
        List of values of epsilon to try.
    min_points : unsigned int
        Minimum number of points in clusters.

    Returns
    -------
    list_silhouette : array
        Values of the silhouette score for each combinaison of (min_samples, epsilon).

    '''

    # Initialization of the silhouette matrix
    list_silhouette = np.full((len(min_samples_list), len(epsilon_list)), -np.inf)
    
    # Moving the parameters and computing the silhouette index every time the condition of the minimum number of points is fullfilled
    for j, epsilon in enumerate(epsilon_list):
        for i, min_samples in enumerate(min_samples_list):
            # Doing DBSCAN
            dbscan_clusterer = DBSCAN(eps=epsilon, min_samples=min_samples)
            # Perform clustering on the mhd matrix
            cluster_labels = dbscan_clusterer.fit_predict(mhd_matrix)
            # Ignoring the noise points
            noise_mask = cluster_labels != -1
            cluster_labels_filtered = cluster_labels[noise_mask]
            unique_labels = np.unique(cluster_labels_filtered)
            mhd_matrix_filtered = mhd_matrix[noise_mask][:, noise_mask]
    
            # Calculate silhouette scores if the condition is fullfilled
            if len(unique_labels) > 1 and len(cluster_labels_filtered) > min_points:
                silhouette_avg = silhouette_score(mhd_matrix_filtered, cluster_labels_filtered)
                # Save the values in a list
                list_silhouette[i, j] = silhouette_avg
    
    return list_silhouette

def DBSCAN_clustering(mhd_matrix, min_samples, epsilon):
    '''
    

    Parameters
    ----------
    mhd_matrix : array
        Matrix of MHD between each pair of experimental pictures.
    min_samples : unsigned int
        Value of min_samples.
    epsilon : unsigned float
        Value of epsilon.

    Returns
    -------
    cluster_labels : list
        Index of the cluster for each image. The value is -1 if the point is considered as noise
    cluster_labels_filtered : list
        cluster_labels without the noise.
    unique_labels : unsigned int
        Number of clusters.
    noise_mask : boolean list
        Mask corresponding to the noise. True if the point is not noise.

    '''
    dbscan_clusterer = DBSCAN(eps=epsilon, min_samples=min_samples)
    # Perform clustering on the mhd matrix
    cluster_labels = dbscan_clusterer.fit_predict(mhd_matrix)
    # Ignoring the noise points
    noise_mask = cluster_labels != -1
    cluster_labels_filtered = cluster_labels[noise_mask]
    unique_labels = np.unique(cluster_labels_filtered)
    return cluster_labels, cluster_labels_filtered, unique_labels, noise_mask

def DBSCAN_distance_matrix(X_filtered, unique_labels, cluster_labels_filtered):
    '''
    

    Parameters
    ----------
    X_filtered : array
        Coordinates of the pictures in the visualization space.
    unique_labels : unsigned int
        Number of clusters.
    cluster_labels_filtered : list
        cluster_labels without the noise.

    Returns
    -------
    distance_matrix : array
        Euclidean distance between each pair of centroids in the visualization space.

    '''
    # Create a list to hold the coordinates of the centroids
    centroids_coordinates = []
    # Iterate over each unique label and compute centroids
    for label in unique_labels:
        # Select the points that belong to the current cluster
        cluster_points = X_filtered[cluster_labels_filtered == label]
        
        # Compute the centroid (mean point) of the cluster
        centroid = np.mean(cluster_points, axis=0)
        
        # Find the point in the cluster that is closest to the centroid
        closest_point = cluster_points[np.argmin(cdist(cluster_points, [centroid]))]
        
        # Add the coordinates of the closest point to the list
        centroids_coordinates.append(closest_point)
    
    # Convert the list to a numpy array for easier manipulation
    centroids_coordinates = np.array(centroids_coordinates)
    
    # Calculate the pairwise distances between centroids
    distance_matrix = cdist(centroids_coordinates, centroids_coordinates, 'euclidean')
    
    return distance_matrix

def DBSCAN_centroids(X_filtered, unique_labels, cluster_labels, cluster_labels_filtered):
    '''
    

    Parameters
    ----------
    X_filtered : array
        Coordinates of the pictures in the visualization space.
    unique_labels : unsigned int
        Number of clusters.
    cluster_labels : list
        Index of the cluster for each image. The value is -1 if the point is considered as noise
    cluster_labels_filtered : list
        cluster_labels without the noise.

    Returns
    -------
    centroids_indices : list
        Indices of the points chosen to be centroids of the clusters.

    '''
    # Create a list to hold the indices of the centroids in the original dataset
    centroids_indices = []
    
    # Iterate over each unique label and compute centroids
    for label in unique_labels:
        # Select the indices of points that belong to the current cluster from the original labels
        cluster_indices = np.where(cluster_labels == label)[0]
        
        # Apply the noise mask to get the corresponding UMAP points
        cluster_points = X_filtered[cluster_labels_filtered == label]
        
        # Compute the centroid (mean point) of the cluster
        centroid = np.mean(cluster_points, axis=0)
        
        # Find the index of the point in the cluster that is closest to the centroid
        closest_point_index = cluster_indices[np.argmin(cdist(cluster_points, [centroid]))]
        
        # Add the index of the closest point to the list
        centroids_indices.append(closest_point_index)
        
    return centroids_indices

def DBSCAN_pi(mhd_matrix, unique_labels, cluster_labels_filtered):
    '''
    

    Parameters
    ----------
    mhd_matrix : array
        Matrix of MHD between each pair of experimental pictures.
    unique_labels : unsigned int
        Number of clusters..
    cluster_labels_filtered : list
        cluster_labels without the noise.

    Returns
    -------
    pi : list
        Proportion of points in each cluster.

    '''
    # Getting pi
    total_points = len(unique_labels)
    pi = []
    for label in unique_labels:
        cluster_points_count = np.count_nonzero(cluster_labels_filtered == label)
        pi.append(cluster_points_count / total_points)
    return pi

def cumulative_sum_with_limit(dataset):
    '''
    Parameters
    ----------
    dataset : 3D numpy array
        A sequence of 2D binary images to be summed cumulatively.

    Returns
    -------
    cumulative_sum : 3D numpy array
        The cumulative sum of the images, with each pixel value limited to 1.
    '''

    # Initialize the cumulative sum to the first image in the dataset
    cumulative_sum = np.expand_dims(dataset[0,:,:], axis=0)

    # Iterate over the rest of the images in the dataset
    for image in dataset[1:,:,:]:
        # Compute the new cumulative sum
        new_cumulative_sum = cumulative_sum[-1,:,:] + image

        # Limit the pixel values to 1
        new_cumulative_sum = np.clip(new_cumulative_sum, 0, 1)

        # Append the new cumulative sum to the array
        cumulative_sum = np.concatenate((cumulative_sum, np.expand_dims(new_cumulative_sum, axis=0)), axis=0)

    return cumulative_sum
