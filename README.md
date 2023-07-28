# Modelling the behaviour of braided rivers by means of Image Quilting

## Description

In this script, we put forward a Bayesian approach for fine-tuning input parameters of stochastic models applied to morphodynamic systems. This process leverages time series image data derived from fieldwork, as well as from numerical and laboratory experiments. The approach involves the creation of artificial time series of images using the stochastic model, and the rejection of those series which do not match key morphodynamic statistics of the data sets available. The fine-tuned stochastic model enables us to measure both the spatial and temporal uncertainties related to the progression of the morphodynamic systems under study.

We used in Python the packages `ImageQuilting.jl` and `Geostats.jl` that were developped by Julio Hoffimann. 

## Installation

To make this program running, you need :
1. to download the `Arthur.yml` environnement, the five jupyter notebooks and the four Python files from this Github. Take care to put the Python files in the same folder as as the notebooks.
2. to install Anaconda or Mamba on your computer.
   - for Anaconda, follow this [link](https://docs.anaconda.com/free/anaconda/install/index.html)
   - for Mamba, follow this [link](https://mamba.readthedocs.io/en/latest/installation.html)
3. to install the `Arthur.yml` environnement in Python. Open the Anaconda or the Mamba prompt and write the command :
```bash
conda env create -f Arthur.yml
```
or 
```bash
mamba env create -f Arthur.yml
```
4. to activate the environnement using the command :
```bash
conda activate Arthur
```
or 
```bash
mamba activate Arthur
```
5. to open Jupyter lab in the environnement. Just write :
 ```bash
jupyter lab
```
after you activated the environnement.

6. to add all the necessary packages of Julia. Open the notebook `installPackages.ipynb` in the jupyter lab window and run it once.

## Utilisation

The process is very easy to follow. You just have to open the four jupyter notebooks in the right order and follow the instructions. 
Here is the right order of the notebooks :
1. `Clustering.ipynb`
2. `IQParameters.ipynb`
3. `Generation.ipynb`
4. `StatisticalValidation.ipynb`


## References

This script tries to replicate Julio Hoffimann's 2019 [paper](https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2019JF005245). We also include some extra tests and visualization. Hoffimann's work took inspiration from Céline Scheidt's [article](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2016JF003922#). 
