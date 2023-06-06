import torch
import math
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.fftpack import dctn

def CFDP(fm_slice, alpha, ours=True):
    '''
    Input:
        - fm_slice: a matrix of size [h, w], which is a slice of a given feature 
        map in spatial domain.
        
        - alpha: a hyperparameter for balancing the spatial and frequency 
        domain contributions
    
    Output:
        - score: a net scoring metric for the channel of interst to indicate the 
        level of information it contributes
    '''
    fm_slice = fm_slice.cpu().numpy()
    Matrix = np.zeros((fm_slice.shape))
    Height = len(Matrix[:, :])
    ratio = 2
    NumBlocks = Height//ratio
    
    Sparsity = []
    for subblock_x in range(NumBlocks):
        for subblock_y in range(NumBlocks):
            block = gaussian_filter(dctn( \
                fm_slice[subblock_x*ratio:subblock_x*ratio+ratio, \
                         subblock_y*ratio:subblock_y*ratio+ratio], \
                norm = 'ortho'), sigma=10)
            Matrix[subblock_x*ratio:subblock_x*ratio+ratio, \
                   subblock_y*ratio:subblock_y*ratio+ratio] = block
            
    if NumBlocks == 0:
        Matrix = gaussian_filter(dctn(fm_slice, norm = 'ortho'), sigma=10)
        Sparsity = (Matrix > np.mean(Matrix)).sum()/(fm_slice.shape[0]**2)

        elem1 = np.linalg.norm(Matrix) * Sparsity
        elem2 = alpha*np.linalg.norm(fm_slice)
        result = elem1 + elem2

    else:
        Sparsity = (Matrix > np.mean(Matrix)).sum()/(fm_slice.shape[0]**2)
        
        elem1 = np.linalg.norm(Matrix) * Sparsity
        elem2 = alpha*np.linalg.norm(fm_slice)
        result = elem1 + elem2

    return result
    