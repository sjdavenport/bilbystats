#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A set of auxiliary functions
"""
import numpy as np

def choose_random_indices(n, index1, index2, index3):
    """
    Select random indices from three input arrays and return a shuffled combination.

    This function randomly samples `n` unique indices (without replacement) from each 
    of the three input index arrays. The selected indices are then concatenated and 
    shuffled to produce a single output array.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    n : int
        Number of indices to randomly select from each input array.
    index1 : array-like
        First array of candidate indices to sample from.
    index2 : array-like
        Second array of candidate indices to sample from.
    index3 : array-like
        Third array of candidate indices to sample from.

    ---------------------------------------------------------------------------
    OUTPUT:
    combined_indices : ndarray
        A 1D NumPy array containing 3 * n shuffled indices selected from the 
        input arrays.

    ---------------------------------------------------------------------------
    AUTHORS: Samuel Davenport
    ---------------------------------------------------------------------------
    """
    random_indices1 = np.random.choice(index1, size=n, replace=False)
    random_indices2 = np.random.choice(index2, size=n, replace=False)
    random_indices3 = np.random.choice(index3, size=n, replace=False)
    combined_indices = np.concatenate([random_indices1, random_indices2, random_indices3])
    np.random.shuffle(combined_indices)
    return combined_indices