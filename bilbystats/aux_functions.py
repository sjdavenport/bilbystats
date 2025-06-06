#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A set of auxiliary functions
"""
import numpy as np
import sys
import time
import tiktoken


from importlib import resources
from dotenv import load_dotenv
import os

# Use importlib to get a temporary path to the installed .env file
env_path = resources.files("bilbystats.defaults").joinpath("model_costs.env")

# Load environment variables from the .env file
load_dotenv(dotenv_path=env_path)

# Global variable to store the start time
_start_time = None


def tic():
    """Start timer"""
    global _start_time
    _start_time = time.time()


def toc():
    """Print and return elapsed time since last tic()"""
    if _start_time is None:
        print("tic() has not been called yet.")
        return None
    elapsed_time = time.time() - _start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")
    return elapsed_time


def choose_random_indices(n1, n2, n3, index1, index2, index3):
    """
    Randomly select indices from three separate index arrays.

    This function samples a specified number of unique indices from three given 
    index arrays without replacement. The selected indices from each array are 
    then concatenated into a single combined array.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    n1 : int
        Number of indices to select from `index1`.
    n2 : int
        Number of indices to select from `index2`.
    n3 : int
        Number of indices to select from `index3`.
    index1 : array-like, shape (n_samples1,)
        Array of indices to sample from for the first selection.
    index2 : array-like, shape (n_samples2,)
        Array of indices to sample from for the second selection.
    index3 : array-like, shape (n_samples3,)
        Array of indices to sample from for the third selection.

    ---------------------------------------------------------------------------
    OUTPUT:
    combined_indices : ndarray, shape (n1 + n2 + n3,)
        Concatenated array of all selected indices from the three input arrays.
    random_indices1 : ndarray, shape (n1,)
        Randomly selected indices from `index1`.
    random_indices2 : ndarray, shape (n2,)
        Randomly selected indices from `index2`.
    random_indices3 : ndarray, shape (n3,)
        Randomly selected indices from `index3`.
    ---------------------------------------------------------------------------
    AUTHORS: Samuel Davenport
    ---------------------------------------------------------------------------
    """
    random_indices1 = np.random.choice(index1, size=n1, replace=False)
    random_indices2 = np.random.choice(index2, size=n2, replace=False)
    random_indices3 = np.random.choice(index3, size=n3, replace=False)
    combined_indices = np.concatenate(
        [random_indices1, random_indices2, random_indices3])
    # np.random.shuffle(combined_indices)
    return combined_indices, random_indices1, random_indices2, random_indices3


def modul(iterand, niterand=100):
    """
    Print the value of `iterand` if it is divisible by `niterand`.

    This function checks whether `iterand` is evenly divisible by `niterand`.
    If the condition is met, it prints the value of `iterand`.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    iterand : int
        The number to be tested for divisibility.
    niterand : int, optional (default=100)
        The number to divide by.

    ---------------------------------------------------------------------------
    OUTPUT:
    None

    ---------------------------------------------------------------------------
    AUTHORS: Samuel Davenport
    ---------------------------------------------------------------------------
    """
    if iterand % niterand == 0:
        print(iterand)


def loader(I, totalI):
    """
    Display a progress bar in the console.

    This function updates a text-based progress bar in the console based on the 
    current progress `I` out of the total iterations `totalI`. It overwrites the 
    same line to give a real-time progress update.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    I : int
        The current iteration or progress count.
    totalI : int
        The total number of iterations or steps to complete.

    ---------------------------------------------------------------------------
    OUTPUT:
    None
        This function only updates the console output and does not return any value.
    ---------------------------------------------------------------------------
    AUTHORS: Samuel Davenport
    ---------------------------------------------------------------------------
    """
    progress = 100*I/totalI
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" %
                     ('='*int(np.floor(progress/5)), progress))
    sys.stdout.flush()


def model_costs(input_tokens, output_tokens, model_name, ndocs):
    """
    Estimate the cost of a language model API call based on token usage.

    This function calculates the number of input and output tokens using the 
    specified model's tokenizer and computes the associated cost using 
    environment variables that specify pricing per million tokens. It returns 
    a breakdown of token counts and cost estimates.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    input_tokens : str
        The raw input text to be tokenized and used for cost estimation.
    output_tokens : str
        The raw output text to be tokenized and used for cost estimation.
    model_name : str
        The name of the language model (e.g., 'gpt-4') used to fetch the appropriate tokenizer and pricing.
    ndocs : int
        The number of documents processed (currently unused in cost calculation but reserved for future use).

    ---------------------------------------------------------------------------
    OUTPUT:
    dict : dict
        A dictionary containing:
        - 'n_input_tokens': int
              Number of tokens in the input text.
        - 'm_output_tokens': int
              Number of tokens in the output text (note: key name appears to be a typo).
        - 'input_cost': float
              Estimated input cost in currency units.
        - 'output_cost': float
              Estimated output cost in currency units.
        - 'total_cost': float
              Combined cost of input and output tokens.

    ---------------------------------------------------------------------------
    AUTHORS: Samuel Davenport
    ---------------------------------------------------------------------------
    """
    tokenizer = tiktoken.encoding_for_model(model_name)

    n_input_tokens = len(tokenizer.encode(input_tokens))
    n_output_tokens = len(tokenizer.encode(output_tokens))

    input_price_per_million = os.getenv(model_name + "_input")
    output_price_per_million = os.getenv(model_name + "_output")
    input_cost = input_price_per_million * (n_input_tokens / 1_000_000)
    output_cost = output_price_per_million * (n_output_tokens / 1_000_000)

    total_cost = input_cost + output_cost

    return {
        "n_input_tokens": n_input_tokens,
        "m_output_tokens": n_output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost
    }
