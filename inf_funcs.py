from collections import Counter
import math
import pandas as pd
import numpy as np
from scipy.stats import entropy
import numpy as np
import pyinform
import itertools 


xs = [0,1,1,1,1,0,0,0,0]
ys = [0,0,1,1,1,1,0,0,0]

def compute_pmf(data):
    """
    Compute the probability mass function (PMF) for a given random variable.

    Parameters:
    data (list): List representing the random variable.

    Returns:
    pmf (dict): PMF dictionary where keys are unique values and values are their probabilities.
    """
    total_count = len(data)
    counts = Counter(data)
    pmf = np.array([count / total_count for value, count in counts.items()])
    return pmf



def entropy_single(random_variable):
    
    PMF = compute_pmf(random_variable)
    return entropy(PMF, base=2)
 
def compute_joint_pmf(X, Y):
    """
    Compute the joint probability mass function (PMF) for two random variables X and Y.

    Parameters:
    X (list): Values of random variable X.
    Y (list): Values of random variable Y.

    Returns:
    joint_pmf (numpy.ndarray): Joint PMF array.
    """
    # Unique values and their counts for X and Y
    unique_values_X, counts_X = np.unique(X, return_counts=True)
    unique_values_Y, counts_Y = np.unique(Y, return_counts=True)
    
    # Initialize joint PMF array
    joint_pmf = np.zeros((len(unique_values_X), len(unique_values_Y)))

    # Compute joint PMF
    for i, x in enumerate(unique_values_X):
        for j, y in enumerate(unique_values_Y):
            joint_count = np.sum((X == x) & (Y == y))
            joint_pmf[i, j] = joint_count / len(X)

    return joint_pmf

def entropy_double(X,Y):
    h_X = entropy_single(X)
    h_Y = entropy_single(Y)
    return h_X + h_Y

def compute_pmf_three(X, Y, Z):
    occurrences = Counter(zip(X, Y, Z))
    total_count = len(X)  # Assuming X, Y, and Z have the same length
    pmf = {xyz: count / total_count for xyz, count in occurrences.items()}
    return pmf

def entropy_triple(X,Y,Z):
    entropy_val = 0
    pmf_dict = compute_pmf_three(X,Y,Z)
    for p in pmf_dict.values():
        entropy_val -= p * math.log2(p)
    return entropy_val
    


def total_mutual_info(X,Y,Z):
    H_X_Y = entropy_double(X,Y)
    H_Z = entropy_single(Z)
    H_X_Y_Z = entropy_triple(X,Y,Z)

    return H_X_Y + H_Z - H_X_Y_Z


def pidMMI(X, Y, Z,_local):
     # Calculate Transfer Entropy

    #Compute the transfer entopy for both source X to Y and for source j to k
    MI_x_z = pyinform.mutual_info(X, Z)
    MI_y_z = pyinform.mutual_info(Y, Z)

    print(f"MI i -> k RESULT: {MI_x_z}")
    print(f"MI j -> k RESULT: {MI_y_z}")
    total_MI = total_mutual_info(X, Y , Z)
    


    RED = min(MI_x_z, MI_y_z)
    UN_x  = MI_x_z - RED
    UN_y = MI_y_z - RED

    
    SYN = total_MI - (UN_x+UN_y) - RED

    print(f"RED: {RED}\t SYN: {SYN}\t UN_x: {UN_x}\t UN_j: {UN_y}\n")
    print(f"I(X,Y;Z) = {UN_x + UN_y + RED + SYN}")
    return RED, SYN, UN_x, UN_y