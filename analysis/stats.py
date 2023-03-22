# -*- coding: utf-8 -*-
"""
Autoprot Analysis Functions.

@author: Wignand, Julian, Johannes

@documentation: Julian
"""
import os
from subprocess import run, PIPE
from importlib import resources
from typing import Union, Literal
from datetime import date

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pylab as plt
import matplotlib.colors as clrs
import pylab as pl
import seaborn as sns

from statsmodels.stats import multitest as mt
from scipy.stats import ttest_1samp, ttest_ind
from scipy.stats import zscore
from scipy.spatial import distance
from scipy import cluster as clst
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn import cluster as clstsklearn
from sklearn.decomposition import PCA

from operator import itemgetter

# noinspection PyUnresolvedReferences
from autoprot import visualization as vis
# noinspection PyUnresolvedReferences
from autoprot import r_helper
# noinspection PyUnresolvedReferences
from autoprot import preprocessing as pp

import warnings
import missingno as msn
from gprofiler import GProfiler

gp = GProfiler(
    user_agent="autoprot",
    return_dataframe=True)
RFUNCTIONS, R = r_helper.return_r_path()

# check where this is actually used and make it local
cmap = sns.diverging_palette(150, 275, s=80, l=55, n=9)


def edm(matrix_a, matrix_b):
    """
    Calculate an euclidean distance matrix between two matrices.

    See:  https://medium.com/swlh/euclidean-distance-matrix-4c3e1378d87f

    Parameters
    ----------
    matrix_a : np.ndarray
        Matrix 1.
    matrix_b : np.ndarray
        Matrix 2.

    Returns
    -------
    np.ndarray
        Distance matrix.

    """
    p1 = np.sum(matrix_a ** 2, axis=1)[:, np.newaxis]
    p2 = np.sum(matrix_b ** 2, axis=1)
    p3 = -2 * np.dot(matrix_a, matrix_b.T)
    return np.sqrt(p1 + p2 + p3)


def make_psm(seq, seq_len):
    # noinspection PyUnresolvedReferences
    """
    Generate a position score matrix for a set of sequences.

    Returns the percentage of each amino acid for each position that
    can be further normalized using a PSM of unrelated/background sequences.

    Parameters
    ----------
    seq : list of str
        list of sequences.
    seq_len : int
        Length of the peptide sequences.
        Must match to the list provided.

    Returns
    -------
    pd.Dataframe
        Dataframe holding the prevalence for every amino acid per position in
        the input sequences.

    Examples
    --------
    >>> autoprot.analysis.make_psm(['PEPTIDE', 'PEGTIDE', 'GGGGGGG'], 7)
              0         1         2         3         4         5         6
    G  0.333333  0.333333  0.666667  0.333333  0.333333  0.333333  0.333333
    P  0.666667  0.000000  0.333333  0.000000  0.000000  0.000000  0.000000
    matrix_a  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
    V  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
    L  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
    I  0.000000  0.000000  0.000000  0.000000  0.666667  0.000000  0.000000
    M  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
    C  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
    F  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
    Y  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
    W  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
    H  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
    K  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
    R  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
    Q  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
    N  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
    E  0.000000  0.666667  0.000000  0.000000  0.000000  0.000000  0.666667
    D  0.000000  0.000000  0.000000  0.000000  0.000000  0.666667  0.000000
    S  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
    T  0.000000  0.000000  0.000000  0.666667  0.000000  0.000000  0.000000

    """
    aa_dic = {
        'G': 0,
        'P': 0,
        'matrix_a': 0,
        'V': 0,
        'L': 0,
        'I': 0,
        'M': 0,
        'C': 0,
        'F': 0,
        'Y': 0,
        'W': 0,
        'H': 0,
        'K': 0,
        'R': 0,
        'Q': 0,
        'N': 0,
        'E': 0,
        'D': 0,
        'S': 0,
        'T': 0,
    }

    seq = [i for i in seq if len(i) == seq_len]
    seq_t = [''.join(s) for s in zip(*seq)]
    score_matrix = []
    for pos in seq_t:
        d = aa_dic.copy()
        for aa in pos:
            aa = aa.upper()
            if aa not in ['.', '-', '_', "dataframe"]:
                d[aa] += 1
        score_matrix.append(d)

    for pos in score_matrix:
        for k in pos.keys():
            pos[k] /= len(seq)

    # empty array -> (sequenceWindow, aa)
    m = np.empty((seq_len, 20))
    for i in range(m.shape[0]):
        x = list(score_matrix[i].values())
        m[i] = x

    m = pd.DataFrame(m, columns=list(aa_dic.keys()))

    return m.T
