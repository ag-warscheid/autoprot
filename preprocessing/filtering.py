# -*- coding: utf-8 -*-
"""
Autoprot Preprocessing Functions.

@author: Wignand, Julian, Johannes

@documentation: Julian
"""

import numpy as np
import pandas as pd
from importlib import resources
import re
import os
from subprocess import run, PIPE, STDOUT, CalledProcessError
from autoprot.decorators import report
from autoprot import r_helper
import requests
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from scipy.stats import pearsonr, spearmanr
from scipy import stats
from sklearn.metrics import auc
from urllib import parse
from ftplib import FTP
import warnings
from typing import Union

RFUNCTIONS, R = r_helper.return_r_path()


# =============================================================================
# Note: When using R functions provided column names might get changed
# Especially, do not use +,- or spaces in your column names. Maybe write decorator to
# validate proper column formatting and handle exceptions
# =============================================================================


@report
def cleaning(df, file="proteinGroups"):
    """
    Remove contaminant, reverse and identified by site only entries.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to clean up.
    file : str, optional
        Which file is provided in the dataframe.
        Possible values are "proteinGroups"; "Phospho (STY)", "evidence",  "modificationSpecificPeptides" or "peptides".
        The default is "proteinGroups".

    Returns
    -------
    df : pd.DataFrame
        The cleaned dataframe.

    Examples
    --------
    Cleaning can target different MQ txt files such as proteinGroups and
    phospho (STY) tables. The variables phos and prot are parsed MQ results tables.

    >>> prot_clean = autoprot.preprocessing.cleaning(prot, "proteinGroups")
    4910 rows before filter operation.
    4624 rows after filter operation.

    >>> phos_clean = autoprot.preprocessing.cleaning(phos, file = "Phospho (STY)")
    47936 rows before filter operation.
    47420 rows after filter operation.
    """
    df = df.copy()
    columns = df.columns
    if file == "proteinGroups":
        if "Potential contaminant" not in columns or "Reverse" not in columns or \
                "Only identified by site" not in columns:
            print("Is this data already cleaned?\nMandatory columns for cleaning not present in data!")
            print("Returning provided dataframe!")
            return df
        df = df[(df['Potential contaminant'].isnull() & df['Reverse'].isnull()) &
                df['Only identified by site'].isnull()]

        df.drop(['Potential contaminant', "Reverse", 'Only identified by site'], axis=1, inplace=True)

    elif file in ["Phospho (STY)", "evidence", "modificationSpecificPeptides", "peptides"]:
        if "Potential contaminant" not in columns or "Reverse" not in columns:
            print("Is this data already cleaned?\nMandatory columns for cleaning not present in data!")

            print("Returning provided dataframe!")
            return df
        df = df[(df['Potential contaminant'].isnull() & df['Reverse'].isnull())]
        df.drop(['Potential contaminant', "Reverse"], axis=1, inplace=True)

    return df


@report
def filter_loc_prob(df, thresh=.75):
    """
    Filter by localization probability.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to filter.
    thresh : int, optional
        Entries with localization probability below will be removed. The default is .75.

    Examples
    --------
    The .filter_loc_prob() function filters a Phospho (STY)Sites.txt file.
    You can provide the desired threshold with the *thresh* parameter.

    >>> phos_filter = autoprot.preprocessing.filter_loc_prob(phos, thresh=.75)
    47936 rows before filter operation.
    33311 rows after filter operation.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.

    """
    df = df.copy()  # make sure to keep the original dataframe unmodified
    if "Localization prob" not in df.columns:
        print("This dataframe has no 'Localization prob' column!")
        return True
    df = df[df["Localization prob"] >= thresh]
    return df


@report
def filter_seq_cov(df, thresh, cols=None):
    """
    Filter by sequence coverage.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to filter.
    thresh : int, optional
        Entries below that value will be excluded from the dataframe.
    cols : list of str, optional
        List of sequence coverage colnames.
        A row is excluded fromt the final dataframe the value in any of the provided columns is below the threshold.
        The default is None.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.

    """
    df = df.copy()  # make sure to keep the original dataframe unmodified
    if cols is not None:
        return df[(df[cols] >= thresh).all(1)]
    return df[df["Sequence coverage [%]"] >= thresh]


@report
def filter_vv(df, groups, n=2, valid_values=True):
    r"""
    Filter dataframe for minimum number of valid values.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to be filtered.
    groups : list of lists of str
        Lists of colnames of the experimental groups.
        Each group is filtered for at least n vv.
    n : int, optional
        Minimum amount of valid values. The default is 2.
    valid_values : bool, optional
        True for minimum amount of valid values; False for maximum amount of missing values. The default is True.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.
    set (optional)
        Set of indices after filtering.

    Examples
    --------
    The function filterVv() filters the dataframe for a minimum number of valid values per group.
    You have to provide the data, the groups as well as the desired number of valid values.
    If the specified n is not reached in one or more groups the respective row is dropped.
    Setting the keyword vv=False inverts the logic and filters the dataframe for a maximum number of missing values.

    >>> protRatio = prot.filter(regex="Ratio .\/. normalized")
    >>> protLog = autoprot.preprocessing.log(prot, protRatio, base=2)

    >>> a = ['log2_Ratio H/M normalized BC18_1','log2_Ratio M/L normalized BC18_2','log2_Ratio H/M normalized BC18_3',
    ...                 'log2_Ratio H/L normalized BC36_1','log2_Ratio H/M normalized BC36_2','log2_Ratio M/L normalized BC36_2']
    >>> b = ["log2_Ratio H/L normalized BC18_1","log2_Ratio H/M normalized BC18_2","log2_Ratio H/L normalized BC18_3",
    ...                 "log2_Ratio M/L normalized BC36_1", "log2_Ratio H/L normalized BC36_2","log2_Ratio H/M normalized BC36_2"]
    >>> c = ["log2_Ratio M/L normalized BC18_1","log2_Ratio H/L normalized BC18_2","log2_Ratio M/L normalized BC18_3",
    ...               "log2_Ratio H/M normalized BC36_1","log2_Ratio M/L normalized BC36_2","log2_Ratio H/L normalized BC36_2"]
    >>> protFilter = autoprot.preprocessing.filter_vv(protLog, groups=[a,b,c], n=3)
    4910 rows before filter operation.
    2674 rows after filter operation.
    """
    df = df.copy()  # make sure to keep the original dataframe unmodified

    if valid_values:
        idxs = [set(df[df[group].notnull().sum(1) >= n].index) for group in groups]
        # idxs = [set(df[(len(group)-df[group].isnull().sum(1)) >= n].index) for\
        #        group in groups]
    else:
        idxs = [set(df[df[group].isnull().sum(1) <= n].index) for group in groups]

    # indices that are valid in all groups
    idx = list(set.intersection(*idxs))
    df = df.loc[idx]

    return df


@report
def remove_non_quant(df, cols):
    r"""
    Remove entries without quantitative data.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to filter.
    cols : list of str
        cols to be evaluated for missingness.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.

    Examples
    --------
    >>> df = pd.DataFrame({'a':[1,2,np.nan,4], 'b':[4,0,np.nan,1], 'c':[None, None, 1, 1]})
    >>> autoprot.preprocessing.remove_non_quant(df, cols=['a', 'b'])
    4 rows before filter operation.
    3 rows after filter operation.
         a    b    c
    0  1.0  4.0  NaN
    1  2.0  0.0  NaN
    3  4.0  1.0  1.0

    Rows are only removed if the all values in the specified columns are NaN.

    >>> autoprot.preprocessing.remove_non_quant(df, cols=['b', 'c'])
    4 rows before filter operation.
    4 rows after filter operation.
         a    b    c
    0  1.0  4.0  NaN
    1  2.0  0.0  NaN
    2  NaN  NaN  1.0
    3  4.0  1.0  1.0

    Example with real data.

    >>> phosRatio = phos.filter(regex="^Ratio .\/.( | normalized )R.___").columns
    >>> phosQuant = autoprot.preprocessing.remove_non_quant(phosLog, phosRatio)
    47936 rows before filter operation.
    39398 rows after filter operation.
    """
    df = df[~(df[cols].isnull().all(1))]
    return df

