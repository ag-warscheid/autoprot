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


def ttest(df, reps, cond="", return_fc=True, adjust_p_vals=True, alternative='two-sided', logged=True):
    # noinspection PyUnresolvedReferences
    """
    Perform one or two sample ttest.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    reps : list of str or list of lists of str
        The replicates to be included in the statistical test.
        Either a list of the replicates or a list containing two list with the
        respective replicates.
    cond : str, optional
        The name of the condition.
        This is used for naming the returned results.
        The default is "".
    return_fc : bool, optional
        Whether to calculate the fold-change of the provided data.
        The processing of the fold-change can be controlled by the logged
        switch.
        The default is True.
    adjust_p_vals : bool, optional
        Whether to adjust P-values. The default is True.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.
        The following options are available (default is 'two-sided'):

        * 'two-sided': the mean of the underlying distribution of the sample
          is different from the given population mean (`popmean`)
        * 'less': the mean of the underlying distribution of the sample is
          less than the given population mean (`popmean`)
        * 'greater': the mean of the underlying distribution of the sample is
          greater than the given population mean (`popmean`)
    logged: bool, optional
        Set to True if input values are log-transformed (or VSN normalised).
        This returns the difference between values as log_fc, otherwise
        values are log2 transformed to gain log_fc.
        Default is true.


    Returns
    -------
    df : pd.DataFrame
        Input dataframe with additional cols.

    Examples
    --------
    >>> twitchVsmild = ['log2_Ratio H/M normalized BC18_1','log2_Ratio M/L normalized BC18_2',
    ...                 'log2_Ratio H/M normalized BC18_3',
    ...                 'log2_Ratio H/L normalized BC36_1','log2_Ratio H/M normalized BC36_2',
    ...                 'log2_Ratio M/L normalized BC36_2']
    >>> protRatio = prot.filter(regex="Ratio .\/. normalized")
    >>> protLog = autoprot.preprocessing.log(prot, protRatio, base=2)
    >>> prot_tt = autoprot.analysis.ttest(df=protLog, reps=twitchVsmild, cond="_TvM", return_fc=True,
    ... adjust_p_vals=True)
    >>> prot_tt["pValue_TvM"].hist(bins=50)
    >>> plt.show()

    .. plot::
        :context: close-figs

        import autoprot.analysis as ana
        import autoprot.preprocessing as pp
        import pandas as pd
        twitchVsmild = ['log2_Ratio H/M normalized BC18_1','log2_Ratio M/L normalized BC18_2',
                        'log2_Ratio H/M normalized BC18_3',
                        'log2_Ratio H/L normalized BC36_1','log2_Ratio H/M normalized BC36_2',
                        'log2_Ratio M/L normalized BC36_2']
        prot = pd.read_csv("_static/testdata/03_proteinGroups.zip", sep='\\t', low_memory=False)
        protRatio = prot.filter(regex="Ratio .\/. normalized")
        protLog = pp.log(prot, protRatio, base=2)
        prot_tt = ana.ttest(df=protLog, reps=twitchVsmild, cond="_TvM", return_fc=True, adjust_p_vals=True)
        prot_tt["pValue_TvM"].hist(bins=50)
        plt.show()

    >>> dataframe = pd.DataFrame({"a1":np.random.normal(loc=0, size=4000),
    ...                           "a2":np.random.normal(loc=0, size=4000),
    ...                           "a3":np.random.normal(loc=0, size=4000),
    ...                           "b1":np.random.normal(loc=0.5, size=4000),
    ...                           "b2":np.random.normal(loc=0.5, size=4000),
    ...                           "b3":np.random.normal(loc=0.5, size=4000),})
    >>> autoprot.analysis.ttest(df=dataframe, reps=[["a1","a2", "a3"],["b1","b2", "b3"]])["pValue"].hist(bins=50)
    >>> plt.show()

    .. plot::
        :context: close-figs

        import autoprot.analysis as ana
        import pandas as pd
        df = pd.DataFrame({"a1":np.random.normal(loc=0, size=4000),
                  "a2":np.random.normal(loc=0, size=4000),
                  "a3":np.random.normal(loc=0, size=4000),
                  "b1":np.random.normal(loc=0.5, size=4000),
                  "b2":np.random.normal(loc=0.5, size=4000),
                  "b3":np.random.normal(loc=0.5, size=4000),})
        ana.ttest(df=df, reps=[["a1","a2", "a3"],["b1","b2", "b3"]])["pValue"].hist(bins=50)
        plt.show()

    """

    def one_samp_ttest(x):
        # nan-containing/masked inputs with nan_policy='omit' are currently not supported by one-sided alternatives.
        x = x[~np.isnan(x)]
        return np.ma.filled(ttest_1samp(x, nan_policy="raise", alternative=alternative, popmean=0)[1], np.nan)

    def two_samp_ttest(x):
        return np.ma.filled(
            ttest_ind(x[: len(reps[0])], x[len(reps[0]):], alternative=alternative, nan_policy="omit")[1], np.nan)

    if isinstance(reps[0], list) and len(reps) == 2:
        print("Performing two-sample t-Test")
        df[f"pValue{cond}"] = df[reps[0] + reps[1]].apply(lambda x: two_samp_ttest(x), 1).astype(float)

        df[f"score{cond}"] = -np.log10(df[f"pValue{cond}"])
        if return_fc:
            if logged:
                df[f"logFC{cond}"] = pd.DataFrame(df[reps[0]].values - df[reps[1]].values).mean(1).values
            else:
                df[f"logFC{cond}"] = np.log2(pd.DataFrame(df[reps[0]].values / df[reps[1]].values).mean(1)).values

    else:
        print("Performing one-sample t-Test")
        df[f"pValue{cond}"] = df[reps].apply(lambda x: one_samp_ttest(x), 1).astype(float)

        df[f"score{cond}"] = -np.log10(df[f"pValue{cond}"])
        if return_fc:
            df[f"logFC{cond}"] = df[reps].mean(1) if logged else np.log2(df[reps].mean(1))
    if adjust_p_vals:
        adjust_p(df, f"pValue{cond}")
    return df


def adjust_p(df, p_col, method="fdr_bh"):
    # noinspection PyUnresolvedReferences
    r"""
    Use statsmodels.multitest on dataframes.

    Note: when nan in p-value this function will return only nan.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    p_col : str
        column containing p-values for correction.
    method : str, optional
        'b': 'Bonferroni',
        's': 'Sidak',
        'h': 'Holm',
        'hs': 'Holm-Sidak',
        'sh': 'Simes-Hochberg',
        'ho': 'Hommel',
        'fdr_bh': 'FDR Benjamini-Hochberg',
        'fdr_by': 'FDR Benjamini-Yekutieli',
        'fdr_tsbh': 'FDR 2-stage Benjamini-Hochberg',
        'fdr_tsbky': 'FDR 2-stage Benjamini-Krieger-Yekutieli',
        'fdr_gbs': 'FDR adaptive Gavrilov-Benjamini-Sarkar'
        The default is "fdr_bh".

    Returns
    -------
    df : pd.DataFrame
        The input dataframe with adjusted p-values.
        The dataframe will have a column named "adj.{pCol}"

    Examples
    --------
    >>> twitchVsmild = ['log2_Ratio H/M normalized BC18_1','log2_Ratio M/L normalized BC18_2',
    ...                 'log2_Ratio H/M normalized BC18_3',
    ...                 'log2_Ratio H/L normalized BC36_1','log2_Ratio H/M normalized BC36_2',
    ...                 'log2_Ratio M/L normalized BC36_2']
    >>> prot = pd.read_csv("_static/testdata/03_proteinGroups.zip", sep='\t', low_memory=False)
    >>> protRatio = prot.filter(regex="Ratio .\/. normalized")
    >>> protLog = pp.log(prot, protRatio, base=2)
    >>> prot_tt = ana.ttest(df=protLog, reps=twitchVsmild, cond="TvM", mean=True, adjust_p_vals=False)
    >>> prot_tt_adj = ana.adjust_p(prot_tt, p_col="pValue_TvM")
    >>> prot_tt_adj.filter(regex='pValue').head()
       pValue_TvM  adj.pValue_TvM
    0         NaN             NaN
    1    0.947334        0.966514
    2         NaN             NaN
    3         NaN             NaN
    4    0.031292        0.206977
    """
    # indices of rows containing values
    idx = df[df[p_col].notnull()].index
    # init new col with for adjusted p-values
    df[f"adj.{p_col}"] = np.nan
    # apply correction for selected rows
    df.loc[idx, f"adj.{p_col}"] = mt.multipletests(df[p_col].loc[idx], method=method)[1]
    return df


def cohen_d(df, group1, group2):
    """
    Calculate Cohen's d effect size for two groups.

    Parameters
    ----------
    df : pd.Dataframe
        Input dataframe.
    group1 : str
        Colname for group1.
    group2 : str
        Colname for group2.

    Returns
    -------
    df : pd.Dataframe
        Input dataframe with a new column "cohenD".

    Notes
    -----
    Cohen's d is defined as the difference between two means divided by a standard deviation for the data.
    Note that the pooled standard deviation here is calculated differently than
    originally proposed by Cohen.

    References
    ----------
    [1] Cohen, Jacob (1988). Statistical Power Analysis for the Behavioral Sciences. Routledge.
    [2] https://www.doi.org/10.22237/jmasm/1257035100

    """
    mean1 = df[group1].mean(1).values
    std1 = df[group1].std(1).values
    mean2 = df[group2].mean(1).values
    std2 = df[group2].std(1).values
    # TODO: the pooled sd here is calculated omitting the sample sizes n
    # This is not exactly what was proposed for cohens d: https://en.wikipedia.org/wiki/Effect_size
    sd_pooled = np.sqrt((std1 ** 2 + std2 ** 2) / 2)
    df["cohenD"] = (abs(mean1 - mean2)) / sd_pooled
    return df


def limma(df, reps, cond="", custom_design=None, coef=None, print_r=False):
    # sourcery skip: extract-method, inline-immediately-returned-variable
    # noinspection PyUnresolvedReferences
    r"""
    Perform moderated ttest as implemented from R LIMMA.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    reps : list of lists of str
        Column names of the replicates.
        Common replicates are grouped together in a single list.
    cond : str, optional
        Term to append to the newly generated colnames.
        The default is "".
    custom_design : str, optional
        Path to custom design file.
        The default is None.
    coef : str, optional
        The coefficients serving as the basis for calculating p-vlaues and fold-changes from the eBayes.
        Must refer to design matrix colnames. If no custom design is specified the default coeficient is "coef".
        Differences are indicated e.g. by "CondA-CondB". See https://rdrr.io/bioc/limma/man/toptable.html for details.
        The default is None.
    print_r : bool, optional
        Whether to print the R output.
        The default is False.

    Returns
    -------
    df : pd.DataFrame
        The input dataframe with additional columns.

    Notes
    -----
    A custom design matriox has rows corresponding to arrays and columns to coefficients to be estimated
    can be provided using customDesign.
    If customDesign is the unit vector meaning that the arrays are treated as replicates.
    See: https://www.rdocumentation.org/packages/limma/versions/3.28.14/topics/lmFit

    Examples
    --------
    >>> data = pd.DataFrame({"a1":np.random.normal(loc=0, size=4000),
    ...                      "a2":np.random.normal(loc=0, size=4000),
    ...                      "a3":np.random.normal(loc=0, size=4000),
    ...                      "b1":np.random.normal(loc=0.5, size=4000),
    ...                      "b2":np.random.normal(loc=0.5, size=4000),
    ...                      "b3":np.random.normal(loc=0.5, size=4000),})
    >>> testRes = ana.limma(df=data, reps=[["a1","a2", "a3"],["b1","b2", "b3"]], cond="_test")
    >>> testRes["P.Value_test"].hist()

    .. plot::
        :context: close-figs

        import autoprot.analysis as ana

        df = pd.DataFrame({"a1":np.random.normal(loc=0, size=4000),
                           "a2":np.random.normal(loc=0, size=4000),
                           "a3":np.random.normal(loc=0, size=4000),
                           "b1":np.random.normal(loc=0.5, size=4000),
                           "b2":np.random.normal(loc=0.5, size=4000),
                           "b3":np.random.normal(loc=0.5, size=4000),})
        testRes = ana.limma(df, reps=[["a1","a2", "a3"],["b1","b2", "b3"]], cond="_test")
        testRes["P.Value_test"].hist()
        plt.show()

    """
    # TODO: better handle coefficient extraction in R
    df = df.copy()
    d = os.getcwd()
    data_loc = d + "/input.csv"
    output_loc = d + "/output.csv"

    # limma handles fold-change calculation opposite to all other autoprot tools
    # this changes the order for function consistency
    if isinstance(reps[0], list) and len(reps) == 2:
        reps = reps[::-1]

    if "UID" not in df.columns:
        df["UID"] = range(1, df.shape[0] + 1)

    # flatten in case of two sample
    pp.to_csv(df[["UID"] + list(pl.flatten(reps))], data_loc)

    # Normally no custom_design is provided
    if custom_design is None:
        design_loc = d + "/design.csv"
        # if two lists are provided with reps, this likely is a twoSample test
        if isinstance(reps[0], list) and len(reps) == 2:
            print("LIMMA: Assuming a two sample test with:")
            print("Sample 1: {}".format(', '.join(['\n\t' + x for x in reps[0]])))
            print("Sample 2: {}".format(', '.join(['\n\t' + x for x in reps[1]])))
            test = "twoSample"
            design = pd.DataFrame({"Intercept": [1] * (len(reps[0]) + len(reps[1])),
                                   "coef": [0] * len(reps[0]) + [1] * len(reps[1])})
            # =============================================================================
            #             creates a design matrix such as
            #                 Intercept  coef
            #              0          1     0
            #              1          1     0
            #              2          1     0
            #              3          1     1
            #              4          1     1
            #              5          1     1
            # =============================================================================
            print("Using design matrix:\n")
            print(design.to_markdown())

            # save the design for R to read
            pp.to_csv(design, design_loc)
        else:
            test = "oneSample"
            print("LIMMA: Assuming a one sample test")
            # The R function will generate a design matrix corresponding to
            # ones
    else:
        print("LIMMA: Assuming a custom design test with:")
        print(f"Design specified at {custom_design}")
        print("Columns: {}".format('\n\t'.join(list(pl.flatten(reps)))))

        design = pd.read_csv(custom_design, sep='\t')
        print("Using design matrix:\n")
        print(design.to_markdown())

        test = "custom"
        design_loc = custom_design

    command = [R, '--vanilla', RFUNCTIONS, "limma", data_loc, output_loc, test, design_loc, coef or ""]

    p = run(command,
            stdout=PIPE,
            stderr=PIPE,
            universal_newlines=True)

    if print_r:
        print(p.stdout)

    res = pp.read_csv(output_loc)
    res.columns = [i + cond if i != "UID" else i for i in res.columns]
    # this keeps the index of the original df in the returned df
    df = df.reset_index().merge(res, on="UID").set_index('index')

    os.remove(data_loc)
    os.remove(output_loc)
    if custom_design is None and isinstance(reps[0], list) and len(reps) == 2:
        os.remove(design_loc)

    return df


def rank_prod(df, reps, cond="", print_r=False, correct_fc=True):
    """
    Perform RankProd test as in R RankProd package.

    At the moment one sample test only.
    Test for up and downregulated genes separatly therefore returns two p values.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    reps : list of lists of str
        Column names of the replicates.
        Common replicates are grouped together in a single list.
    cond : str, optional
        Term to append to the newly generated colnames.
        The default is "".
    print_r : bool, optional
        Whether to print the R output.
        The default is False.
    correct_fc : bool, optional
        The rankProd package does not calculate fold-changes for rows with
        missing values (p Values are calculated). If correct_fc is False
        the original fold changes from rankProd are return, else fold
        changes are calculated for all values after ignoring NaNs.

    Returns
    -------
    df : pd.DataFrame
        Input dataframe with additional columns from RankProd.


    Notes
    -----
    The adjusted p-values returned from the R backend are the percentage of
    false postives calculated by the rankProd package. This is akin to corrected
    p values, but care should be taken to name these values accordingly.

    """

    d = os.getcwd()
    data_loc = d + "/input.csv"
    output_loc = d + "/output.csv"

    if "UID" not in df.columns:
        df["UID"] = range(1, df.shape[0] + 1)

    if isinstance(reps[0], list) and len(reps) == 2:
        class_labels = [0, ] * len(reps[0]) + [1, ] * len(reps[1])
        print("rankProd: Assuming a two sample test with:")
        print("Sample 1: {}".format(', '.join(['\n\t' + x for x in reps[0]])))
        print("Sample 2: {}".format(', '.join(['\n\t' + x for x in reps[1]])))
        print(f"Class labels: {', '.join([str(x) for x in class_labels])}")

    else:
        print("rankProd: Assuming a one sample test")
        class_labels = [1, ] * len(reps)

    # flatten in case of two sample
    pp.to_csv(df[["UID"] + list(pl.flatten(reps))], data_loc)

    command = [R, '--vanilla',
               RFUNCTIONS,  # script location
               "rankProd",  # functionName
               data_loc,  # data location
               output_loc,  # output file,
               ','.join([str(x) for x in class_labels]),
               ]

    p = run(command,
            stdout=PIPE,
            stderr=PIPE,
            universal_newlines=True)

    if print_r:
        print(p.stdout)

    res = pp.read_csv(output_loc)
    res.columns = [i + cond if i != "UID" else i for i in res.columns]
    df = df.reset_index().merge(res, on="UID").set_index('index')

    if correct_fc:
        if isinstance(reps[0], list) and len(reps) == 2:
            df['log_fc' + cond] = df[reps[0]].mean(axis=1, skipna=True) - df[reps[1]].mean(axis=1, skipna=True)
        else:
            df['log_fc' + cond] = df[reps].mean(axis=1)

    os.remove(data_loc)
    os.remove(output_loc)

    return df
