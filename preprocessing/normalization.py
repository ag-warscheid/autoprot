# -*- coding: utf-8 -*-
"""
Autoprot Preprocessing Functions.

@author: Wignand, Julian, Johannes

@documentation: Julian
"""

import numpy as np
import pandas as pd
import os
from subprocess import run, PIPE, CalledProcessError
from typing import Union
from .. import r_helper
from .. import preprocessing as pp

RFUNCTIONS, R = r_helper.return_r_path()


# =============================================================================
# Note: When using R functions provided column names might get changed
# Especially, do not use +,- or spaces in your column names. Maybe write decorator to
# validate proper column formatting and handle exceptions
# =============================================================================


def quantile_norm(df, cols: Union[list[str], pd.Index], return_cols=False, backend="r"):
    # noinspection PyUnresolvedReferences
    r""" 
    Perform quantile normalization.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    cols : list of str
        Colnames to perform normlisation on.
    return_cols : bool, optional
        if True also the column names of the normalized columns are returned.
        The default is False.
    backend : str, optional
        'py' or 'r'. The default is "r".
        While the python implementation is much faster than r (since R is executed in a subroutine), the
        R Function handles NaNs in a more sophisticated manner than the python function (which just ignores NaNs)

    Returns
    -------
    pd.DataFrame
        The original dataframe with extra columns _normalized.

    Notes
    -----
    The quantile normalization forces the distributions of the samples to be the
    same on the basis of the quantiles of the samples by replacing each point of a
    sample with the mean of the corresponding quantile.
    This is applicable for large datasets with only few changes but will introduce
    errors if the rank assumption is violated i.e. if there are large variations
    across groups to compare. See [2].

    References
    ----------
    [1] https://doi.org/10.1093/bioinformatics/19.2.185

    [2] https://www.biorxiv.org/content/10.1101/012203v1.full

    Examples
    --------
    >>> import autoprot.preprocessing as pp
    >>> import autoprot.visualization as vis
    >>> phosRatio = phos.filter(regex="^Ratio .\/.( | normalized )R.___").columns
    >>> phosLog = pp.log(phos, phosRatio, base=2)
    >>> noNorm = phosLog.filter(regex="log2_Ratio ./. R.___").columns

    Until now this was only preprocessing for the normalisation.

    >>> phos_norm_r = pp.quantile_norm(phosLog, noNorm, backend='r')
    >>> vis.boxplot(phos_norm_r, [noNorm, phos_norm_r.filter(regex="_norm").columns], compare=True)
    >>> plt.show() #doctest: +SKIP

    .. plot::
        :context: close-figs

        import autoprot.preprocessing as pp
        import autoprot.visualization as vis
        import pandas as pd
        phos = pd.read_csv("_static/testdata/Phospho (STY)Sites_mod.zip", sep="\t", low_memory=False)
        phosRatio = phos.filter(regex="^Ratio .\/.( | normalized )R.___").columns
        phosLog = pp.log(phos, phosRatio, base=2)
        noNorm = phosLog.filter(regex="log2_Ratio ./. R.___").columns
        phos_norm_r = pp.quantile_norm(phosLog, noNorm, backend='r')
        vis.boxplot(phos_norm_r, [noNorm, phos_norm_r.filter(regex="_norm").columns], compare=True)
        plt.show()

    """
    if "UID" not in df.columns:
        df["UID"] = range(1, df.shape[0] + 1)

    if not isinstance(cols, list):
        cols = cols.to_list()

    # TODO: Check why python backend fails so poorly
    # See https://github.com/bmbolstad/preprocessCore/blob/master/R/normalize.quantiles.R
    if backend == "py":
        sub_df = df[cols + ["UID"]].copy()
        idx = sub_df["UID"].values
        sub_df = sub_df.drop("UID", axis=1)
        sub_df.index = idx
        # use numpy sort to sort columns independently
        sub_df_sorted = pd.DataFrame(np.sort(sub_df.values, axis=0), index=sub_df.index, columns=sub_df.columns)
        sub_df_mean = sub_df_sorted.mean(axis=1)
        sub_df_mean.index = np.arange(1, len(sub_df_mean) + 1)
        # Assign ranks across the cols, stack the cols so that a multiIndex series
        # is created, map the sub_df_mean series on the series and unstack again
        df_norm = sub_df.rank(axis=0, method="min").stack().astype(int).map(sub_df_mean).unstack()
        res_cols = [f"{i}_normalized" for i in df_norm.columns]
        df_norm.columns = res_cols
        df_norm["UID"] = df_norm.index
        print(df_norm)
        df = df.merge(df_norm, on="UID", how="left")

    elif backend == "r":
        d = os.getcwd()
        data_loc = d + "/input.csv"
        output_loc = d + "/output.csv"

        pp.to_csv(df[["UID"] + cols], data_loc)

        command = [R, '--vanilla',
                   RFUNCTIONS,  # script location
                   "quantile",  # functionName
                   data_loc,  # data location
                   output_loc  # output file
                   ]

        try:
            run(command, stdout=PIPE, check=True, stderr=PIPE, universal_newlines=True)
        except CalledProcessError as err:
            raise Exception(f'Error during execution of R function:\n{err.stderr}') from err

        res = pp.read_csv(output_loc)
        res_cols = [f"{i}_normalized" if i != "UID" else i for i in res.columns]
        res.columns = res_cols
        df = df.merge(res, on="UID")

        os.remove(data_loc)
        os.remove(output_loc)

    else:
        raise (Exception('Please supply either "r" or "py" as value for the backend arg'))

    return (df, [i for i in res_cols if i != "UID"]) if return_cols else df


def vsn(df, cols: Union[list[str], pd.Index], return_cols=False):
    # noinspection PyUnresolvedReferences
    r"""
    Perform Variance Stabilizing Normalization.
    VSN acts on raw intensities and returns the transformed intensities.
    These are similar in scale to a log2 transformation.
    The columns generated by VSN have the suffix _norm.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    cols : list of str
        Colnames to perform normalization on.
        Should correspond to columns with raw intensities/iBAQs (the VSN will transform them eventually).
    return_cols : bool, optional
        if True also the column names of the normalized columns are returned.
        The default is False.

    Returns
    -------
    pd.DataFrame
        The original dataframe with extra columns _normalized.
    list
        Column names after vsn transformation

    References
    ----------
    [1] Huber, W, von Heydebreck, A, Sueltmann, H, Poustka, A, Vingron, M (2002). Variance
    stabilization applied to microarray data calibration and to the quantification of differential expression.
    Bioinformatics 18 Supplement 1, S96-S104.

    Notes
    -----
    The Vsn is a statistical method aiming at making the sample
    variances nondependent from their mean intensities and bringing the
    samples onto a same scale with a set of parametric transformations
    and maximum likelihood estimation.
    
    See https://www.bioconductor.org/packages/release/bioc/html/vsn.html: Differences between transformed intensities
    are analogous to "normalized log-ratios". However, in contrast to the latter, their variance is independent of
    the mean, and they are usually more sensitive and specific in detecting differential transcription.

    Examples
    --------
    >>> import autoprot.preprocessing as pp
    >>> import autoprot.visualization as vis
    >>> import pandas as pd
    >>> phos_lfq = pd.read_csv("_static/testdata/Phospho (STY)Sites_lfq.zip", sep="\t", low_memory=False)
    >>> intens_cols = phos_lfq.filter(regex="Intensity .").columns
    >>> phos_lfq[intens_cols] = phos_lfq[noNorm].replace(0, np.nan)

    Until now this was only preprocessing for the normalisation. We will also log2-transform the intensity data to
    show that VSN normalisation results in values of similar scale than log2 transformation.

    >>> phos_lfq = pp.vsn(phos_lfq, intens_cols)
    >>> norm_cols = phos_lfq.filter(regex="_norm").columns
    >>> phos_lfq, log_cols = pp.log(phos_lfq, intens_cols, base=2, return_cols=True)
    >>> vis.boxplot(phos_lfq, [log_cols, norm_cols], data='Intensity', compare=True)
    >>> plt.show() #doctest: +SKIP

    Note how the VSN normalisation and the log2 transformation result in values of similar magnitude.
    However, the exact variances of the two transformations are different.

    .. plot::
        :context: close-figs

        import autoprot.preprocessing as pp
        import autoprot.visualization as vis
        import pandas as pd
        phos_lfq = pd.read_csv("_static/testdata/Phospho (STY)Sites_lfq.zip", sep="\t", low_memory=False)
        intens_cols = phos_lfq.filter(regex="Intensity .").columns.to_list()
        phos_lfq[intens_cols] = phos_lfq[intens_cols].replace(0, np.nan)
        phos_lfq = pp.vsn(phos_lfq, intens_cols)
        norm_cols = phos_lfq.filter(regex="_norm").columns.to_list()
        phos_lfq, log_cols = pp.log(phos_lfq, intens_cols, base=2, return_cols=True)
        vis.boxplot(phos_lfq, reps=[log_cols, norm_cols], compare=True)
    """
    d = os.getcwd()
    data_loc = d + "/input.csv"
    output_loc = d + "/output.csv"

    if "UID" not in df.columns:
        df["UID"] = range(1, df.shape[0] + 1)

    if not isinstance(cols, list):
        cols = cols.to_list()
    pp.to_csv(df[["UID"] + cols], data_loc)

    command = [R, '--vanilla',
               RFUNCTIONS,  # script location
               "vsn",  # functionName
               data_loc,  # data location
               output_loc  # output file
               ]

    try:
        run(command, stdout=PIPE, check=True, stderr=PIPE, universal_newlines=True)
    except CalledProcessError as err:
        raise Exception(f'Error during execution of R function:\n{err.stderr}') from err

    res = pp.read_csv(output_loc)
    res_cols = [f"{i}_normalized" if i != "UID" else i for i in res.columns]
    res.columns = res_cols

    df = df.merge(res, on="UID")

    os.remove(data_loc)
    os.remove(output_loc)

    return (df, [i for i in res_cols if i != "UID"]) if return_cols else df


def cyclic_loess(df, cols: Union[list[str], pd.Index], return_cols: bool = False):
    # noinspection PyUnresolvedReferences
    r"""
    Perform cyclic Loess normalization.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    cols : list of str
        Colnames to perform normlisation on.
    return_cols : bool, optional
        Whether to return a list of names corresponding to the columns added
        to the dataframe. The default is False.

    Returns
    -------
    pd.DataFrame
        The original dataframe with extra columns _normalized.

    References
    ----------
    [1] https://doi.org/10.1093/bioinformatics/19.2.185

    [2] Cleveland,W.S. and Devlin,S.J. (1998) Locally-weighted regression: an approach to regression analysis by local fitting. J. Am. Stat. Assoc., 83, 596–610

    [3] https://en.wikipedia.org/wiki/Local_regression

    Notes
    -----
    Cyclic loess normalization applies loess normalization to all possible pairs of arrays,
    usually cycling through all pairs several times.
    Loess normalisation (also referred to as Savitzky-Golay filter) locally approximates
    the data around every point using low-order functions and giving less weight to distant
    data points.

    Cyclic loess is slower than quantile, but allows probe-wise weights and
    is more robust to unbalanced differential expression.

    Examples
    --------
    >>> import autoprot.preprocessing as pp
    >>> import autoprot.visualization as vis
    >>> phosRatio = phos.filter(regex="^Ratio .\/.( | normalized )R.___").columns
    >>> phosLog = pp.log(phos, phosRatio, base=2)
    >>> noNorm = phosLog.filter(regex="log2_Ratio ./. R.___").columns

    Until now this was only preprocessing for the normalisation.

    >>> phos_norm_r = pp.cyclic_loess(phosLog,noNorm,backend='r')
    >>> vis.boxplot(phos_norm_r, [noNorm, phos_norm_r.filter(regex="_norm").columns], compare=True)
    >>> plt.show() #doctest: +SKIP

    .. plot::
        :context: close-figs

        import autoprot.preprocessing as pp
        import autoprot.visualization as vis
        import pandas as pd
        phos = pd.read_csv("_static/testdata/Phospho (STY)Sites_mod.zip", sep="\t", low_memory=False)
        phosRatio = phos.filter(regex="^Ratio .\/.( | normalized )R.___").columns
        phosLog = pp.log(phos, phosRatio, base=2)
        noNorm = phosLog.filter(regex="log2_Ratio ./. R.___").columns
        phos_norm_r = pp.cyclic_loess(phosLog, noNorm, backend='r')
        vis.boxplot(phos_norm_r, [noNorm, phos_norm_r.filter(regex="_norm").columns], compare=True)
        plt.show()

    """
    d = os.getcwd()
    data_loc = d + "/input.csv"
    output_loc = d + "/output.csv"

    if "UID" not in df.columns:
        df["UID"] = range(1, df.shape[0] + 1)

    if not isinstance(cols, list):
        cols = cols.to_list()
    pp.to_csv(df[["UID"] + cols], data_loc)

    command = [R, '--vanilla',
               RFUNCTIONS,  # script location
               "cloess",  # functionName
               data_loc,  # data location
               output_loc  # output file
               ]

    run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)

    res = pp.read_csv(output_loc)
    res_cols = [f"{i}_normalized" if i != "UID" else i for i in res.columns]
    res.columns = res_cols

    df = df.merge(res, on="UID")

    os.remove(data_loc)
    os.remove(output_loc)

    return (df, [i for i in res_cols if i != "UID"]) if return_cols else df


def norm_to_prot(entry: pd.Series, prot_df: pd.DataFrame, to_normalize: list[str]):
    """
    Normalize phospho data to total protein level.

    Function has to be applied to phosphosite table.
    e.g. phosTable.apply(lambda x: normToProt(x, dfProt, toNormalize),1)

    Parameters
    ----------
    entry : pd.Series
        Row-like object with index "Protein group IDs".
    prot_df : pd.DataFrame
        MQ ProteinGroups data to which data is normalized.
    to_normalize : list of str
        Which columns to normalize.

    Raises
    ------
    ValueError
        The input array does not contain an index "Protein group IDs".

    Returns
    -------
    pd.Series
        Input array with normalized values.

    Notes
    -----
    Normalization is calculated by subtracting the value of columns toNormalize
    of the protein dataframe from that of the entry, i.e. if intensity ratios
    such as log(pep/prot) should be obtained the operation has to be applied to
    log transformed columns as log(pep) - log(prot) = log(pep/prot).
    """
    try:
        prot_ids = entry["Protein group IDs"]
    except Exception:
        raise ValueError('The input array does not contain an index "Protein group IDs"')
    if ';' in prot_ids:
        prot_ids = [int(i) for i in prot_ids.split(';')]
        prot_df = prot_df[prot_df["id"].isin(prot_ids)]
        poi = prot_df.groupby("Gene names").median()
    else:
        # generate subset of protDf matching the ID of the current protein
        poi = prot_df[prot_df["id"] == int(prot_ids)]
    if poi.shape[0] == 0:
        # can*t normalize. either return non-normalized or drop value?
        corrected = pd.DataFrame([np.nan] * len(to_normalize)).T
    else:
        # log(pep) - log(prot) or log(pep/prot)
        # TODO Does this work? isnt poi[toNormalize] a df and entry a series?
        corrected = entry[to_normalize] - poi[to_normalize]
    return pd.Series(corrected.values[0])
