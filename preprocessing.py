# -*- coding: utf-8 -*-
"""
Autoprot Preprocessing Functions.

@author: Wignand

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

def read_csv(file, sep='\t', low_memory=False, **kwargs):
    r"""
    pd.read_csv with modified default args.

    Parameters
    ----------
    file : str
        Path to input file.
    sep : str, optional
        Column separator. The default is '\t'.
    low_memory : bool, optional
        Whether to reduce memory consumption by inferring dtypes from chunks. The default is False.
    **kwargs :
        see https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html.

    Returns
    -------
    pd.DataFrame
        The parsed dataframe.

    """
    return pd.read_csv(file, sep=sep, low_memory=low_memory, **kwargs)


def to_csv(df, file, sep='\t', index=False, **kwargs):
    r"""
    Write to CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to write.
    file : str
        Path to output file.
    sep : str, optional
        Column separator. The default is '\t'.
    index : bool, optional
        Whether to add the dataframe index to the output. The default is False.
    **kwargs :
        see https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html.

    Returns
    -------
    None.

    """
    df.to_csv(file, sep=sep, index=index, **kwargs)


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


def log(df, cols, base=2, invert=None, return_cols=False, replace_inf=True):
    """
    Perform log transformation.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    cols : list of str
        Cols which are transformed.
    base : int, optional
        Base of log. The default is 2.
    invert : list of int, optional
        Vector corresponding in length to number of to columns.
        Columns are multiplied with corresponding number.
        The default is None.
    return_cols : bool, optional
        Whether to return a list of names corresponding to the columns added
        to the dataframe. The default is False.
    replace_inf : bool, optional
        Whether to replace inf and -inf values by np.nan

    Returns
    -------
    pd.Dataframe
        The log transformed dataframe.
    list
        A list of column names (if returnCols is True).

    Examples
    --------
    First collect colnames holding the intensity ratios.

    >>> protRatio = prot.filter(regex="^Ratio .\/.( | normalized )B").columns
    >>> phosRatio = phos.filter(regex="^Ratio .\/.( | normalized )R.___").columns

    Some ratios need to be inverted as a result from label switches.
    This can be accomplished using the invert variable.
    Log transformations using arbitrary bases can be used, however, 2 and 10 are
    most commonly applied.

    >>> invert = [-1., -1., 1., 1., -1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
    ...       1., 1., 1., 1., 1., 1., 1., -1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
    ...       1., -1.]
    >>> prot2 = autoprot.preprocessing.log(prot, protRatio, base=2, invert=invert*2)
    >>> phos2 = autoprot.preprocessing.log(phos, phosRatio, base=10)

    The resulting dataframe contains log ratios or NaN.

    >>> prot2.filter(regex="log.+_Ratio M/L BC18_1$").head()
       log2_Ratio M/L BC18_1
    0                    NaN
    1              -0.478609
    2                    NaN
    3                    NaN
    4               1.236503
    """
    df = df.copy()
    with np.errstate(divide='ignore'):
        if base == 2:
            for c in cols:
                if replace_inf:
                    df[f"log2_{c}"] = np.nan_to_num(np.log2(df[c]), nan=np.nan, neginf=np.nan, posinf=np.nan)
                else:
                    df[f"log2_{c}"] = np.log2(df[c])
        elif base == 10:
            for c in cols:
                if replace_inf:
                    df[f"log10_{c}"] = np.nan_to_num(np.log10(df[c]), nan=np.nan, neginf=np.nan, posinf=np.nan)
                else:
                    df[f"log10_{c}"] = np.log10(df[c])
        else:
            for c in cols:
                if replace_inf:
                    df[f"log{base}_{c}"] = np.nan_to_num(np.log(df[c]) / np.log(base), nan=np.nan, neginf=np.nan,
                                                         posinf=np.nan)
                else:
                    df[f"log{base}_{c}"] = np.log(df[c]) / np.log(base)

    new_cols = [f"log{base}_{c}" for c in cols]
    if invert is not None:
        df[new_cols] = df[new_cols] * invert
    return (df, new_cols) if return_cols == True else df


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


def expand_site_table(df, cols, replace_zero=True):
    r"""
    Convert a phosphosite table into a phosphopeptide table.

    This functions is used for Phospho (STY)Sites.txt files.
    It converts the phosphosite table into a phosphopeptide table.
    After expansion peptides with no quantitative information are dropped.
    You might want to consider to remove some columns after the expansion.
    For example if you expanded on the normalized ratios it might be good to remove the non-normalized ones, or vice versa.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to be expanded. Must contain a column named "id.
    cols : list of str
        Cols which are going to be expanded (format: Ratio.*___.).
    replace_zero : bool
        If true 0 values in the provided columns are replaced by np.nan (default).
        Set to False if you want explicitely to keep the 0 values after expansion.

    Raises
    ------
    ValueError
        Raised if the dataframe does not contain all columns correspondiong to the
        provided columns without __n extension.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.

    Examples
    --------
    >>> phosRatio = phos.filter(regex="^Ratio .\/.( | normalized )R.___").columns
    >>> phosLog = autoprot.preprocessing.log(phos, phosRatio, base=2)
    >>> phosRatio = phosLog.filter(regex="log2_Ratio .\/. normalized R.___").columns
    >>> phos_expanded = autoprot.preprocessing.expand_site_table(phosLog, phosRatio)
    47936 phosphosites in dataframe.
    47903 phosphopeptides in dataframe after expansion.
    """
    df = df.copy(deep=True)
    print(f"{df.shape[0]} phosphosites in dataframe.")
    dfs = []
    expected = df.shape[0] * 3
    # columns to melt
    melt = cols  # e.g. "Ratio M/L normalized R2___1"
    # drop duplicates and preserve order (works with Python >3.7)
    melt_set = list(dict.fromkeys([i[:-4] for i in melt]))  # e.g. "Ratio M/L normalized R2"
    # Due to MaxQuant column names we might have to drop some columns
    check = [i in df.columns for i in melt_set]  # check if the colnames are present in the df
    if False not in check:
        df.drop(melt_set, axis=1, inplace=True)  # remove cols w/o ___n
    if True in check and False in check:
        raise ValueError("The columns you provided or the dataframe are not suitable!")

    if df[melt].eq(0).any().any() and replace_zero:
        warnings.warn(
            "The dataframe contains 0 values that will not be filtered out eventually. Will replace by np.nan. If "
            "this is not intended set replace_zero to False.")
        df[melt].replace(0, np.nan)

    # generate a separated melted df for every entry in the melt_set
    for i in melt_set:
        cs = list(df.filter(regex=i + '___').columns) + ["id"]  # reconstruct the ___n cols for each melt_set entry
        # melt the dataframe generating an 'id' column,
        # a 'variable' col with the previous colnames
        # and a 'value' col with the previous values
        dfs.append(pd.melt(df[cs], id_vars='id'))

    # =============================================================================
    #     pd.melt
    #
    #     x, a__1, a__2, a__3
    #     ->
    #     x, a, 1
    #     x, a, 2
    #     x, a, 3
    # =============================================================================

    # the large first df is now called temp
    temp = df.copy(deep=True)
    temp = temp.drop(melt, axis=1)  # drop the ___n entries from the original df

    t = pd.DataFrame()
    for idx, df in enumerate(dfs):
        x = df["variable"].iloc[0].split('___')[0]  # reconstructs the colname w/o ___n, e.g. Ratio M/L normalized R2
        if idx == 0:  # the first df contains all peptides and all multiplicities as rows
            t = df.copy(deep=True)
            t.columns = ["id", "Multiplicity", x]  # e.g. 0, Ratio M/L normalized R2___1, 0.67391
            t["Multiplicity"] = t["Multiplicity"].apply(lambda col_header: col_header.split('___')[1])  # 0, 1, 0.673
        else:  # in the subsequent dfs id and multiplicities can be dropped as only the ratio is new information
            # compared to the first df
            df.columns = ["id", "Multiplicity", x]
            df = df.drop(["id", "Multiplicity"], axis=1)  # keep only the x col
            t = t.join(df, rsuffix=f'_{idx}')  # horizontally joins the new col with the previous df
    # merging on ids gives the melted peptides their names back´
    temp = temp.merge(t, on='id', how='left')
    temp["Multiplicity"] = temp["Multiplicity"].astype(int)  # is previously str

    if temp.shape[0] != expected:
        print("The expansion of site table is probably not correct!!! Check it! Maybe you provided wrong columns?")

    # remove rows that contain no modified peptides
    # this requires that unidentified modifications are set to np.nan! See warning above that checks just this.
    temp = temp[~(temp[melt_set].isnull().all(1))]
    print(f"{temp.shape[0]} phospho peptides in dataframe after expansion.")
    return temp


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


def go_annot(prots: pd.DataFrame, gos: list, only_prots: bool = False, exact: bool = True) \
        -> Union[pd.DataFrame, pd.Series]:
    """
    Filter a list of experimentally determined gene names by GO annotation.

    Homo sapiens.gene_info and gene2go files are needed for annotation

    In case of multiple gene names per line (e.g. AKT1;PKB)
    only the first name will be extracted.

    Parameters
    ----------
    prots : list of str
        List of Gene names.
    gos : list of str
        List of go terms.
    only_prots : bool, optional
        Whether to return dataframe or only list of gene names annotated with terms. The default is False.
    exact : bool, optional
        whether go term must match exactly. i.e. MAPK activity <-> regulation of MAPK acitivity etc. The default is True.

    Returns
    -------
    pd.DataFrame or pd.Series
        Dataframe with columns "index", "Gene names", "GeneID", "GO_ID", "GO_term"
        or
        Series with gene names

    Examples
    --------
    >>> gos = ["ribosome"]
    >>> go = autoprot.preprocessing.go_annot(prot["Gene names"],gos, only_prots=False)
    >>> go.head()
       index Gene names  GeneID       GO_ID   GO_term
    0   1944      RPS27    6232  GO:0005840  ribosome
    1   6451      RPS25    6230  GO:0005840  ribosome
    2   7640     RPL36A    6173  GO:0005840  ribosome
    3  11130      RRBP1    6238  GO:0005840  ribosome
    4  16112        SF1    7536  GO:0005840  ribosome
    """
    with resources.open_binary("autoprot.data", "Homo_sapiens.zip") as d:
        gene_info = pd.read_csv(d, sep='\t', compression='zip')
    with resources.open_binary("autoprot.data", "gene2go_alt.zip") as d:
        gene2go = pd.read_csv(d, sep='\t', compression='zip')
    # generate dataframe with single columns corresponding to experimental gene names
    prots = pd.DataFrame(pd.Series([str(i).upper().split(';')[0] for i in prots]), columns=["Gene names"])
    # add the column GeneID by merging with the gene_info table
    prots = prots.merge(gene_info[["Symbol", "GeneID"]], left_on="Gene names", right_on="Symbol", how='inner')
    # add the columns GO_ID and GO_term by merging on GeneID
    prots = prots.merge(gene2go[["GeneID", "GO_ID", "GO_term"]], on="GeneID", how='inner')

    # if the go terms must match exactly, pandas' isin is used
    if exact:
        red_prots = prots[prots["GO_term"].isin(gos)]
    # if they should only contain the go term, the str contains method with the OR separator is used
    else:
        red_prots = prots[prots["GO_term"].str.contains('|'.join(gos), regex=True)]

    # if only the proteins should be returned, the Symbol column from the GO annotation is returned
    if only_prots:
        return red_prots['Symbol'].drop_duplicates().reset_index(drop=True)
    # else the complete dataframe without the Symabol column is returned
    else:
        return red_prots.drop_duplicates().drop("Symbol", axis=1).reset_index(drop=True)


def motif_annot(df, motif, col="Sequence window"):
    """
    Search for phosphorylation motif in the provided dataframe.

    If not specified, the "Sequence window" column is searched.
    The phosphorylated central residue in a motif has to be indicated with "S/T".
    Arbitrary amino acids can be denoted with x.

    Parameters
    ----------
    df : pd.Dataframe
        input dataframe.
    motif : str
        Target motif. E.g. "RxRxxS/T", "PxS/TP" or "RxRxxS/TxSxxR"
    col : str, optional
        Alternative column to be searched in if Sequence window is not desired.
        The default is "Sequence window".

    Returns
    -------
    pd.dataframe
        Dataframe with additional boolean column with True/False for whether the motif is found in this .

    """
    df = df.copy()  # make sure to keep the original dataframe unmodified

    # TODO
    # make some assertions that the column is indeed the proper MQ output
    # (might want to customize the possibilities later)

    def find_motif(x, col, motif, motlen):
        seq = x[col]
        seqs = seq.split(';') if ";" in seq else [seq]
        for seq in seqs:
            pos = 0
            pos2 = re.finditer(motif, seq)
            if pos2:
                # iterate over re match objects
                for p in pos2:
                    # p.end() is the index of the last matching element of the searchstring
                    pos = p.end()
                    # only return a match if the motif in centred in the sequence window
                    # i.e. if the corresponding peptide was identified
                    if pos == np.floor(motlen / 2 + 1):
                        return 1
        return 0

    assert (col in df.columns)
    assert (len(df[col].iloc[0]) % 2 == 1)

    # generate a regex string out of the input motif
    search = motif.replace('x', '.').replace('S/T', '(S|T)').upper()
    i = search.index("(S|T)")
    before = search[:i]
    after = search[i + 5:]
    # the regex contains a lookbehind (?<=SEQUENCEBEFORE), the actual modified residues (S/T)
    # and a lookahead with the following seqeunce for this motif (?=SEQUENCEAFTER)
    search = f"(?<={before})(S|T)(?={after})"
    # the lengths of the sequences in the sequence window column are all the same, take it from the first row
    motlen = len(df[col].iloc[0])
    df[motif] = df.apply(find_motif, col=col, motif=search, motlen=motlen, axis=1)

    return df


# =============================================================================
# IMPUTATION ALGORITHMS
# =============================================================================
def imp_min_prob(df, cols_to_impute, max_missing=None, downshift=1.8, width=.3):
    r"""
    Perform an imputation by modeling a distribution on the far left site of the actual distribution.

    The final distribution will be mean shifted and has a smaller variation.
    Intensities should be log transformed before being supplied to this function.

    Downsshift: mean - downshift*sigma
    Var: width*sigma

    Parameters
    ----------
    df : pd.dataframe
        Dataframe on which imputation is performed.
    cols_to_impute : list
        Columns to impute. Should correspond to a single condition (i.e. control).
    max_missing : int, optional
        How many missing values have to be missing across all columns to perfom imputation
        If None all values have to be missing. The default is None.
    downshift : float, optional
        How many Stds to lower values the mean of the new population is shifted. The default is 1.8.
    width : float, optional
        How to scale the Std of the new distribution with respect to the original. The default is .3.

    Returns
    -------
    pd.dataframe
        The dataframe with imputed values.

    Examples
    --------
    >>> forImp = np.log10(phos.filter(regex="Int.*R1").replace(0, np.nan))
    >>> impProt = pp.imp_min_prob(forImp, phos.filter(regex="Int.*R1").columns,
    ...                         width=.4, downshift=2.5)
    >>> impProt.filter(regex="Int.*R1")[impProt["Imputed"]==False].mean(1).hist(density=True, bins=50,
    ...                                                                         label="not Imputed")
    >>> impProt.filter(regex="Int.*R1")[impProt["Imputed"]==True].mean(1).hist(density=True, bins=50,
    ...                                                                        label="Imputed")
    >>> plt.legend()

    .. plot::
        :context: close-figs

        import autoprot.preprocessing as pp
        import autoprot.visualization as vis
        import pandas as pd
        phos = pd.read_csv("_static/testdata/Phospho (STY)Sites_mod.zip", sep="\t", low_memory=False)
        forImp = np.log10(phos.filter(regex="Int.*R1").replace(0, np.nan))
        impProt = pp.imp_min_prob(forImp, phos.filter(regex="Int.*R1").columns, width=.4, downshift=2.5)
        fig, ax1 = plt.subplots(1)
        impProt.filter(regex="Int.*R1")[impProt["Imputed"]==False].mean(1).hist(density=True, bins=50, label="not Imputed", ax=ax1)
        impProt.filter(regex="Int.*R1")[impProt["Imputed"]==True].mean(1).hist(density=True, bins=50, label="Imputed", ax=ax1)
        plt.legend()
        plt.show()
    """
    df = df.copy(deep=True)

    if max_missing is None:
        max_missing = len(cols_to_impute)
    # idxs of all rows in which imputation will be performed
    idx_no_ctrl = df[df[cols_to_impute].isnull().sum(1) >= max_missing].index
    df["Imputed"] = False
    df.loc[idx_no_ctrl, "Imputed"] = True

    for col in cols_to_impute:
        df[col + '_imputed'] = df[col]
        mean = df[col].mean()
        var = df[col].std()
        mean_ = mean - downshift * var
        var_ = var * width

        # generate random numbers matching the target dist
        rnd = np.random.normal(mean_, var_, size=len(idx_no_ctrl))
        for i, idx in enumerate(idx_no_ctrl):
            df.loc[idx, col + '_imputed'] = rnd[i]

    return df


def imp_seq(df, cols, print_r=True):
    """
    Perform sequential imputation in R using impSeq from rrcovNA.

    See https://rdrr.io/cran/rrcovNA/man/impseq.html for a description of the
    algorithm.
    SEQimpute starts from a complete subset of the data set Xc and estimates sequentially the missing values in an incomplete observation, say x*, by minimizing the determinant of the covariance of the augmented data matrix X* = [Xc; x']. Then the observation x* is added to the complete data matrix and the algorithm continues with the next observation with missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    cols : list of str
        Colnames to perform imputation of.
    print_r : bool, optional
        Whether to print the output of R, default is True.

    Returns
    -------
    pd.DataFrame
        Dataframe with imputed values.
        Cols with imputed values are named _imputed.
        Contains a col UID that was used for processing.

    """
    d = os.getcwd()
    dataLoc = d + "/input.csv"
    outputLoc = d + "/output.csv"

    if "UID" not in df.columns:
        # UID is basically a row index starting at 1
        df["UID"] = range(1, df.shape[0] + 1)

    if not isinstance(cols, list):
        cols = cols.to_list()
    to_csv(df[["UID"] + cols], dataLoc)

    command = [R, '--vanilla',
               RFUNCTIONS,  # script location
               "impSeq",  # functionName
               dataLoc,  # data location
               outputLoc  # output file
               ]

    p = run(command,
            stdout=PIPE,
            stderr=STDOUT,
            universal_newlines=True)

    if print_r:
        print(p.stdout)

    res = read_csv(outputLoc)
    # append a string to recognise the cols
    res_cols = [f"{i}_imputed" if i != "UID" else i for i in res.columns]
    # change back the R colnames
    res_cols = [x.replace('.', ' ') for x in res_cols]
    res.columns = res_cols

    # merge and retain the rows of the original df
    df = df.merge(res, how='left', on="UID")
    # drop UID again
    df.drop("UID", axis=1, inplace=True)

    # os.remove(dataLoc)
    # os.remove(outputLoc)

    return df


def dima(df, cols, selection_substr=None, ttest_substr='cluster', methods='fast',
         npat=20, performance_metric='RMSE', print_r=True):
    """
    Perform Data-Driven Selection of an Imputation Algorithm.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    cols : list of str
        Colnames to perform imputation on.
        NOTE: if used on intensities, use log-transformed values.
    selection_substr : str
        pattern to extract columns for processing during DIMA run.
    ttest_substr : 2-element list or str
        If string, two elements need to be separated by ','
        If list, concatenation will be done automatically.
        The two elements must be substrings of the columns to compare.
        Make sure that for each substring at least two matching colnames
        are present in the data.
    methods : str or list of str, optional
        Methods to evaluate. Default 'fast' for the 9 most used imputation
        methods. Possible values are 'impSeqRob','impSeq','missForest',
        'imputePCA','ppca','bpca', ...
    npat : int, optional
        Number of missing value patterns to evaluate
    performance_metric : str, optional
        Metric used to select the best algorithm. Possible values are
        Dev, RMSE, RSR, pF,  Acc, PCC, RMSEt.
    print_r : bool
        Whether to print the R output to the Python console.

    Returns
    -------
    pd.DataFrame
        Input dataframe with imputed values.
    pd.DataFrame
        Overview of performance metrices of the different algorithms.

    Examples
    --------
    We will use a standard sample dataframe and generate some missing values to
    demonstrate the imputation.
    
    >>> from autoprot import preprocessing as pp
    >>> import seaborn as sns
    >>> import pandas as pd
    >>> import numpy as np
    >>> iris = sns.load_dataset('iris')
    >>> _ = iris.pop('species')
    >>> for col in iris.columns:
    ...     iris.loc[iris.sample(frac=0.1).index, col] = np.nan

    >>> imp, perf = pp.dima(
    ...     iris, iris.columns, performance_metric="RMSEt", ttest_substr=["petal", "sepal"]
    ... )
    
    >>> imp.head()
       sepal_length  sepal_width  petal_length  ...  sepal_width_imputed  petal_length_imputed  petal_width_imputed
    0           5.1          3.5           1.4  ...                  3.5                   1.4                  0.2
    1           4.9          3.0           1.4  ...                  3.0                   1.4                  0.2
    2           4.7          3.2           1.3  ...                  3.2                   1.3                  0.2
    3           4.6          3.1           1.5  ...                  3.1                   1.5                  0.2
    4           5.0          3.6           1.4  ...                  3.6                   1.4                  0.2
    
    [5 rows x 9 columns]
    
    >>> perf.head()
                Deviation      RMSE       RSR  p-Value_F-test   Accuracy       PCC  RMSEttest
    impSeqRob    0.404402  0.531824  0.265112        0.924158  94.735915  0.997449   0.222656
    impSeq       0.348815  0.515518  0.256984        0.943464  95.413732  0.997563   0.223783
    missForest   0.348815  0.515518  0.256984        0.943464  95.413732  0.997563   0.223783
    imputePCA    0.404402  0.531824  0.265112        0.924158  94.735915  0.997449   0.222656
    ppca         0.377638  0.500354  0.249424        0.933919  95.000000  0.997721   0.199830
    
    References
    ----------
    Egert, J., Brombacher, E., Warscheid, B. & Kreutz, C. DIMA: Data-Driven Selection of an Imputation Algorithm. Journal of Proteome Research 20, 3489–3496 (2021-06).
    """
    if not df.isnull().values.any():
        raise ValueError('Your dataframe does not contain missing values. Will return as is.')
    df = df.copy(deep=True)

    d = os.getcwd()
    data_loc = d + "/input.csv"
    output_loc = d + "/output.csv"

    for col in cols:
        mvs = df[col].isna().sum() / df[col].size
        print(f"{mvs * 100:.2f}% MVs in column {col}")

    if selection_substr is not None:
        df = df.filter(regex=selection_substr)

    if "UID" not in df.columns:
        # UID is basically a row index starting at 1
        df["UID"] = range(1, df.shape[0] + 1)

    if not isinstance(cols, list):
        cols = cols.to_list()
    to_csv(df[["UID"] + cols], data_loc)

    if isinstance(ttest_substr, list):
        ttest_substr = ','.join(ttest_substr)

    if isinstance(methods, list):
        methods = ','.join(methods)

    command = [R, '--vanilla',
               RFUNCTIONS,  # script location
               "dima",  # functionName
               data_loc,  # data location
               output_loc,  # output file
               ttest_substr,  # substring for ttesting
               methods,  # method(s) aka algorithms to benchmark
               str(npat),  # number of patterns
               performance_metric  # to select the best algorithm
               ]

    p = run(command,
            stdout=PIPE,
            stderr=STDOUT,
            universal_newlines=True)

    if print_r:
        print(p.stdout)

    res = read_csv(output_loc)
    # keep only the columns added by DIMA and the UID for merging
    res = res.loc[:, (res.columns.str.contains('Imputation')) | (res.columns.str.contains('UID'))]
    # append a string to recognise the cols
    res_cols = [f"{i}_imputed" if i != "UID" else i for i in res.columns]
    # remove the preceding string Imputation
    res_cols = [x.replace('Imputation.', '') for x in res_cols]
    res.columns = res_cols

    # merge and retain the rows of the original df
    df = df.merge(res, how='left', on="UID")
    # drop UID again
    df.drop("UID", axis=1, inplace=True)

    perf = read_csv(output_loc[:-4] + '_performance.csv')

    os.remove(data_loc)
    os.remove(output_loc)
    os.remove(output_loc[:-4] + '_performance.csv')

    return df, perf


def exp_semi_col(df, scCol, newCol, castTo=None):
    r"""
    Expand a semicolon containing string column and generate a new column based on its content.

    Parameters
    ----------
    df : pd.dataframe
        Dataframe to expant columns.
    scCol : str
        Colname of a column containing semicolon-separated values.
    newCol : str
        Name for the newly generated column.
    castTo : dtype, optional
        If provided new column will be set to the provided dtype. The default is None.

    Returns
    -------
    df : pd.dataframe
        Dataframe with the semicolon-separated values on separate rows.

    Examples
    --------
    >>> expSemi = phos.sample(100)
    >>> expSemi["Proteins"].head()
    0    P61255;B1ARA3;B1ARA5
    0    P61255;B1ARA3;B1ARA5
    0    P61255;B1ARA3;B1ARA5
    1    Q6XZL8;F7CVL0;F6SJX8
    1    Q6XZL8;F7CVL0;F6SJX8
    Name: Proteins, dtype: object
    >>> expSemi = autoprot.preprocessing.expSemiCol(expSemi, "Proteins", "SingleProts")
    >>> expSemi["SingleProts"].head()
    0    P61255
    0    B1ARA3
    0    B1ARA5
    1    Q6XZL8
    1    F7CVL0
    Name: SingleProts, dtype: object
    """
    df = df.copy(deep=True)
    df = df.reset_index(drop=True)

    # make temp df with expanded columns
    temp = df[scCol].str.split(";", expand=True)
    # use stack to bring it into long format series and directly convert back to df
    temp = pd.DataFrame(temp.stack(), columns=[newCol])
    # get first level of multiindex (corresponds to original index)
    temp.index = temp.index.get_level_values(0)
    # join on idex
    df = df.join(temp)

    if castTo is not None:
        df[newCol] = df[newCol].astype(castTo)

    return df


def merge_semi_cols(m1: pd.DataFrame, m2: pd.DataFrame, semicolon_col1: str, semicolon_col2: str = None):
    """
    Merge two dataframes on a semicolon separated column.

    Here m2 is merged to m1 (left merge).
    -> entries in m2 which are not matched to m1 are dropped

    Parameters
    ----------
    m1 : pd.Dataframe
        First dataframe to merge.
    m2 : pd.Dataframe
        Second dataframe to merge with first.
    semicolon_col1 : str
        Colname of a column containing semicolon-separated values in m1.
    semicolon_col2 : str, optional
        Colname of a column containing semicolon-separated values in m2.
        If sCol2 is None it is assumed to be the same as sCol1.
        The default is None.

    Returns
    -------
    pd.dataframe
        Merged dataframe with expanded columns.

    """

    # =============================================================================
    #     Example
    #
    #     df1
    #       col1  num1
    #     0  A;B     0
    #     1  C;D     1
    #     2  E;F     2
    #
    #     df2
    #       col2  num2
    #     0  A;B    10
    #     1  C;D    11
    #     2    E    12
    #
    #     mergeSemiCols(df1, df2, 'col1', 'col2')
    #       col1_x  num2 col1_y  num1
    #     0    A;B    10    A;B     0
    #     1    C;D    11    C;D     1
    #     2      E    12    E;F     2
    # =============================================================================

    # helper functions
    def _form_merge_pairs(s):
        """
        Group the data back on the main data identifier and create the appropiate matching entries of the other data.

        Parameters
        ----------
        s : pd.groupby
            Groupby object grouped on the first identifier.

        Returns
        -------
        pd.Series
            A Series with the ids corresponding to m1 and the entries to the matching idcs in m2.

        """
        ids = list({i for i in s if not np.isnan(i)})
        return ids or [np.nan]

    def _aggregate_duplicates(s):
        # this might be a oversimplification but there should only be
        # object columns and numerical columns in the data

        if s.name == "mergeID_m1":
            print(s)

        if s.dtype != "O":
            return s.median()
        try:
            return s.mode()[0]
        except Exception:
            return s.mode()

    m1, m2 = m1.copy(), m2.copy()

    # make IDs to reduce data after expansion
    m1["mergeID_m1"] = range(m1.shape[0])
    m2["mergeID_m2"] = range(m2.shape[0])

    # rename ssCol2 if it is not the same as ssCol1
    if semicolon_col1 != semicolon_col2:
        m2.rename(columns={semicolon_col2: semicolon_col1}, inplace=True)

    # expand the semicol columns and name the new col sUID
    m1_exp = expSemiCol(m1, semicolon_col1, "sUID")
    m2_exp = expSemiCol(m2, semicolon_col1, "sUID")

    # add the appropriate original row indices of m2 to the corresponding rows
    # of m1_exp
    # here one might want to consider other kind of merges
    merge = pd.merge(m1_exp, m2_exp[["mergeID_m2", "sUID"]], on="sUID", how='left')

    merge_pairs = merge[["mergeID_m1", "mergeID_m2"]].groupby("mergeID_m1").agg(_form_merge_pairs)
    # This is neccessary if there are more than one matching columns
    merge_pairs = merge_pairs.explode("mergeID_m2")

    # merge of m2 columns
    merge_pairs = (merge_pairs
                   .reset_index()
                   .merge(m2, on="mergeID_m2", how="left")
                   .groupby("mergeID_m1")
                   .agg(_aggregate_duplicates)
                   .reset_index())

    # merge of m1 columns (those should all be unique)
    merge_pairs = (merge_pairs
                   .merge(m1, on="mergeID_m1", how="outer"))

    return merge_pairs.drop(["mergeID_m1", "mergeID_m2"], axis=1)


def quantile_norm(df, cols, return_cols=False, backend="r"):
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

        to_csv(df[["UID"] + cols], data_loc)

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

        res = read_csv(output_loc)
        res_cols = [f"{i}_normalized" if i != "UID" else i for i in res.columns]
        res.columns = res_cols
        df = df.merge(res, on="UID")

        os.remove(data_loc)
        os.remove(output_loc)

    else:
        raise (Exception('Please supply either "r" or "py" as value for the backend arg'))

    return (df, [i for i in res_cols if i != "UID"]) if return_cols else df


def vsn(df, cols, return_cols=False, backend='r'):
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
        Colnames to perform normalisation on.
        Should correspond to columns with raw intensities/iBAQs (the VSN will transform them eventually).
    return_cols : bool, optional
        if True also the column names of the normalized columns are returned.
        The default is False.
    backend : str, optional
        Only 'r' is implemented. The default is "r".

    Returns
    -------
    pd.DataFrame
        The original dataframe with extra columns _normalized.

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
    to_csv(df[["UID"] + cols], data_loc)

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

    res = read_csv(output_loc)
    res_cols = [f"{i}_normalized" if i != "UID" else i for i in res.columns]
    res.columns = res_cols

    df = df.merge(res, on="UID")

    os.remove(data_loc)
    os.remove(output_loc)

    return (df, [i for i in res_cols if i != "UID"]) if return_cols else df


def cyclic_loess(df, cols, backend='r'):
    r"""
    Perform cyclic Loess normalization.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    cols : list of str
        Colnames to perform normlisation on.
    backend : str, optional
        Only 'r' is implemented. The default is "r".

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

    >>> phos_norm_r = pp.cyclic_loess(phosLog, noNorm, backend='r')
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
    to_csv(df[["UID"] + cols], data_loc)

    command = [R, '--vanilla',
               RFUNCTIONS,  # script location
               "cloess",  # functionName
               data_loc,  # data location
               output_loc  # output file
               ]

    run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)

    res = read_csv(output_loc)
    res_cols = [f"{i}_normalized" if i != "UID" else i for i in res.columns]
    res.columns = res_cols

    df = df.merge(res, on="UID")

    os.remove(data_loc)
    os.remove(output_loc)

    return df


def to_canonical_ps(series, organism="human", get_seq="online", uniprot=None):
    """
    Convert phosphosites to "canonical" phosphosites.

    Parameters
    ----------
    series : pd.Series
        Series containing the indices "Gene names" and "Sequence Window".
        Corresponds e.g. to a row in MQ Phospho(STY)Sites.txt.
    organism : str, optional
        This conversion is based on Uniprot Identifier used in PSP data.
        possible organisms: 'mouse', 'human', 'rat', 'sheep', 'SARSCoV2', 'guinea pig', 'cow',
        'hamster', 'fruit fly', 'dog', 'rabbit', 'pig', 'chicken', 'frog',
        'quail', 'horse', 'goat', 'papillomavirus', 'water buffalo',
        'marmoset', 'turkey', 'cat', 'starfish', 'torpedo', 'SARSCoV1',
        'green monkey', 'ferret'. The default is "human".
    get_seq : "local" or "online"

    Notes
    -----
    This function compares a certain gene name to the genes found in the
    phosphosite plus (https://www.phosphosite.org) phosphorylation site dataset.

    Returns
    -------
    list of (str, str, str)
        (UniProt ID, Position of phosphosite in the UniProt sequence, score)
    Proteins with two Gene names seperated by a semicolon are given back in the same way and order.

    Examples
    --------
    The correct position of the phosphorylation is returned independent of the
    completeness of the sequence window.

    >>> series=pd.Series(['PEX14', "VSNESTSSSPGKEGHSPEGSTVTYHLLGPQE"], index=['Gene names', 'Sequence window'])
    >>> autoprot.preprocessing.to_canonical_ps(series, organism='human')
    ('O75381', 282)
    >>> series=pd.Series(['PEX14', "_____TSSSPGKEGHSPEGSTVTYHLLGP__"], index=['Gene names', 'Sequence window'])
    >>> autoprot.preprocessing.to_canonical_ps(series, organism='human')
    ('O75381', 282)
    """

    # open the phosphosite plus phosphorylation dataset
    with resources.open_binary('autoprot.data', "phosphorylation_site_dataset.zip") as d:
        ps = pd.read_csv(d, sep='\t', compression='zip')

    if uniprot is None:
        # open the uniprot datatable if not provided
        with resources.open_binary('autoprot.data',
                                   r"uniprot-compressed_true_download_true_fields_accession_2Cid_2Cgene_n-2022.11.29-14.49.20.07.tsv.gz") as e:
            uniprot = pd.read_csv(e, sep='\t', compression='gzip')

    def get_uniprot_accession(gene, organism):
        """Find the matching UniProt ID in the phosphorylation_site_dataset given a gene name and a corresponding
         organism. """
        gene = gene.upper()
        try:
            gene_in_GENE = (ps['GENE'].str.upper() == gene) & (ps['ORGANISM'] == organism)
            gene_in_PROTEIN = (ps['PROTEIN'].str.upper() == gene) & (ps['ORGANISM'] == organism)

            uniprot_acc = ps.loc[(gene_in_GENE | gene_in_PROTEIN), 'ACC_ID'].iloc[0]

            return uniprot_acc

        except:
            return False

    def get_uniprot_sequence(uniprot_acc):
        """Download sequence from uniprot by UniProt ID."""
        url = f"https://www.uniprot.org/uniprot/{uniprot_acc}.fasta"
        response = requests.get(url)
        seq = "".join(response.text.split('\n')[1:])
        return seq

    def get_uniprot_sequence_locally(uniprot_acc, organism):
        """Get sequence from a locally stored uniprot file by UniProt ID."""

        if organism == "mouse":
            uniprot_organism = "Mus musculus (Mouse)"
        else:
            uniprot_organism = "Homo sapiens (Human)"

        seq = uniprot["Sequence"][(uniprot["Entry"] == uniprot_acc) & (uniprot["Organism"] == uniprot_organism)]
        try:
            seq = seq.values.tolist()[0]
        except IndexError:
            print(f"no match found for {uniprot_acc}")
            seq = False
        return seq

    def get_canonical_psite(seq, ps_seq, aa_to_ps):
        """Align an experimental phosphosite sequence window to the corresponding UniProt sequence."""
        alignment = pairwise2.align.localms(sequenceA=seq, sequenceB=ps_seq, match=2, mismatch=-2, open=-1, extend=-1)

        form_align = format_alignment(*alignment[0])
        start = int(form_align.lstrip(' ').split(' ')[0])
        missmatched_aa = form_align.split('\n')[0].split(' ')[1].count("-")

        try:
            offset = int(form_align.split('\n')[2].lstrip(' ').split(' ')[0]) - 1
        except:
            offset = 0

        canonical_psite = start + (aa_to_ps - missmatched_aa - offset)

        # debugging
        seq_window_alignment = form_align.split('\n')
        score = form_align.split('\n')[3].split(' ')[2]
        score = int(score[6:])

        return canonical_psite, score

    uniprot_acc_extr = []
    ps_seq_extr = []
    gene = str(series["Gene names"])
    ps_seq = series["Sequence window"]

    ps_seq_list = ps_seq.split(';')
    gene_list = gene.split(';')
    if len(ps_seq_list) != len(gene_list):

        if get_seq == "online":
            print(f'Gene list does not match sequence list:\n {gene}\n{ps_seq}')

        ps_seq_list = ps_seq_list * len(gene_list)

    for idx, g in enumerate(gene_list):
        uniprot_acc_ex = get_uniprot_accession(g, organism)
        if not uniprot_acc_ex:
            continue
        uniprot_acc_extr.append(uniprot_acc_ex)
        ps_seq_extr.append(ps_seq_list[idx])

    if len(uniprot_acc_extr) == 0:
        return "No matching Uniprot ID found"

    canonical_ps_list = []
    score_list = []
    for uniprot_acc, ps_seq in zip(uniprot_acc_extr, ps_seq_extr):

        if get_seq == "local":
            seq = get_uniprot_sequence_locally(uniprot_acc, organism)
        if get_seq == "online":
            seq = get_uniprot_sequence(uniprot_acc)

        if seq == False:
            canonical_ps_list.append("no match")
        else:
            aa_to_ps = len(ps_seq[0:15].lstrip('_'))
            ps_seq = ps_seq.strip('_')
            canonical_ps, score = get_canonical_psite(seq, ps_seq, aa_to_ps)
            canonical_ps_list.append(str(canonical_ps))
            score_list.append(str(score))

    return [(";".join(uniprot_acc_extr)), (";".join(canonical_ps_list)), (";".join(score_list))]


def calculate_iBAQ(intensity, gene_name=None, protein_id=None, organism="human", get_seq="online", uniprot=None):
    """
    Convert raw intensities to ‘intensity-based absolute quantification’ or iBAQ intensities.
    Given intensities are divided by the number of theoretically observable tryptic peptides. 

    Parameters
    ----------
    intensity : int
        Integer with raw MS intensity for Transformation.
    gene_name : str
        Gene name of the protein related to the intensity given.
    protein_id= str
        Uniprot Protein ID of the protein related to the intensity given.
    organism : str, optional
        This conversion is based on Uniprot Identifier used in data.
        possible organisms: 'mouse', 'human', 'rat', 'sheep', 'SARSCoV2', 'guinea pig', 'cow',
        'hamster', 'fruit fly', 'dog', 'rabbit', 'pig', 'chicken', 'frog',
        'quail', 'horse', 'goat', 'papillomavirus', 'water buffalo',
        'marmoset', 'turkey', 'cat', 'starfish', 'torpedo', 'SARSCoV1',
        'green monkey', 'ferret'. The default is "human".
    get_seq : str, "local" or "online"
        Defines if sequence is fetched locally or downloaded from uniprot.
        It is advised to give a locally loaded dataframe when function is used in batch processing.
    uniprot : pd.DataFrame, optional
        contains Sequences listed by Gene Names and UniProt IDs

    Notes
    -----
    This function gets the protein sequence online at UniProt.
    For batch processing it is advisable to provide local Sequence data or
    use the local copy of the UniProt in autoprot, be aware to keep it up to date.

    Returns
    -------
    int : iBAQ intensity

    Examples
    --------

    """

    if protein_id is None and get_seq == "online":
        # open the phosphosite plus phosphorylation dataset
        with resources.open_binary('autoprot.data', "phosphorylation_site_dataset.zip") as d:
            ps = pd.read_csv(d, sep='\t', compression='zip')

    if uniprot is None and get_seq == "offline":
        # open the uniprot datatable if not provided
        with resources.open_binary('autoprot.data',
                                   r"uniprot-compressed_true_download_true_fields_accession_2Cid_2Cgene_n-2022.11.29-14.49.20.07.tsv.gz") as e:
            uniprot = pd.read_csv(e, sep='\t', compression='gzip')

    def get_uniprot_accession(gene_name, organism):
        """Find the matching UniProt ID in the phosphorylation_site_dataset given a gene name and a corresponding
        organism. """
        gene_name = gene_name.upper()
        try:
            gene_in_GENE = (ps['GENE'].str.upper() == gene_name) & (ps['ORGANISM'] == organism)
            gene_in_PROTEIN = (ps['PROTEIN'].str.upper() == gene_name) & (ps['ORGANISM'] == organism)

            uniprot_acc = ps.loc[(gene_in_GENE | gene_in_PROTEIN), 'ACC_ID'].iloc[0]

            return uniprot_acc

        except:
            return False

    def get_uniprot_sequence(uniprot_acc):
        """Download sequence from uniprot by UniProt ID."""
        url = f"https://www.uniprot.org/uniprot/{uniprot_acc}.fasta"
        response = requests.get(url)
        sequence = "".join(response.text.split('\n')[1:])
        return sequence

    def get_uniprot_sequence_locally(uniprot_acc, organism):
        """Get sequence from a locally stored uniprot file by UniProt ID."""

        if organism == "mouse":
            uniprot_organism = "Mus musculus (Mouse)"
        else:
            uniprot_organism = "Homo sapiens (Human)"

        sequence = uniprot["Sequence"][(uniprot["Entry"] == uniprot_acc) & (uniprot["Organism"] == uniprot_organism)]
        try:
            sequence = sequence.values.tolist()[0]
        except IndexError:
            print(f"no match found for {uniprot_acc}")
            sequence = False
        return sequence

    def count_tryptic_peptides(sequence):
        """count tryptic peptides 6<=pep<=30 after cleavage """
        peptide_counter = 0
        # trypsin cuts after K and R, could be adjustet for different enzymes
        for peptide in sequence.split("K"):
            peptide = peptide + "K"
            pep = peptide.split("R")
            for p in pep:
                if len(p) > 0 and p[-1] != 'K':
                    p = p + "R"
                # peptide length exclusion
                if 6 <= len(p) <= 30:
                    peptide_counter = peptide_counter + 1
        return peptide_counter

    if protein_id is None:
        uniprot_acc = get_uniprot_accession(gene_name, organism)
    if get_seq == "online":
        sequence = get_uniprot_sequence(uniprot_acc)
    if get_seq == "offline":
        sequence = get_uniprot_sequence_locally(uniprot_acc, organism)
        if sequence == False:
            sequence = get_uniprot_sequence(uniprot_acc)

    iBAQ_pep_count = count_tryptic_peptides(sequence)
    iBAQ = intensity / iBAQ_pep_count

    return iBAQ


def get_subcellular_loc(series, database="compartments", loca=None, colname="Gene names"):
    """
    Annotate the df with subcellular localization.

    For compartments gene names are required.

    Parameters
    ----------
    series : pd.Series
        Must contain the colname to identify genes.
    database : str, optional
        Possible values are "compartments" and "hpa".
        The default is "compartments".
    loca : str, optional
        Only required for the compartments database.
        Filter the returned localisation table by this string.
        Must match exactly to the localisation terms in the compartments DB.
        The default is None.
    colname : str, optional
        Colname holding the gene names.
        The default is "Gene names".

    Raises
    ------
    ValueError
        Wrong value is provided for the database arg.

    Notes
    -----
    The compartments database is obtained from https://compartments.jensenlab.org/Downloads .
    The hpa database is the human protein atlas available at https://www.proteinatlas.org .

    Returns
    -------
    pd.DataFrame
        Dataframe with columns "ENSMBL", "Gene name", "LOCID", "LOCNAME", "SCORE"
        for compartments database.
    tuple of lists (main_loc, alt_loc)
        Lists of main and alternative localisations if the hpa database was chosen.

    Examples
    --------
    >>> series = pd.Series(['PEX14',], index=['Gene names'])

    Find all subcellular localisations of PEX14.
    The second line filters the returned dataframe so that only values with the
    highest score are retained. The dataframe is converted to list for better
    visualisation.

    >>> loc_df = autoprot.preprocessing.get_subcellular_loc(series)
    >>> sorted(loc_df.loc[loc_df[loc_df['SCORE'] == loc_df['SCORE'].max()].index,
    ...                   'LOCNAME'].tolist())
    ['Bounding membrane of organelle', 'Cellular anatomical entity', 'Cytoplasm', 'Intracellular', 'Intracellular membrane-bounded organelle', 'Intracellular organelle', 'Membrane', 'Microbody', 'Microbody membrane', 'Nucleus', 'Organelle', 'Organelle membrane', 'Peroxisomal membrane', 'Peroxisome', 'Whole membrane', 'cellular_component', 'membrane-bounded organelle', 'protein-containing complex']

    Get the score for PEX14 being peroxisomally localised

    >>> loc_df = autoprot.preprocessing.get_subcellular_loc(series, loca='Peroxisome')
    >>> loc_df['SCORE'].tolist()[0]
    5.0

    Using the Human Protein Atlas, a tuple of two lists containing the main and
    alternative localisations is returned

    >>> autoprot.preprocessing.get_subcellular_loc(series, database='hpa')
    (['Peroxisomes'], ['Nucleoli fibrillar center'])
    """
    gene = series[colname]
    if database == "compartments":
        with resources.open_binary("autoprot.data", "human_compartment_integrated_full.zip") as d:
            comp_data = pd.read_csv(d, sep='\t', compression='zip', header=None)
            comp_data.columns = ["ENSMBL", "Gene name", "LOCID", "LOCNAME", "SCORE"]
        if loca is None:
            # if loca is not provided, a table with all predicted localisations
            # is returned
            return comp_data[(comp_data["Gene name"] == gene)][["LOCNAME", "SCORE"]]
        # if loca is provided, only rows with the correspoding locname and score
        # are returned
        return comp_data[(comp_data["Gene name"] == gene) &
                         (comp_data["LOCNAME"] == loca)]
    elif database == "hpa":
        cols = "g,scl,scml,scal"
        # obtain protein atlas subset for the gene of interest
        html = requests.get(
            f"https://www.proteinatlas.org/api/search_download.php?search={gene}&format=json&columns={cols}&compress=no").text
        main_loc = html.split('Subcellular main location')[1].split(',"Subcellular additional location')[0].lstrip(
            '":[').split(',')
        alt_loc = html.split('Subcellular additional location')[1].split('}')[0].lstrip('":[').split(',')
        main_loc = [i.strip('"]') for i in main_loc]
        alt_loc = [i.strip('"]').rstrip('}') for i in alt_loc]
        return main_loc, alt_loc
    else:
        raise ValueError('Database can be either "compartments" or "hpa"')


def make_sim_score(m1, m2, corr="pearson"):
    """
    Calculate similarity score.

    To quantitatively describe the resemblance between the temporal profiles
    observed after subjecting the cells to the two treatments.
    Implemented as described in [1].

    Parameters
    ----------
    m1 : array-like
        Time course of SILAC ratios after treatment 1.
    m2 : array-like
        Time course of SILAC ratios after treatment 2.
    corr : str, optional
        Correlation parameter.
        'Pearson' or 'Spearman'. The default is "pearson".

    Returns
    -------
    float
        S-score that describes both the resemblance of the patterns of
        regulation and the resemblance between the degrees of regulation
        in the range from zero to infinity.

    Examples
    --------
    Similar temporal profiles result in high S-scores

    >>> s1 = [1,1,1,2,3,4,4]
    >>> s2 = [1,1,1,2,3,3,4]
    >>> autoprot.preprocessing.make_sim_score(s1, s2)
    50.97173553835997

    Low resemblance results in low scores

    >>> s2 = [1.1,1.1,1,1,1,1,1]
    >>> autoprot.preprocessing.make_sim_score(s1, s2)
    16.33374591446012

    References
    ----------
    [1] https://www.doi.org/10.1126/scisignal.2001570

    """

    def calc_magnitude(m1, m2):
        auca = auc(range(len(m1)), m1)
        aucb = auc(range(len(m2)), m2)
        # mcomp = np.divide(np.subtract(auca, aucb), np.add(auca, aucb))
        mcomp = (auca - aucb) / (auca + aucb)
        return abs(mcomp)

    def calc_corr(m1, m2, corr=corr):
        if corr == "pearson":
            r = pearsonr(m1, m2)[0]
        elif corr == "spearman":
            r = spearmanr(m1, m2)[0]
        else:
            raise ValueError('Invalid correlation parameter.')
        dof = len(m1) - 2
        t = (r * np.sqrt(dof)) / np.sqrt(1 - r ** 2)
        pval = stats.t.sf(np.abs(t), dof)
        return pval

    p_comp = calc_corr(m1, m2)
    m_comp = calc_magnitude(m1, m2)
    return -10 * np.log10(p_comp * m_comp)


def norm_to_prot(entry, prot_df, to_normalize):
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


def fetch_from_pride(accession, term, ignore_caps=True):
    """
    Get download links files belonging to a PRIDE identifier.

    Parameters
    ----------
    accession : str
        PRIDE identifier.
    term : str
        Part of the filename belonging to the project.
        For example 'proteingroups'
    ignore_caps : bool, optional
        Whether to ignore capitalisation during matching of terms.
        The default is True.

    Returns
    -------
    file_locs : dict
        Dict mapping filenames to FTP download links.

    Examples
    --------
    Generate a dict mapping file names to ftp download links.
    Not that only files containing the string proteingroups are retrieved.

    >>> ftpDict = pp.fetch_from_pride("PXD031829", 'proteingroups')

    """
    js_list = requests.get(f'https://www.ebi.ac.uk/pride/ws/archive/v2/files/byProject?accession={accession}',
                           headers={'Accept': 'application/json'}).json()

    file_locs = {}

    for fdict in js_list:
        fname = fdict['fileName']
        if ignore_caps is True:
            fname = fname.lower()
            term = term.lower()
        if term in fname:
            for protocol in fdict['publicFileLocations']:
                if protocol['name'] == 'FTP Protocol':
                    file_locs[fname] = protocol['value']
                    print(f'Found file {fname}')
    return file_locs


def download_from_ftp(url, save_dir, login_name='anonymous', login_pw=''):
    r"""
    Download a file from FTP.

    Parameters
    ----------
    url : TYPE
        DESCRIPTION.
    save_dir : TYPE
        DESCRIPTION.
    login_name : str
        Login name for the FTP server.
        Default is 'anonymous' working for the PRIDE FTP server.
    login_pw : str
        Password for access to the FTP server.
        Default is ''
    Returns
    -------
    str
        Path to the downloaded file.

    Examples
    --------
    Download all files from a dictionary holding file names and ftp links and
    save the paths to the downloaded files in a list.

    >>> downloadedFiles = []
    >>> for file in ftpDict.keys():
    ...     downloadedFiles.append(pp.download_from_ftp(ftpDict[file], r'C:\Users\jbender\Documents\python_playground'))

    """
    path, file = os.path.split(parse.urlparse(url).path)
    ftp = FTP(parse.urlparse(url).netloc)
    ftp.login(login_name, login_pw)
    ftp.cwd(path)
    ftp.retrbinary("RETR " + file, open(os.path.join(save_dir, file), 'wb').write)
    print(f'Downloaded {file}')
    ftp.quit()
    return os.path.join(save_dir, file)
