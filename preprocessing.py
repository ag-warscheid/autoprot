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
from subprocess import run, PIPE, CalledProcessError
from autoprot.decorators import report
from autoprot import RHelper
import requests
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from scipy.stats import pearsonr, spearmanr
from scipy import stats
from sklearn.metrics import auc

RSCRIPT, R = RHelper.returnRPath()

# =============================================================================
# Note: When using R functions provided column names might get changes
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
    df=df.copy() # make sure to keep the original dataframe unmodified
    columns = df.columns
    if file == "proteinGroups":
        if ("Potential contaminant" not in columns) or\
        ("Reverse" not in columns) or\
        ("Only identified by site" not in columns):
            print("Is this data already cleaned?\nMandatory columns for cleaning not present in data!")
            print("Returning provided dataframe!")
            return df
        df = df[(df['Potential contaminant'].isnull()) &
               (df['Reverse'].isnull()) &
               (df['Only identified by site'].isnull())]
        df.drop(['Potential contaminant',"Reverse", 'Only identified by site'], axis=1, inplace=True)
    elif (file == "Phospho (STY)") or (file == "evidence") or (file == "modificationSpecificPeptides"):
        if ("Potential contaminant" not in columns) or\
        ("Reverse" not in columns):
            print("Is this data already cleaned?\nMandatory columns for cleaning not present in data!")
            print("Returning provided dataframe!")
            return df
        df = df[(df['Potential contaminant'].isnull()) &
               (df['Reverse'].isnull())]
        df.drop(['Potential contaminant',"Reverse"], axis=1, inplace=True)
    elif file == "peptides":
        if ("Potential contaminant" not in columns) or\
        ("Reverse" not in columns):
            print("Is this data already cleaned?\nMandatory columns for cleaning not present in data!")
            print("Returning provided dataframe!")
            return df
        df = df[(df['Potential contaminant'].isnull()) &
               (df['Reverse'].isnull())]
        df.drop(['Potential contaminant',"Reverse"], axis=1, inplace=True)
    return df


def log(df, cols, base=2, invert=None, returnCols=False):
    r"""
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
    returnCols : bool, optional
        Whether to return a list of names corresponding to the columns added to the dataframe. The default is False.

    Returns
    -------
    pd.Dataframe
        The log transformed dataframe.

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
    df=df.copy() # make sure to keep the original dataframe unmodified
    # ignore divide by 0 errors
    with np.errstate(divide='ignore'):
        if base == 2:
            for c in cols:
                df[f"log2_{c}"] = np.log2(df[c])
        elif base==10:
            for c in cols:
                df[f"log10_{c}"] = np.log10(df[c])
        else:
            for c in cols:
                df[f"log{base}_{c}".format(base)] = np.log(df[c]) / np.log(base)
    newCols = [f"log{base}_{c}" for c in cols]

    if invert is not None:
        df[newCols] = df[newCols] * invert
    if returnCols == True:
        return df, newCols
    else:
        return df

@report
def filterLocProb(df, thresh=.75):
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
    The .filterLocProb() function filters a Phospho (STY)Sites.txt file.
    You can provide the desired threshold with the *thresh* parameter.
    
    >>> phos_filter = autoprot.preprocessing.filterLocProb(phos, thresh=.75)
    47936 rows before filter operation.
    33311 rows after filter operation.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.

    """
    df=df.copy() # make sure to keep the original dataframe unmodified
    if "Localization prob" not in df.columns:
        print("This dataframe has no 'Localization prob' column!")
        return True
    df = df[df["Localization prob"]>=thresh]
    return df

@report
def filterSeqCov(df, thresh, cols=None):
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
    df=df.copy() # make sure to keep the original dataframe unmodified
    if cols is not None:
        return df[(df[cols] >= thresh).all(1)]
    return df[df["Sequence coverage [%]"] >= thresh]

@report
def removeNonQuant(df, cols):
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
    >>> autoprot.preprocessing.removeNonQuant(df, cols=['a', 'b'])
    4 rows before filter operation.
    3 rows after filter operation.
         a    b    c
    0  1.0  4.0  NaN
    1  2.0  0.0  NaN
    3  4.0  1.0  1.0
    
    Rows are only removed if the all values in the specified columns are NaN.
    
    >>> autoprot.preprocessing.removeNonQuant(df, cols=['b', 'c'])
    4 rows before filter operation.
    4 rows after filter operation.
         a    b    c
    0  1.0  4.0  NaN
    1  2.0  0.0  NaN
    2  NaN  NaN  1.0
    3  4.0  1.0  1.0
    
    Example with real data.

    >>> phosRatio = phos.filter(regex="^Ratio .\/.( | normalized )R.___").columns
    >>> phosQuant = autoprot.preprocessing.removeNonQuant(phosLog, phosRatio)
    47936 rows before filter operation.
    39398 rows after filter operation.
    """
    df = df[~(df[cols].isnull().all(1))]
    return df


def expandSiteTable(df, cols):
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
    >>> phosRatio = phosLog.filter(regex="log2_Ratio .\/. normalized R.___")
    >>> phos_expanded = autoprot.preprocessing.expandSiteTable(phosLog, phosRatio)
    47936 phosphosites in dataframe.
    47903 phosphopeptides in dataframe after expansion.
    """
    df = df.copy(deep=True)
    print(f"{df.shape[0]} phosphosites in dataframe.")
    dfs = []
    expected = df.shape[0]*3
    #columns to melt
    melt = cols # e.g. "Ratio M/L normalized R2___1"
    melt_set = list(set([i[:-4] for i in melt])) # e.g. "Ratio M/L normalized R2"
    #Due to MaxQuant column names we might have to drop some columns
    check = [i in df.columns for i in melt_set] # check if the colnames are present in the df
    if False not in check:
        df.drop(melt_set, axis=1, inplace=True) # remove cols w/o ___n
    if True in check and False in check:
        raise ValueError("The columns you provided or the dataframe are not suitable!")

    # generate a separated melted df for every entry in the melt_set
    for i in melt_set:
        cs = list(df.filter(regex=i+'___').columns) + ["id"] # reconstruct the ___n cols for each melt_set entry
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
    temp = temp.drop(melt, axis=1) # drop the ___n entries from the original df

    for idx,df in enumerate(dfs):
        x = df["variable"].iloc[0].split('___')[0] # reconstructs the colname without ___n, e.g. Ratio M/L normalized R2
        if idx==0: # the first df contains all peptides and all multiplicities as rows
            t = df.copy(deep=True)
            t.columns = ["id", "Multiplicity", x] # e.g. 0, Ratio M/L normalized R2___1, 0.67391
            t["Multiplicity"] = t["Multiplicity"].apply(lambda x: x.split('___')[1]) # e.g. 0, 1, 0.67391
        else: # in the subsquent dfs id and multplicities can be dropped as only the ratio is new information compared to the first df
            df.columns = ["id", "Multiplicity", x]
            df = df.drop(["id", "Multiplicity"], axis=1) # keep only the x col
            t = t.join(df,rsuffix=idx) # horizontally joins the new col with the previous df
    # merging on ids gives the melted peptides their names back´
    temp = temp.merge(t,on='id', how='left')
    temp["Multiplicity"] = temp["Multiplicity"].astype(int) # is previously str

    if temp.shape[0] != expected:
        print("The expansion of site table is probably not correct!!! Check it! Maybe you provided wrong columns?")

    # remove rows that contain no modified peptides
    temp = temp[~(temp[melt_set].isnull().all(1))]
    print(f"{temp.shape[0]} phosphopeptides in dataframe after expansion.")
    return temp

@report
def filterVv(df, groups,n=2, vv=True):
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
    vv : bool, optional
        True for minimum amount of valid values; False for maximum amount of missing values. The default is True.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.

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
    >>> protFilter = autoprot.preprocessing.filterVv(protLog, groups=[a,b,c], n=3)
    4910 rows before filter operation.
    2674 rows after filter operation.
    """
    df=df.copy() # make sure to keep the original dataframe unmodified

    if vv == True:
        #TODO: Why not use notnull?
        # idxs = [set(df[df[group].notnull().sum(1)) >= n].index) for\
        #        group in groups]
        idxs = [set(df[(len(group)-df[group].isnull().sum(1)) >= n].index) for\
                group in groups]
    else:
        idxs = [set(df[df[group].isnull().sum(1) <= n].index) for\
               group in groups]

    # indices that are valid in all groups
    idx = set.intersection(*idxs)
    df = df.loc[idx]
    return df


def goAnnot(prots, gos, onlyProts=False, exact=True):
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
    onlyProts : bool, optional
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
    >>> go = autoprot.preprocessing.goAnnot(prot["Gene names"],gos, onlyProts=False)
    >>> go.head()
       index Gene names  GeneID       GO_ID   GO_term
    0   1944      RPS27    6232  GO:0005840  ribosome
    1   6451      RPS25    6230  GO:0005840  ribosome
    2   7640     RPL36A    6173  GO:0005840  ribosome
    3  11130      RRBP1    6238  GO:0005840  ribosome
    4  16112        SF1    7536  GO:0005840  ribosome
    """
    with resources.open_text("autoprot.data","Homo_sapiens.gene_info") as d:
        geneInfo = pd.read_csv(d, sep='\t')
    with resources.open_text("autoprot.data","gene2go_alt") as d:
        gene2go = pd.read_csv(d, sep='\t')
    # generate dataframe with single columns corresponding to experimental gene names
    prots = pd.DataFrame(pd.Series([str(i).upper().split(';')[0] for i in prots]), columns=["Gene names"])
    # add the column GeneID by merging with the geneInfo table
    prots = prots.merge(geneInfo[["Symbol", "GeneID"]], left_on="Gene names", right_on="Symbol", how='inner')
    # add the columns GO_ID and GO_term by merging on GeneID
    prots = prots.merge(gene2go[["GeneID", "GO_ID", "GO_term"]], on="GeneID", how='inner')
    if onlyProts == True:
        if exact == True:
            for idx, go in enumerate(gos):
                if idx == 0:
                    redProts = prots["Symbol"][prots["GO_term"]==go]
                else:
                    redProts = redProts.append(prots["Symbol"][prots["GO_term"]==go])
            return redProts.drop_duplicates().reset_index(drop=True)
        else:
            for idx, go in enumerate(gos):
                if idx == 0:
                    redProts = prots["Symbol"][prots["GO_term"].str.contains(go)]
                else:
                    redProts = redProts.append(prots["Symbol"][prots["GO_term"].str.contains(go)])
            return redProts.drop_duplicates().reset_index(drop=True)
    else:
        if exact == True:
            for idx, go in enumerate(gos):
                if idx == 0:
                    redProts = prots[prots["GO_term"]==go]
                else:
                    redProts = redProts.append(prots[prots["GO_term"]==go])
            return redProts.drop_duplicates().drop("Symbol", axis=1).reset_index()
        else:
            for idx, go in enumerate(gos):
                if idx == 0:
                    redProts = prots[prots["GO_term"].str.contains(go)]
                else:
                    redProts = redProts.append(prots[prots["GO_term"].str.contains(go)])
            return redProts.drop_duplicates().drop("Symbol", axis=1).reset_index(drop=True)


def motifAnnot(df, motif, col="Sequence window"):
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
    df=df.copy() # make sure to keep the original dataframe unmodified

    # TODO
    # make some assertions that the column is indeed the proper MQ output
    # (might want to customize the possibilites later)

    def findMotif(x,col, motif, motlen):
        seq = x[col]
        if ";" in seq:
            seqs = seq.split(';')
        else: seqs = [seq]
        for seq in seqs:
            pos = 0
            pos2 = re.finditer(motif,seq)
            if pos2:
                # iterate over re match objects
                for p in pos2:
                    # p.end() is the index of the last matching element of the searchstring
                    pos = p.end()
                    # only return a match if the motif in centred in the sequence window
                    # i.e. if the corresponding peptide was identified
                    if pos == np.floor(motlen/2+1):
                        return 1
        return 0

    assert(col in df.columns)
    assert(len(df[col].iloc[0]) % 2 == 1)

    # generate a regex string out of the input motif
    search = motif.replace('x', '.').replace('S/T', '(S|T)').upper()
    i = search.index("(S|T)")
    before = search[:i]
    after  = search[i+5:]
    # the regex contains a lookbehind (?<=SEQUENCEBEFORE), the actual modified residues (S/T)
    # and a lookahead with the following seqeunce for this motif (?=SEQUENCEAFTER)
    search = f"(?<={before})(S|T)(?={after})"
    # the lengths of the sequences in the sequence window column are all the same, take it from the first row
    motlen = len(df[col].iloc[0])
    df[motif] = df.apply(findMotif, col=col, motif=search, motlen=motlen, axis=1)

    return df


def impMinProb(df, colsToImpute, maxMissing=None, downshift=1.8, width=.3):
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
    colsToImpute : list
        Columns to impute. Should correspond to a single condition (i.e. control).
    maxMissing : int, optional
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
    >>> impProt = pp.impMinProb(forImp, phos.filter(regex="Int.*R1").columns,
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
        impProt = pp.impMinProb(forImp, phos.filter(regex="Int.*R1").columns, width=.4, downshift=2.5)
        fig, ax1 = plt.subplots(1)
        impProt.filter(regex="Int.*R1")[impProt["Imputed"]==False].mean(1).hist(density=True, bins=50, label="not Imputed", ax=ax1)
        impProt.filter(regex="Int.*R1")[impProt["Imputed"]==True].mean(1).hist(density=True, bins=50, label="Imputed", ax=ax1)
        plt.legend()
        plt.show()
    """
    df = df.copy(deep=True)

    if maxMissing is None:
        maxMissing = len(colsToImpute)
    # idxs of all rows in which imputation will be performed
    idx_noCtrl = df[df[colsToImpute].isnull().sum(1) >= maxMissing].index
    df["Imputed"] = False
    df.loc[idx_noCtrl,"Imputed"] = True

    for col in colsToImpute:
        mean = df[col].mean()
        var  = df[col].std()
        mean_ = mean - downshift*var
        var_ = var*width

        #generate random numbers matching the target dist
        rnd = np.random.normal(mean_, var_, size=len(idx_noCtrl))
        for i, idx in enumerate(idx_noCtrl):
            # ctrl so that really no data is overwritten
            if np.isnan(df.loc[idx, col]):
                df.loc[idx, col] = rnd[i]

    return df


def expSemiCol(df, scCol, newCol, castTo=None):
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

    #make temp df with expanded columns
    temp = df[scCol].str.split(";", expand=True)
    #use stack to bring it into long format series and directly convert back to df
    temp = pd.DataFrame(temp.stack(), columns=[newCol])
    #get first level of multiindex (corresponds to original index)
    temp.index = temp.index.get_level_values(0)
    #join on idex
    df = df.join(temp)

    if castTo is not None:
        df[newCol] = df[newCol].astype(castTo)

    return df


def mergeSemiCols(m1, m2, scCol1, scCol2=None):
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
    scCol1 : str
        Colname of a column containing semicolon-separated values in m1.
    scCol2 : str, optional
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

    #helper functions
    def _formMergePairs(s):
        """
        Group the data back on the main data identifier and create the appropiate matching entries of the other data.

        Parameters
        ----------
        s : pd.groupby
            Groupby object grouped on the first identifier.

        Returns
        -------
        pd.Series
            A Series with the idcs corresponding to m1 and the entries to the matching idcs in m2.

        """
        ids = list(set([i for i in s if not np.isnan(i)]))
        if len(ids) == 0:
            return [np.nan]
        return ids

    def _aggreagateDuplicates(s):
        # this might be a oversimplicfication but there should only be
        # object columns and numerical columns in the data

        if s.name == "mergeID_m1":
            print(s)

        if s.dtype != "O":
            return s.median()
        else:
            try:
                return s.mode()[0]
            except:
                return s.mode()

    m1, m2 = m1.copy(), m2.copy()

    #make IDs to reduce data after expansion
    m1["mergeID_m1"] = range(m1.shape[0])
    m2["mergeID_m2"] = range(m2.shape[0])

    # renome ssCol2 if it is not the same as ssCol1
    if scCol1 != scCol2:
        m2.rename(columns={scCol2:scCol1}, inplace=True)

    #expand the semicol columns and name the new col sUID
    m1Exp = expSemiCol(m1, scCol1,"sUID")
    m2Exp = expSemiCol(m2, scCol1, "sUID")

    # add the appropriate original row indices of m2 to the corresponding rows
    # of m1Exp
    # here one might want to consider other kind of merges
    merge = pd.merge(m1Exp, m2Exp[["mergeID_m2", "sUID"]], on="sUID", how='left')

    mergePairs = merge[["mergeID_m1", "mergeID_m2"]].groupby("mergeID_m1").agg(_formMergePairs)
    #This is neccessary if there are more than one matching columns
    mergePairs = mergePairs.explode("mergeID_m2")

    # merge of m2 columns
    mergePairs = (mergePairs
         .reset_index()
         .merge(m2, on="mergeID_m2", how="left")
         .groupby("mergeID_m1")
         .agg(_aggreagateDuplicates)
         .reset_index())

    # merge of m1 columns (those should all be unique)
    mergePairs = (mergePairs
                 .merge(m1, on="mergeID_m1", how="outer"))

    return mergePairs.drop(["mergeID_m1", "mergeID_m2"], axis=1)


def impSeq(df, cols):
    """
    Perform sequential imputation in R using impSeq from rrcovNA.

    See https://rdrr.io/cran/rrcovNA/man/impseq.html for a description of the
    algorithm.
    SEQimpute starts from a complete subset of the data set Xc and estimates sequentially the missing values in an incomplete observation, say x*, by minimizing the determinant of the covariance of the augmented data matrix X* = [Xc; x']. Then the observation x* is added to the complete data matrix and the algorithm continues with the next observation with missing values.

    Parameters
    ----------
    df : pd.dataframe
        Input dataframe.
    cols : list of str
        Colnames to perform imputation of.

    Returns
    -------
    df : pd.dataframe
        Dataframe with imputed values.
        Cols with imputed values are names _imputed.
        Contains a col UID that was used for processing.

    """
    d = os.getcwd()
    dataLoc = d + "/input.csv"
    outputLoc = d + "/output.csv"

    if "UID" in df.columns:
        pass
    else:
        # UID is basically a row index starting at 1
        df["UID"] = range(1, df.shape[0]+1)

    if not isinstance(cols, list):
        cols = cols.to_list()
    to_csv(df[["UID"] + cols], dataLoc)

    command = [R, '--vanilla',
    RSCRIPT, #script location
    "impSeq", #functionName
    dataLoc, #data location
    outputLoc #output file
    ]
    run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)

    res = read_csv(outputLoc)
    resCols = [f"{i}_imputed" if i != "UID" else i for i in res.columns]
    res.columns = resCols

    df = df.merge(res, on="UID")

    os.remove(dataLoc)
    os.remove(outputLoc)

    return df


def quantileNorm(df, cols, returnCols=False, backend="r"):
    r"""
    Perform quantile normalization.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    cols : list of str
        Colnames to perform normlisation on.
    returnCols : bool, optional
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
    
    >>> phos_norm_r = pp.quantileNorm(phosLog, noNorm, backend='r')
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
        phos_norm_r = pp.quantileNorm(phosLog, noNorm, backend='r')
        vis.boxplot(phos_norm_r, [noNorm, phos_norm_r.filter(regex="_norm").columns], compare=True)
        plt.show()

    """
    if "UID" in df.columns:
        pass
    else:
        df["UID"] = range(1, df.shape[0]+1)

    if not isinstance(cols, list):
        cols = cols.to_list()

    #TODO: Check why python backend fails so poorly
    # See https://github.com/bmbolstad/preprocessCore/blob/master/R/normalize.quantiles.R
    if backend == "py":
        subDf = df[cols+["UID"]].copy()
        idx = subDf["UID"].values
        subDf = subDf.drop("UID", axis=1)
        subDf.index = idx
        #use numpy sort to sort columns independently
        subDf_sorted = pd.DataFrame(np.sort(subDf.values, axis=0), index=subDf.index, columns = subDf.columns)
        subDf_mean = subDf_sorted.mean(axis=1)
        subDf_mean.index = np.arange(1, len(subDf_mean)+1)
        # Assign ranks across the cols, stack the cols so that a multiIndex series
        # is created, map the subDf_mean series on the series and unstack again
        df_norm = subDf.rank(axis=0, method="min").stack().astype(int).map(subDf_mean).unstack()
        resCols = [f"{i}_normalized" for i in df_norm.columns]
        df_norm.columns = resCols
        df_norm["UID"] = df_norm.index
        print(df_norm)
        df = df.merge(df_norm, on="UID", how="left")

    elif backend == "r":
        d = os.getcwd()
        dataLoc = d + "/input.csv"
        outputLoc = d + "/output.csv"

        to_csv(df[["UID"] + cols], dataLoc)

        command = [R, '--vanilla',
        RSCRIPT, #script location
        "quantile", #functionName
        dataLoc, #data location
        outputLoc #output file
        ]

        try:
            run(command, stdout=PIPE, check=True, stderr=PIPE, universal_newlines=True)
        except CalledProcessError as err:
            raise Exception(f'Error during execution of R function:\n{err.stderr}')

        res = read_csv(outputLoc)
        resCols = [f"{i}_normalized" if i != "UID" else i for i in res.columns]
        res.columns = resCols
        df = df.merge(res, on="UID")

        os.remove(dataLoc)
        os.remove(outputLoc)

    else:
        raise(Exception('Please supply either "r" or "py" as value for the backend arg'))

    if returnCols == True:
        return df, [i for i in resCols if i != "UID"]
    return df


def vsn(df, cols, returnCols=False, backend='r'):
    r"""
    Perform Variance Stabilizing Normalization.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    cols : list of str
        Colnames to perform normlisation on.
    returnCols : bool, optional
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
    [1] Huber, W, von Heydebreck, A, Sueltmann, H, Poustka, A, Vingron, M (2002). Variance stabilization applied to microarray data calibration and to the quantification of differential expression. Bioinformatics 18 Supplement 1, S96-S104.

    Notes
    -----
    The Vsn is a statistical method aiming at making the sample
    variances nondependent from their mean intensities and bringing the
    samples onto a same scale with a set of parametric transformations
    and maximum likelihood estimation.

    Examples
    --------
    >>> import autoprot.preprocessing as pp
    >>> import autoprot.visualization as vis
    >>> import pandas as pd
    >>> phos_lfq = pd.read_csv("_static/testdata/Phospho (STY)Sites_lfq.zip", sep="\t", low_memory=False)
    >>> noNorm = phos_lfq.filter(regex="Intensity .").columns
    >>> phos_lfq[noNorm] = phos_lfq.filter(regex="Intensity .").replace(0, np.nan)
    
    Until now this was only preprocessing for the normalisation.
    Note that we are treating LFQ pre-normalised values with VSN normalisation.
    
    >>> phos_lfq = pp.vsn(phos_lfq, noNorm)
    >>> vis.boxplot(phos_lfq, [noNorm, phos_lfq.filter(regex="_norm").columns], data='Intensity', compare=True)
    >>> plt.show() #doctest: +SKIP
    
    .. plot::
        :context: close-figs
    
        import autoprot.preprocessing as pp
        import autoprot.visualization as vis
        import pandas as pd
        phos_lfq = pd.read_csv("_static/testdata/Phospho (STY)Sites_lfq.zip", sep="\t", low_memory=False)
        noNorm = phos_lfq.filter(regex="Intensity .").columns
        phos_lfq[noNorm] = phos_lfq.filter(regex="Intensity .").replace(0, np.nan)
        phos_lfq = pp.vsn(phos_lfq, noNorm)
        vis.boxplot(phos_lfq, [noNorm, phos_lfq.filter(regex="_norm").columns], data='Intensity', compare=True)
    """
    d = os.getcwd()
    dataLoc = d + "/input.csv"
    outputLoc = d + "/output.csv"

    if "UID" in df.columns:
        pass
    else:
        df["UID"] = range(1, df.shape[0]+1)

    if not isinstance(cols, list):
        cols = cols.to_list()
    to_csv(df[["UID"] + cols], dataLoc)

    command = [R, '--vanilla',
    RSCRIPT, #script location
    "vsn", #functionName
    dataLoc, #data location
    outputLoc #output file
    ]

    try:
        run(command, stdout=PIPE, check=True, stderr=PIPE, universal_newlines=True)
    except CalledProcessError as err:
        raise Exception(f'Error during execution of R function:\n{err.stderr}')

    res = read_csv(outputLoc)
    resCols = [f"{i}_normalized" if i != "UID" else i for i in res.columns]
    res.columns = resCols

    df = df.merge(res, on="UID")

    os.remove(dataLoc)
    os.remove(outputLoc)

    if returnCols == True:
        return df, [i for i in resCols if i != "UID"]
    return df


def cyclicLOESS(df, cols, backend='r'):
    r"""
    Perform cyclic Loess normalization.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    cols : list of str
        Colnames to perform normlisation on.
    returnCols : bool, optional
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
    
    >>> phos_norm_r = pp.cyclicLOESS(phosLog, noNorm, backend='r')
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
        phos_norm_r = pp.cyclicLOESS(phosLog, noNorm, backend='r')
        vis.boxplot(phos_norm_r, [noNorm, phos_norm_r.filter(regex="_norm").columns], compare=True)
        plt.show()

    """
    d = os.getcwd()
    dataLoc = d + "/input.csv"
    outputLoc = d + "/output.csv"

    if "UID" in df.columns:
        pass
    else:
        df["UID"] = range(1, df.shape[0]+1)

    if not isinstance(cols, list):
        cols = cols.to_list()
    to_csv(df[["UID"] + cols], dataLoc)

    command = [R, '--vanilla',
    RSCRIPT, #script location
    "cloess", #functionName
    dataLoc, #data location
    outputLoc #output file
    ]

    run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)

    res = read_csv(outputLoc)
    resCols = [f"{i}_normalized" if i != "UID" else i for i in res.columns]
    res.columns = resCols

    df = df.merge(res, on="UID")

    os.remove(dataLoc)
    os.remove(outputLoc)

    return df

def toCanonicalPS(series, organism="human"):
    """
    Convert phosphosites to "canonical" phosphosites.

    Parameters
    ----------
    series : pd.Series
        Series containing the indices "Gene names" and "Sequence Window".
        Corresponds e.g. to MQ Phospho(STY)Sites.txt.
    organism : str, optional
        This conversion is based on Uniprot Identifier used in PSP data.
        possible organisms: 'mouse', 'human', 'rat', 'sheep', 'SARSCoV2', 'guinea pig', 'cow',
        'hamster', 'fruit fly', 'dog', 'rabbit', 'pig', 'chicken', 'frog',
        'quail', 'horse', 'goat', 'papillomavirus', 'water buffalo',
        'marmoset', 'turkey', 'cat', 'starfish', 'torpedo', 'SARSCoV1',
        'green monkey', 'ferret'. The default is "human".

    Notes
    -----
    This function compares a certain gene name to the genes found in the
    phosphosite plus (https://www.phosphosite.org) phosphorylation site dataset.

    Returns
    -------
    tuple of (str, int)
        (UniProt ID, Position of phosphosite in the UniProt sequence)

    Examples
    --------
    The correct position of the phosphorylation is returned independent of the
    completeness of the sequence window.
    
    >>> series=pd.Series(['PEX14', "VSNESTSSSPGKEGHSPEGSTVTYHLLGPQE"], index=['Gene names', 'Sequence window'])
    >>> autoprot.preprocessing.toCanonicalPS(series, organism='human')
    ('O75381', 282)
    >>> series=pd.Series(['PEX14', "_____TSSSPGKEGHSPEGSTVTYHLLGP__"], index=['Gene names', 'Sequence window'])
    >>> autoprot.preprocessing.toCanonicalPS(series, organism='human')
    ('O75381', 282)
    """
    # open the phosphosite plus phosphorylation dataset
    with resources.open_text('autoprot.data',"phosphorylation_site_dataset") as d:
                ps = pd.read_csv(d, sep='\t')

    def getUPAcc(gene, organism):
        """Find the matching UniProt ID in the phosphorylation_site_dataset given a gene name and a corresponding organism."""
        gene = gene.upper()

        try:
            upacc = ps.loc[(ps["GENE"].apply(lambda x: str(x).upper()==gene)) &
                           (ps["ORGANISM"]==organism), "ACC_ID"].iloc[0]
            return upacc
        except:
            return "notFound"

    def getUPSeq(upacc):
        """Download sequence from uniprot by UniProt ID."""
        url = f"https://www.uniprot.org/uniprot/{upacc}.fasta"
        response = requests.get(url)
        seq = ("").join(response.text.split('\n')[1:])
        return seq

    def getCanonicalPos(seq, psSeq, n):
        """Align an experimental phosphosite sequence window to the corresponding UniProt sequence."""
        alignment = pairwise2.align.localms(sequenceA=seq,
                                            sequenceB=psSeq,
                                            match=2,# match score
                                            mismatch=-2, # mismatch penalty
                                            open=-1, # open penalty
                                            extend=-1) # exted penalty
        # Format the alignment e.g.
        # '1 ACCG\n  | ||\n1 A-CG\n  Score=5\n'
        formAlign = format_alignment(*alignment[0])
        # get first position of alignment of seq i.e. in the UniProt sequence
        start = int(formAlign.lstrip(' ').split(' ')[0])
        # get first position of the alignment of psSeq
        start2 = int(formAlign.split('\n')[2].lstrip(' ').split(' ')[0]) - 1
        # the position of the phosphosite in the UniProt Sequence is the first
        # position of the alignment in seq minus the position of the alignment
        # in psSeq (which is usually 1) minus the number of blanks in the sequence
        # window plus 15 which is half the sequence for a sequence window length of 31
        canPos = start + ((15-n)-start2)
        return canPos

    gene = str(series["Gene names"])
    # get uniprot acc
    # counter used if first gene name not found
    # in that case appropiate seq window has to be used
    counter = 0
    if ';' in gene:
        for g in gene.split(';'):
            upacc = getUPAcc(g, organism)
            if upacc != "notFound":
                continue
            counter += 1
    else:
        upacc = getUPAcc(gene, organism)
    if upacc == "notFound":
        return "No matching phosphosite found"

    #get sequence
    seq = getUPSeq(upacc)
    # get the sequence from the phosphosite annotation
    psSeq = series["Sequence window"]
    # if multiple proteins were matched to the phosphosite sequence
    # take only the sequence corresponding to the last match
    if ';' in psSeq:
        psSeq = psSeq.split(';')[counter]

    # n = number of blanks in the sequence window
    n = len(psSeq) - len(psSeq.lstrip('_'))
    # clean the experimental sequence from preceding and trailing blanks
    psSeq = psSeq.strip('_')

    return upacc,getCanonicalPos(seq, psSeq, n)


def getSubCellLoc(series, database="compartments", loca=None, colname="Gene names"):
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
    tuple of lists (mainLoc, altLoc)
        Lists of main and alternative localisations if the hpa database was chosen.
    
    Examples
    --------
    >>> series = pd.Series(['PEX14',], index=['Gene names'])
    
    Find all subcellular localisations of PEX14.
    The second line filters the returned dataframe so that only values with the
    highest score are retained. The dataframe is converted to list for better
    visualisation.
    
    >>> loc_df = autoprot.preprocessing.getSubCellLoc(series)    
    >>> sorted(loc_df.loc[loc_df[loc_df['SCORE'] == loc_df['SCORE'].max()].index,
    ...                   'LOCNAME'].tolist())
    ['Bounding membrane of organelle', 'Cellular anatomical entity', 'Cytoplasm', 'Intracellular', 'Intracellular membrane-bounded organelle', 'Intracellular organelle', 'Membrane', 'Microbody', 'Microbody membrane', 'Nucleus', 'Organelle', 'Organelle membrane', 'Peroxisomal membrane', 'Peroxisome', 'Whole membrane', 'cellular_component', 'membrane-bounded organelle', 'protein-containing complex']
    
    Get the score for PEX14 being peroxisomally localised
    
    >>> loc_df = autoprot.preprocessing.getSubCellLoc(series, loca='Peroxisome')
    >>> loc_df['SCORE'].tolist()[0]
    5.0
    
    Using the Human Protein Atlas, a tuple of two lists containing the main and
    alternative localisations is returned
    
    >>> autoprot.preprocessing.getSubCellLoc(series, database='hpa')
    (['Peroxisomes'], ['Nucleoli fibrillar center'])
    """
    gene = series[colname]
    if database == "compartments":
        with resources.open_text("autoprot.data","human_compartment_integrated_full") as d:
            compData = pd.read_csv(d, sep='\t', header=None)
            compData.columns = ["ENSMBL", "Gene name", "LOCID", "LOCNAME", "SCORE"]
        if loca is None:
            # if loca is not provided, a table with all predicted localisations
            # is returned
            return compData[(compData["Gene name"]==gene)][["LOCNAME", "SCORE"]]
        # if loca is provided, only rows with the correspoding locname and score
        # are returned
        return compData[(compData["Gene name"]==gene) &
                (compData["LOCNAME"]==loca)]
    elif database == "hpa":
        cols = "g,scl,scml,scal"
        # obtain protein atlas subset for the gene of interest
        html = requests.get(f"https://www.proteinatlas.org/api/search_download.php?search={gene}&format=json&columns={cols}&compress=no").text
        mainLoc = html.split('Subcellular main location')[1].split(',"Subcellular additional location')[0].lstrip('":[').split(',')
        altLoc  = html.split('Subcellular additional location')[1].split('}')[0].lstrip('":[').split(',')
        mainLoc = [i.strip('"]') for i in mainLoc]
        altLoc  = [i.strip('"]').rstrip('}') for i in altLoc]
        return mainLoc, altLoc
    else:
        raise ValueError('Database can be either "compartments" or "hpa"')

def makeSimScore(m1, m2, corr="pearson"):
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
    >>> autoprot.preprocessing.makeSimScore(s1, s2)
    50.97173553835997

    Low resemblance results in low scores
    
    >>> s2 = [1.1,1.1,1,1,1,1,1]
    >>> autoprot.preprocessing.makeSimScore(s1, s2)
    16.33374591446012

    References
    ----------
    [1] https://www.doi.org/10.1126/scisignal.2001570

    """
    def calcMagnitude(m1,m2):
        auca = auc(range(len(m1)), m1)
        aucb = auc(range(len(m2)), m2)
        #mcomp = np.divide(np.subtract(auca, aucb), np.add(auca, aucb))
        mcomp = (auca-aucb)/(auca+aucb)
        return abs(mcomp)

    def calcCorr(m1,m2, corr=corr):
        if corr == "pearson":
            r = pearsonr(m1, m2)[0]
        elif corr == "spearman":
            r = spearmanr(m1, m2)[0]
        else:
            raise ValueError('Invalid correlation parameter.')
        dof = len(m1) - 2
        t = (r*np.sqrt(dof))/np.sqrt(1-r**2)
        pval = stats.t.sf(np.abs(t), dof)
        return pval

    pComp = calcCorr(m1,m2)
    mComp = calcMagnitude(m1,m2)
    return -10*np.log10(pComp*mComp)

def normToProt(entry, protDf, toNormalize):
    """
    Normalize phospho data to total protein level.

    Function has to be applied to phosphosite table.
    e.g. phosTable.apply(lambda x: normToProt(x, dfProt, toNormalize),1)

    Parameters
    ----------
    entry : pd.Series
        Row-like object with index "Protein group IDs".
    protDf : pd.DataFrame
        MQ ProteinGroups data to which data is normalized.
    toNormalize : list of str
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
        protIds = entry["Protein group IDs"]
    except:
        raise ValueError('The input array does not contain an index "Protein group IDs"')
    if ';' in protIds:
        protIds = [int(i) for i in protIds.split(';')]
        protDf = protDf[protDf["id"].isin(protIds)]
        poi = protDf.groupby("Gene names").median()
    else:
        # generate subset of protDf matching the ID of the current protein
        poi = protDf[protDf["id"]==int(protIds)]
    if poi.shape[0] == 0:
        #can*t normalize. either return non-normalized or drop value?
        corrected = pd.DataFrame([np.nan]*len(toNormalize)).T
    else:
        # log(pep) - log(prot) or log(pep/prot)
        # TODO Does this work? isnt poi[toNormalize] a df and entry a series?
        corrected = entry[toNormalize] - poi[toNormalize]
    return pd.Series(corrected.values[0])