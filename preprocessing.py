# -*- coding: utf-8 -*-
"""

Created on Mon Jul  8 09:26:07 2019

@author: Wignand

DataProcessing

:function cleaning: for first processing of dataframe ratio cols
"""

import numpy as np
import pandas as pd
from importlib import resources
import re
import os
from subprocess import run, PIPE
from autoprot.decorators import report
from autoprot import RHelper
import requests
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from scipy.stats import pearsonr, spearmanr
from scipy import stats
from sklearn.metrics import auc


RSCRIPT, R = RHelper.returnRPath()
with resources.open_text("autoprot.data","Phosphorylation_site_dataset") as d:
            ps = pd.read_csv(d, sep='\t')

"""
Note: When using R functions provided column names might get changes
Especially, do not use +,- or spaces in your column names. Maybe write decorator to
validate proper column formatting and handle exceptions
"""



def read_csv(file, sep='\t', low_memory=False, **kwargs):
    return pd.read_csv(file, sep=sep, **kwargs)
    
    
def to_csv(df, file, sep='\t', index=False, **kwargs):
    df.to_csv(file, sep=sep, index=index, **kwargs)

@report
def cleaning(df, file="proteinGroups"):
    """
    removes contaminant, reverse and identified by site only entries
    @file:: which file is provided:
        proteinGroups; Phospho (STY); evidence; 
        modificationSpecificPeptides 
    """
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
    """
    performs log transformation. Returns dataframe with additional log columns
    @params
    ::cols: cols which are transformed
    ::base: base of log, default=2, alternative: 10
    ::invert: vector corresponding to columns telling which to invert
    """
    if base == 2:
        for c in cols:
            df[f"log2_{c}"] = np.log2(df[c])
        newCols = [f"log2_{c}" for c in cols]
    elif base==10:
        for c in cols:
            df[f"log10_{c}"] = np.log10(df[c])
            newCols = [f"log10_{c}" for c in cols]
    else:
        print("This base is not implemented!")
    if invert is not None:
        lcols = [f"log2_{i}" for i in cols]
        df[lcols] = df[lcols] * invert
    if returnCols == True:
        return df, newCols
    else:
        return df


def filterLocProb(df, thresh=.75):
    """
    removes entries with localiatoin probabiliy below threshold
    @params
    @df :: dataframe to be filtered
    @thresh :: threshold of localization probability
    """
    if "Localization prob" not in df.columns:
        print("This dataframe has no 'Localization prob' column!")
        return True
    print(f"{df.shape[0]} entries in dataframe.")
    df = df[df["Localization prob"]>=thresh]
    print(f"{df.shape[0]} entries in dataframe with localization prob >= {thresh*100}%.")
    return df


    
    
@report
def filterSeqCov(df, thresh, cols=None):
    """
    filters MQ total data based on sequence coverage [%]
    """
    if cols is not None:
        return df[(df[cols] >= thresh).all(1)]
    return df[df["Sequence coverage [%]"] >= thresh]


@report
def removeNonQuant(df, cols):
    """
    removes entries without quantitative data
    @params
    @df :: dataframe to be filtered
    @cols :: cols to be evaluated for missingness
    """
    df = df[~(df[cols].isnull().all(1))]
    return df


def expandSiteTable(df, cols):
    """
    function that expands the phosphosite table Sites -> peptides
    x, a__1, a__2, a__3
    ->
    x, a, 1
    x, a, 2
    x, a, 3
    @params
    @df :: dataframe to be expanded (important that an "id" column is provided)
    @cols :: cols which are going to be expanded (format: Ratio.*___.)
    """
    df = df.copy(deep=True)
    print(f"{df.shape[0]} phosphosites in dataframe.")
    dfs = []
    expected = df.shape[0]*3
    #columns to melt
    melt = cols
    melt_set = list(set([i[:-4] for i in melt]))
    #Due to MaxQuant column names we might have to drop some columns
    check = [i in df.columns for i in melt_set]
    if False not in check:
        df.drop(melt_set, axis=1, inplace=True)
    if True in check and False in check:
        print("Your provided columns ")
        raise ValueError("The columns you provided are not suitable!")
    for i in melt_set:
        cs = list(df.filter(regex=i+'___').columns )+ ["id"]
        dfs.append(pd.melt(df[cs], id_vars='id'))
    temp = df.copy(deep=True)
    temp = temp.drop(melt, axis=1)
    
    for idx,df in enumerate(dfs):
        x = df["variable"].iloc[0].split('___')[0]
        if idx==0:
            t = df.copy(deep=True)
            t.columns = ["id", "Multiplicity", x]
            t["Multiplicity"] = t["Multiplicity"].apply(lambda x: x.split('___')[1])
        else:
            df.columns = ["id", "Multiplicity", x]
            df = df.drop(["id", "Multiplicity"], axis=1)
            t = t.join(df,rsuffix=idx)
    temp = temp.merge(t,on='id', how='left')
    if temp.shape[0] != expected:
        print("The expansion of site table is probably not correct!!! Check it! Maybe you provided wrong columns?")
    temp = temp[~(temp[melt_set].isnull().all(1))]
    print(f"{temp.shape[0]} phosphopeptides in dataframe after expansion.")
    temp["Multiplicity"] = temp["Multiplicity"].astype(int)
    return temp

@report
def filterVv(df, groups,n=2, vv=True):
    """
....function that filters dataframe for minimum number of valid values
....@params
    df :: dataframe to be filtered - copy is returned
    groups :: the experimental groups. Each group is filtered for at least n vv
    n :: minimum amount of valid values
    vv :: True for minimum amount of valid values; False for maximum amount of missing values
...."""
    if vv == True:
        idxs = [set(df[(len(group)-df[group].isnull().sum(1)) >= n].index) for\
               group in groups]
    else:
        idxs =  [set(df[df[group].isnull().sum(1) <= n].index) for\
               group in groups]

    #take intersection of idxs
    idx = set.intersection(*idxs)
    df = df.loc[idx]
    return df


def goAnnot(prots, gos, onlyProts=False, exact=True):
    """
    function that finds kinases based on go annoation in 
    list of gene names. If there are multiple gene names separated by semicolons
    only the first entry will be used.
    :@Prots: List of Gene names
    :@go: List of go terms
    :@onlyProts: Boolean, whether to return dataframe or only list of gene names annotated with terms
    :@exact: Boolean, whether go term must match exactly. i.e. MAPK activity <-> regulation of MAPK acitivity etc
    Notes:
        Homo sapiens.gene_info and gene2go files 
        are needed for annotation
        
        In case of multiple gene names per line (e.g. AKT1;PKB)
        only the first name will be extracted.
        
    """
    with resources.open_text("autoprot.data","Homo_sapiens.gene_info") as d:
        geneInfo = pd.read_csv(d, sep='\t')
    with resources.open_text("autoprot.data","gene2go_alt") as d:
        gene2go = pd.read_csv(d, sep='\t')
    prots = pd.DataFrame(pd.Series([str(i).upper().split(';')[0] for i in prots]), columns=["Gene names"])
    prots = prots.merge(geneInfo[["Symbol", "GeneID"]], left_on="Gene names", right_on="Symbol", how='inner')
    
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


def motifAnnot(df, motif, col=None):
    """
    Function that searches for phosphorylation motif in the provided dataframe.
    If not specified "Sequence window" column is searched. Phosphorylated central residue
    has to indicated with S/T, arbitrary amino acids with x. 
    Examples:
    - RxRxxS/T
    - PxS/TP
    - RxRxxS/TxSxxR

    :@df: dataframe
    :@motif: str; motif to be searched for
    :@col: str; alternative column to be searched in if Sequence window is not desired
    
    return boolean column
    """

    #make some assertions that the column is indeed the proper MQ output
    #(might want to customize the possibilites later)
    
    def findMotif(x,col, motif, motlen):
        seq = x[col]
        if ";" in seq:
            seqs = seq.split(';')
        else: seqs = [seq]
        for seq in seqs:
            pos = 0
            pos2 = re.finditer(motif,seq)
            if pos2:
                for p in pos2:
                    pos = p.end()
                    if pos == np.floor(motlen/2+1):
                        return 1
        return 0
    
    if col is None:
        col = "Sequence window"
    
    assert(col in df.columns)
    assert(len(df[col].iloc[0]) % 2 == 1)
    
    
    
    search = motif.replace('x', '.').replace('S/T', '(S|T)').upper()
    i = search.index("(S|T)")
    before = search[:i]
    after  = search[i+5:]
    search = f"(?<={before})(S|T)(?={after})"
    motlen = len(df[col].iloc[0])
    df[motif] = df.apply(findMotif, col=col, motif=search, motlen=motlen, axis=1)
    
    return df


def impMinProb(df, colsToImpute, maxMissing=None, downshift=1.8, width=.3):

    """
    Function that performs an imputation by modeling a distribution on the far left site of actual distribution
    This will be mean shifted and has a smaller variation. Intensities should be log transformed. 
    Downsshift: mean - downshift*sigma
    Var: width*sigma
    @params
    :df: dataframe; dataframe on which imputation is performed
    :colsToImpute; list; columns to impute, these should correspond to one condition (i.e. control)
    :maxMissing; int; how many missing values to perfom imputation; if None all values have to be missing
    :downshift; how far to the left the mean of the new population is shifted
    
    """

    def shift(s, mean, var):
        return np.random.normal(mean, var)


    df = df.copy(deep=True)

    if maxMissing is None:
        maxMissing = len(colsToImpute)
    idx_noCtrl = df[df[colsToImpute].isnull().sum(1) >= maxMissing].index
    df["Imputed"] = False
    df.loc[idx_noCtrl,"Imputed"] = True
    
    for col in colsToImpute:
        mean = df[col].mean()
        var  = df[col].std()
        mean_ = mean - downshift*var
        var_ = var*width
        
        #generate random numbers
        rnd = np.random.normal(mean_, var_, size=len(idx_noCtrl))
        for i, idx in enumerate(idx_noCtrl):
            if np.isnan(df.loc[idx, col]):
                df.loc[idx, col] = rnd[i]
     
    return df


def expSemiCol(df, scCol, newCol, castTo=None):
    """
    Function that expands a semicolon containing string column
    and generates a new column based on its content
    @params:
    scCol: string; The given column has to be string dtype and contain semicolon separated values
    e.g. string1;string2;...;stringn
    newCol: string; A new columns named "newCol" will be generated
    castTo: type; if provided new column will be set to the provided type
    """

    df = df.copy(deep=True)
    df = df.reset_index(drop=True)

    #make temp df with expanded scan numbers
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
    Function to merge two matrices on a semicolon separated column.
    Here m2 is merged to m1 (left merge).
    -> entries in m2 which are not matched to m1 are dropped
    if sCol2 is not provided it is assumed to be the same as sCol1
    """

    #helper functions
    def formMergePairs(s):
        """
        This function groups the data back on the main data identifier and 
        creates the appropiate matching entries of the other data
        """
        ids = list(set([i for i in s if not np.isnan(i)]))
        if len(ids) == 0:
            return [np.nan]
        return ids
    
    def aggreagateDuplicates(s):
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

    if scCol1 != scCol2:
        m2.rename(columns={scCol2:scCol1}, inplace=True)
    
    #expand the uniprot ids (can also be other semicol columns)
    m1Exp = expSemiCol(m1, scCol1,"sUID")
    m2Exp = expSemiCol(m2, scCol1, "sUID")

    #here one might want to consider other kind of merges
    merge = pd.merge(m1Exp, m2Exp[["mergeID_m2", "sUID"]], on="sUID", how='left')

    mergePairs = merge[["mergeID_m1", "mergeID_m2"]].groupby("mergeID_m1").agg(formMergePairs)
    #This is neccessary if there are more than one matching columns
    mergePairs = mergePairs.explode("mergeID_m2")
        
    # merge of m2 columns 
    mergePairs = (mergePairs
         .reset_index()
         .merge(m2, on="mergeID_m2", how="left")
         .groupby("mergeID_m1")
         .agg(aggreagateDuplicates)
         .reset_index())
       
    # merge of m1 columns (those should all be unique)
    mergePairs = (mergePairs
                 .merge(m1, on="mergeID_m1", how="outer"))
    
    return mergePairs.drop(["mergeID_m1", "mergeID_m2"], axis=1)


def impSeq(df, cols):
    """
    Function that performs sequentiel imputaion in R using impSeq() fnction from rrcovNA
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


def quantileNorm(df, cols, returnCols=False, backup="r"):
    """
    Function that performs quantile normalization
    :@returnCols: boolean, if True also the column names of the normalized columns are returned
    :@backup: string, py or r, whether to use R or Python implentation of quantile normalization
    
    NOTE:
    While the python implementation is much faster than r (since R is executed in a subroutine), the
    R Function handles NaNs in a more sophisticated manner than the python function (which just ignores NaNs)
    """
    
        
    if "UID" in df.columns:
        pass
    else: 
        df["UID"] = range(1, df.shape[0]+1)
        
    if not isinstance(cols, list):
        cols = cols.to_list()
    
    if backup == "py":
        subDf = df[cols+["UID"]].copy()
        idx = subDf["UID"].values
        subDf = subDf.drop("UID", axis=1)
        subDf.index = idx
        #use numpy sort to sort columns independently
        subDf_sorted = pd.DataFrame(np.sort(subDf.values, axis=0), index=subDf.index, columns = subDf.columns)
        subDf_mean = subDf_sorted.mean(1)
        subDf_meanIndex = np.arange(1, len(subDf_mean)+1)
        df_norm = subDf.rank(method="min").stack().astype(int).map(subDf_mean).unstack()
        resCols = [f"{i}_normalized" for i in df_norm.columns]
        df_norm.columns = resCols
        df_norm["UID"] = df_norm.index
        df = df.merge(df_norm, on="UID", how="left")
        
    elif backup == "r":
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
        run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)
        
        res = read_csv(outputLoc)
        resCols = [f"{i}_normalized" if i != "UID" else i for i in res.columns]
        res.columns = resCols
        df = df.merge(res, on="UID")
        
        os.remove(dataLoc)
        os.remove(outputLoc)
        
    if returnCols == True:
        return df, [i for i in resCols if i != "UID"]
    return df


def vsn(df, cols, returnCols=False):
    """
    Function that performs vsn normalization
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
    
    run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    
    
    res = read_csv(outputLoc)
    resCols = [f"{i}_normalized" if i != "UID" else i for i in res.columns]
    res.columns = resCols
    
    df = df.merge(res, on="UID")
        
    os.remove(dataLoc)
    os.remove(outputLoc)
    
    if returnCols == True:
        return df, [i for i in resCols if i != "UID"]
    return df


def cyclicLOESS(df, cols):
    """
    Function that performs vsn normalization
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
    function that converts phosphosites to "canonical" phosphosites
    This conversion is based on Uniprot Identifier used in PSP data
    possible organisms: 'mouse', 'human', 'rat', 'sheep', 'SARSCoV2', 'guinea pig', 'cow',
       'hamster', 'fruit fly', 'dog', 'rabbit', 'pig', 'chicken', 'frog',
       'quail', 'horse', 'goat', 'papillomavirus', 'water buffalo',
       'marmoset', 'turkey', 'cat', 'starfish', 'torpedo', 'SARSCoV1',
       'green monkey', 'ferret'
    """

    def getUPAcc(gene, organism):
        global ps
        gene = gene.upper()
        try:
            upacc = ps.loc[(ps["GENE"].apply(lambda x: str(x).upper()==gene)) &
                           (ps["ORGANISM"]==organism), "ACC_ID"].iloc[0]
            return upacc
        except:
            return "notFound"
        
        
    def getUPSeq(upacc):
        url = f"https://www.uniprot.org/uniprot/{upacc}.fasta"
        response = requests.get(url)
        seq = ("").join(response.text.split('\n')[1:])
        return seq
        
        
    def getCanonicalPos(seq, psSeq, n):
        alingment = pairwise2.align.localms(seq, psSeq, 2, -2, -1, -1)
        formAlign = format_alignment(*alingment[0])
        start = int(formAlign.lstrip(' ').split(' ')[0])
        start2 = int(formAlign.split('\n')[2].lstrip(' ').split(' ')[0]) - 1
        canPos = start + ((15-n)-start2)
        return canPos
        

    gene = str(series["Gene names"])
    # get uniprot acc
    #counter used if first gene name not found
    #in that case appropiate seq window has to be used
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
    psSeq = series["Sequence window"]
    if ';' in psSeq:
        psSeq = psSeq.split(';')[counter]

    n = len(psSeq) - len(psSeq.lstrip('_'))
    psSeq = psSeq.strip('_')

    return upacc,getCanonicalPos(seq, psSeq, n)
    
    
def getSubCellLoc(series, database="compartments", loca=None, colname="Gene names"):
    """
    Annotates the df with subcellular localization
    For compartments gene names are required
    """
    gene = series[colname]
    if database == "compartments":
        with resources.open_text("autoprot.data","human_compartment_integrated_full") as d:
            compData = pd.read_csv(d, sep='\t', header=None)
            compData.columns = ["ENSMBL", "Gene name", "LOCID", "LOCNAME", "SCORE"]
        if loca is None:
            #raise ValueError("Please include a subcellular localization when using the compartments database")
            return compData[(compData["Gene name"]==series["Gene names"])][["LOCNAME", "SCORE"]]
        return compData[(compData["Gene name"]==series["Gene names"]) &
                (compData["LOCNAME"]==loca)]
    elif database == "hpa":
        cols = "g,scl,scml,scal"
        html = requests.get(f"https://www.proteinatlas.org/api/search_download.php?search={gene}&format=json&columns={cols}&compress=no").text
        mainLoc = html.split('Subcellular main location')[1].split(',"Subcellular additional location')[0].lstrip('":[').split(',')
        altLoc  = html.split('Subcellular additional location')[1].split('}')[0].lstrip('":[').split(',')
        mainLoc = [i.strip('"]') for i in mainLoc]
        altLoc  = [i.strip('"]').rstrip('}') for i in altLoc]
        return mainLoc, altLoc
    

def makeSimScore(m1, m2, corr="pearson"):

    """
    DOI: 10.1126/scisignal.2001570 
    :corr: pearson, spearman
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
            print("Invalid correlation parameter.")
            print("Pearson correlation used instead")
            r = pearsonr(m1, m2)[0]
        dof = len(m1) - 2
        t = (r*np.sqrt(dof))/np.sqrt(1-r**2)
        pval = stats.t.sf(np.abs(t), dof)
        return pval
        
    pComp = calcCorr(m1,m2)
    mComp = calcMagnitude(m1,m2)
    return -10*np.log10(pComp*mComp)
    
    
def normToProt(entry, protDf, toNormalize):
    """
    Function that normalizes phospho data to total protein level
    Function has to be applied to phosphosite table.
    e.g. phosTable.apply(lambda x: normToProt(x, dfProt, toNormalize),1)
    :@dfProt: MQ ProteinGroups data to which data is normalized
    :@toNormalize: which columns to normalize
    """
    try:
        protIds = entry["Protein group IDs"]
    except:
        display(entry["Gene names"])
        raise ValueError()
    if ';' in protIds:
        protIds = [int(i) for i in protIds.split(';')]
        protDf = protDf[protDf["id"].isin(protIds)]
        poi = protDf.groupby("Gene names").median()        
    else: 
        poi = protDf[protDf["id"]==int(protIds)]
    if poi.shape[0] == 0:
        #can*t normalize. either return non-normalized or drop value?
        corrected = pd.DataFrame([np.nan]*len(toNormalize)).T
    else:
        #display(poi[toNormalize])
        #log(pep) - log(prot) or log(pep/prot)
        corrected = entry[toNormalize] - poi[toNormalize]
        #display(corrected)
    return pd.Series(corrected.values[0])