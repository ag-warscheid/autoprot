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


def log(df, cols, base=2, invert=None, return_cols=False, replace_inf=True):
    """
    Perform log transformation.

    Parameters
    ----------
    df : pd.dfFrame
        Input dfframe.
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
        to the dfframe. The default is False.
    replace_inf : bool, optional
        Whether to replace inf and -inf values by np.nan

    Returns
    -------
    pd.dfframe
        The log transformed dfframe.
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

    The resulting dfframe contains log ratios or NaN.

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

