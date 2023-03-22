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


def annotate_phosphosite(df, ps, cols_to_keep=None):
    """
    Annotate phosphosites with information derived from PhosphositePlus.

    Parameters
    ----------
    df : pd.Dataframe
        dataframe containing PS of interst.
    ps : str
        Column containing info about the PS.
        Format: GeneName_AminoacidPositoin (e.g. AKT_T308).
    cols_to_keep : list, optional
        Which columns from original dataframe (input df) to keep in output.
        The default is None.

    Returns
    -------
    pd.Dataframe
        The input dataframe with the kept columns and additional phosphosite cols.

    """

    if cols_to_keep is None:
        cols_to_keep = []

    def make_merge_col(df_to_merge, file="regSites"):
        """Format the phosphosite positions and gene names so that merging is possible."""
        if file == "regSites":
            return df_to_merge["GENE"].fillna("").apply(lambda x: str(x).upper()) + '_' + \
                   df_to_merge["MOD_RSD"].fillna("").apply(
                       lambda x: x.split('-')[0])
        return df_to_merge["SUB_GENE"].fillna("").apply(lambda x: str(x).upper()) + '_' + \
               df_to_merge["SUB_MOD_RSD"].fillna("")

    with resources.open_binary("autoprot.data", "Kinase_Substrate_Dataset.zip") as d:
        ks = pd.read_csv(d, sep='\t', compression='zip')
        ks["merge"] = make_merge_col(ks, "KS")
    with resources.open_binary("autoprot.data", "Regulatory_sites.zip") as d:
        reg_sites = pd.read_csv(d, sep='\t', compression='zip')
        reg_sites["merge"] = make_merge_col(reg_sites)

    ks_coi = ['KINASE', 'DOMAIN', 'IN_VIVO_RXN', 'IN_VITRO_RXN', 'CST_CAT#', 'merge']
    reg_sites_coi = ['ON_FUNCTION', 'ON_PROCESS', 'ON_PROT_INTERACT', 'ON_OTHER_INTERACT',
                     'PMIDs', 'NOTES', 'LT_LIT', 'MS_LIT', 'MS_CST', 'merge']

    df = df.copy(deep=True)
    df.rename(columns={ps: "merge"}, inplace=True)
    df = df[["merge"] + cols_to_keep]
    df = df.merge(ks[ks_coi], on="merge", how="left")
    df = df.merge(reg_sites[reg_sites_coi], on="merge", how="left")

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
