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


def go_analysis(gene_list, organism="hsapiens"):
    # noinspection PyUnresolvedReferences
    """
    Perform go Enrichment analysis (also KEGG and REAC).

    Parameters
    ----------
    gene_list : list of str
        list of gene names.
    organism : str, optional
        identifier for the organism.
        See https://biit.cs.ut.ee/gprofiler/page/organism-list for details.
        The default is "hsapiens".

    Raises
    ------
    ValueError
        If the input could not be parsed as list of gene names.

    Returns
    -------
    gp.profile
        Dataframe-like object with the GO annotations.

    Examples
    --------
    >>> autoprot.analysis.go_analysis(['PEX14', 'PEX18']).iloc[:3,:3]
    source      native                                   name
    0  CORUM  CORUM:1984                 PEX14 homodimer complex
    1  GO:CC  GO:1990429          peroxisomal importomer complex
    2  GO:BP  GO:0036250  peroxisome transport along microtubule
    """
    if not isinstance(gene_list, list):
        try:
            gene_list = list(gene_list)
        except Exception:
            raise ValueError("Please provide a list of gene names")
    return gp.profile(organism=organism, query=gene_list, no_evidences=False)


class KSEA:
    # noinspection PyUnresolvedReferences
    r"""
    Perform kinase substrate enrichment analysis.

    Notes
    -----
    KSEA uses the Kinase-substrate dataset and the
    regulatory-sites dataset from https://www.phosphosite.org/staticDownloads

    Examples
    --------
    KSEA is a method to get insights on which kinases are active in a given
    phosphoproteomic dataset. This is a great method to gain deeper insights
    on the underlying signaling mechanisms and also to generate novel
    hypothesis and find new connections in signaling processes.
    The KSEA class allows you to easily perform the analysis and
    comes with helpful functions to visualize and interpret your results.

    In the first step of the analysis you have to generate a KSEA object.

    >>> ksea = autoprot.analysis.KSEA(phos)

    Next, you can annotate the data with respective kinases.
    You can provide the function with a organism of your choice as well as
    toggle whether to screen for only in vivo determined substrate
    phosphorylation of the respective kinases.

    >>> ksea.annotate(organism="mouse", only_in_vivo=True)

    After the annotation it is always a good idea to get an overview of the
    kinases in the data an how many substrates the have. Based on this you
    might want to adjust a cutoff specifying the minimum number of substrates
    per kinase.

    >>> ksea.get_kinase_overview(kois=["Akt1","MKK4", "P38A", "Erk1"])

    Next, you can perform the actual kinase substrate enrichment analysis.
    The analysis is based on the log fold change of your data.
    Therefore, you have to provide the function with the appropiate column of
    your data and the minimum number of substrates per kinase.

    >>> ksea.ksea(col="logFC_TvC", min_subs=5)

    After the ksea has finished, you can get information for further analysis
    such as the substrates of a specific kinase (or a list of kinases)

    >>> ksea.return_kinase_substrate(kinase=["Akt1", "MKK4"]).sample() # doctest: +SKIP

    or a new dataframe with additional columns for every kinase showing if the
    protein is a substrate of that kinase or not

    >>> ksea.annotate_df(kinases=["Akt1", "MKK4"]).iloc[:2,-5:]

    Eventually, you can also generate plots of the enrichment analysis.

    >>> ksea.plot_enrichment(up_col="salmon")

    .. plot::
        :context: close-figs

        import autoprot.preprocessing as pp
        import autoprot.analysis as ana
        import pandas as pd

        phos = pd.read_csv("_static/testdata/Phospho (STY)Sites_mod.zip", sep="\t", low_memory=False)
        phos = pp.cleaning(phos, file = "Phospho (STY)")
        phosRatio = phos.filter(regex="^Ratio .\/.( | normalized )R.___").columns
        phos = pp.log(phos, phosRatio, base=2)
        phos = pp.filter_loc_prob(phos, thresh=.75)
        phosRatio = phos.filter(regex="log2_Ratio .\/.( | normalized )R.___").columns
        phos = pp.remove_non_quant(phos, phosRatio)

        phosRatio = phos.filter(regex="log2_Ratio .\/. normalized R.___").columns
        phos_expanded = pp.expand_site_table(phos, phosRatio)

        twitchVsmild = ['log2_Ratio H/M normalized R1','log2_Ratio M/L normalized R2','log2_Ratio H/M normalized R3',
                        'log2_Ratio H/L normalized R4','log2_Ratio H/M normalized R5','log2_Ratio M/L normalized R6']
        twitchVsctrl = ["log2_Ratio H/L normalized R1","log2_Ratio H/M normalized R2","log2_Ratio H/L normalized R3",
                        "log2_Ratio M/L normalized R4", "log2_Ratio H/L normalized R5","log2_Ratio H/M normalized R6"]

        phos = ana.ttest(df=phos_expanded, reps=twitchVsmild, cond="_TvM")
        phos = ana.ttest(df=phos_expanded, reps=twitchVsctrl, cond="_TvC")

        ksea = ana.KSEA(phos)
        ksea.annotate(organism="mouse", only_in_vivo=True)
        ksea.get_kinase_overview(kois=["Akt1","MKK4", "P38A", "Erk1"])
        ksea.ksea(col="logFC_TvC", min_subs=5)

        ksea.plot_enrichment(up_col="salmon")

    You can also highlight a list of kinases in volcano plots.
    This is based on the autoprot volcano function.
    You can pass all the common parameters to this function.

    >>> ksea.plot_volcano(log_fc="logFC_TvC", p="pValue_TvC", kinases=["Akt1", "MKK4"],
    ...               annot="Gene names", sig_col="gray")

    .. plot::
        :context: close-figs

        ksea.volcanos(log_fc="logFC_TvC", p="pValue_TvC", kinases=["Akt1", "MKK4"],
                      annot="Gene names", sig_col="gray")

    Sometimes the enrichment is crowded by various kinase isoforms.
    In such cases it makes sense to simplify the annotation by grouping those
    isoforms together.

    >>> simplify = {"ERK":["ERK1","ERK2"],
    ...             "GSK3":["GSK3A", "GSK3B"]}
    >>> ksea.ksea(col="logFC_TvC", min_subs=5, simplify=simplify)
    >>> ksea.plot_enrichment()

    .. plot::
        :context: close-figs

        simplify = {"ERK":["ERK1","ERK2"],
                    "GSK3":["GSK3A", "GSK3B"]}
        ksea.ksea(col="logFC_TvC", min_subs=5, simplify=simplify)
        ksea.plot_enrichment()

    Of course, you can also get the ksea results as a dataframe to save or to further customize.

    >>> ksea.return_enrichment()

    Of course is the database not exhaustive and you might want to add additional
    substrates manually. This can be done the following way.
    Manually added substrates are always added irrespective of the species used
    for the annotation.

    >>> ksea = ana.KSEA(phos)
    >>> genes = ["RPGR"]
    >>> modRsds = ["S564"]
    >>> kinases = ["mTOR"]
    >>> ksea.add_substrate(kinase=kinases, substrate=genes, sub_mod_rsd=modRsds)

    >>> ksea.annotate(organism="mouse", only_in_vivo=True)
    >>> ksea.ksea(col="logFC_TvC", min_subs=5)
    >>> ksea.plot_enrichment(plot_bg=False)

    >>> ksea.clear_manual_substrates()
    >>> ksea.annotate(organism="mouse", only_in_vivo=True)
    >>> ksea.ksea(col="logFC_TvC", min_subs=5)
    >>> ksea.plot_enrichment(plot_bg=False)
    """

    def __init__(self, data):
        """
        Initialise the KSEA object.

        Parameters
        ----------
        data : pd.DataFrame
            matrix_a phosphoproteomics datasaet.
            This data has to contain information about Gene name, position and amino acid of the peptides with
            "Gene names", "Position" and "Amino acid" as the respective column names.
            Optionally you can provide a "Multiplicity" column.

        Returns
        -------
        None.

        """
        with resources.open_binary("autoprot.data", "Kinase_Substrate_Dataset.zip") as d:
            self.PSP_KS = pd.read_csv(d, sep='\t', compression='zip')
        # harmonize gene naming
        self.PSP_KS["SUB_GENE"] = self.PSP_KS["SUB_GENE"].fillna("NA").apply(lambda x: x.upper())
        # add source information
        self.PSP_KS["source"] = "PSP"
        with resources.open_binary("autoprot.data", "Regulatory_sites.zip") as d:
            self.PSP_regSits = pd.read_csv(d, sep='\t', compression='zip')
        # Harmonize the input data and store them to the class
        self.data = self._preprocess(data.copy(deep=True))
        # init other class objects
        self.annotDf = None
        self.kseaResults = None
        self.koi = None
        self.simpleDf = None

    @staticmethod
    def _preprocess(data):
        """Define MOD_RSD, ucGene and mergeID cols in the input dataset."""
        # New column containing the modified residue as Ser201
        data["MOD_RSD"] = data["Amino acid"] + data["Position"].fillna(0).astype(int).astype(str)
        # The Gene names as defined for the Kinase substrate dataset
        data["ucGene"] = data["Gene names"].fillna("NA").apply(lambda x: x.upper())
        # an index column
        data["mergeID"] = range(data.shape[0])
        return data

    @staticmethod
    def _enrichment(df, col, kinase):
        """
        Calculate the enrichment score for a certain kinase.

        Parameters
        ----------
        df : pd.Dataframe
            Input datafame with enrichment information.
        col : str
            Column containing enrichment information e.g. intensity ratios.
            Must be present in df.
        kinase : str
            Kinase to calculate the enrichment for.

        Returns
        -------
        list
            pair of kinase name and score.

        """
        # get enrichment values for rows containing the kinase of interest
        ks = df[col][df["KINASE"].fillna('').apply(lambda x: kinase in x)]
        s = ks.mean()  # mean FC of kinase subs
        p = df[col].mean()  # mean FC of all substrates
        m = ks.shape[0]  # number of kinase substrates
        sig = df[col].std()  # standard dev of FC of all
        score = ((s - p) * np.sqrt(m)) / sig

        return [kinase, score]

    @staticmethod
    def _extract_kois(df):
        """
        Count the number of substrates for each kinase in a merged df.

        Parameters
        ----------
        df : pd.DataFrame
            Merged dataframe containing kinase substrate pairs present in the
            input dataframe.

        Returns
        -------
        pd.DataFrame
            Dataframe with columns "Kinase" and "#Subs" containing the
            numbers of appearances of each kinase in the merged input dataset.

        """
        # Extract all strings present in the KINASE column as list of str
        # This is mainly out of caution as all entries in the kinase col should be
        # strings
        koi = [i for i in list(df["KINASE"].values.flatten()) if isinstance(i, str)]
        # remove duplicates
        ks = set(koi)
        # empty list to take on sets of kinase:count pairs
        temp = [(k, koi.count(k)) for k in ks]
        return pd.DataFrame(temp, columns=["Kinase", "#Subs"])

    def add_substrate(self, kinase: list, substrate: list, sub_mod_rsd: list):
        """
        Manually add a substrate to the database.

        Parameters
        ----------
        kinase : list of str
            Name of the kinase e.g. PAK2.
        substrate : list of str
            Name of the substrate e.g. Prkd1.
        sub_mod_rsd : list of str
            Phosphorylated residues e.g. S203.

        Raises
        ------
        ValueError
            If the three provided lists do not match in length.

        Returns
        -------
        None.

        """
        # a bit cumbersome way to check if all lists
        # are of the same lengths
        it = iter([kinase, substrate, sub_mod_rsd])
        the_len = len(next(it))
        if any(len(x) != the_len for x in it):
            raise ValueError('not all lists have same length!')

        # generate new empty df to fill in the new kinases
        temp = pd.DataFrame(columns=self.PSP_KS.columns)
        for i in range(len(kinase)):
            temp.loc[i, "KINASE"] = kinase[i]
            temp.loc[i, "SUB_GENE"] = substrate[i]
            temp.loc[i, "SUB_MOD_RSD"] = sub_mod_rsd[i]
            temp.loc[i, "source"] = "manual"
        # append to the original database from PSP
        self.PSP_KS = self.PSP_KS.append(temp, ignore_index=True)

    def clear_manual_substrates(self):
        """Remove all manual entries from the PSP database."""
        self.PSP_KS = self.PSP_KS[self.PSP_KS["source"] == "PSP"]

    def annotate(self, organism="human", only_in_vivo=False):
        """
        Annotate with known kinase substrate pairs.

        Parameters
        ----------
        organism : str, optional
            The target organism. The default is "human".
        only_in_vivo : bool, optional
            Whether to restrict analysis to in vivo evidence.
            The default is False.

        Notes
        -----
        Manually added kinases will be included in the annotation search
        independent of the setting of organism and onInVivo.

        Returns
        -------
        None.
        """
        # return a kinase substrate dataframe including only entries of the
        # target organism that were validated in vitro
        if only_in_vivo:
            temp = self.PSP_KS[((self.PSP_KS["KIN_ORGANISM"] == organism) &
                                (self.PSP_KS["SUB_ORGANISM"] == organism) &
                                (self.PSP_KS["IN_VIVO_RXN"] == "X")) | (self.PSP_KS["source"] == "manual")]
        # only filter for the target organism
        else:
            temp = self.PSP_KS[((self.PSP_KS["KIN_ORGANISM"] == organism) &
                                (self.PSP_KS["SUB_ORGANISM"] == organism)) | (self.PSP_KS["source"] == "manual")]

        # merge the kinase substrate data tables with the input dataframe
        # include the multiplicity column in the merge if present in the
        # input dataframe
        # the substrate gene names and the modification position are used for
        # merging
        if "Multiplicity" in self.data.columns:
            self.annotDf = pd.merge(self.data[["ucGene", "MOD_RSD", "Multiplicity", "mergeID"]],
                                    temp,
                                    left_on=["ucGene", "MOD_RSD"],
                                    right_on=["SUB_GENE", "SUB_MOD_RSD"],
                                    how="left")  # keep only entries that are present in the input dataframe
        else:
            self.annotDf = pd.merge(self.data[["ucGene", "MOD_RSD", "mergeID"]],
                                    temp,
                                    left_on=["ucGene", "MOD_RSD"],
                                    right_on=["SUB_GENE", "SUB_MOD_RSD"],
                                    how="left")

        # generate a df with kinase:number of substrate pairs for the dataset
        self.koi = self._extract_kois(self.annotDf)

    # noinspection PyBroadException
    def get_kinase_overview(self, kois=None):
        """
        Plot a graphical overview of the kinases acting on the proteins in the dataset.

        Parameters
        ----------
        kois : list of str, optional
            Kinases of interest for which a detailed overview of substrate numbers
            is plotted. The default is None.

        Returns
        -------
        None.

        """
        # ax[0] is a histogram of kinase substrate numbers and
        # ax[1] is a table of top10 kinases
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

        sns.histplot(self.koi["#Subs"], bins=50, ax=ax[0])
        sns.despine(ax=ax[0])
        ax[0].set_title("Overview of #Subs per kinase")

        # get axis[1] ready - basically remove everthing
        ax[1].spines["left"].set_visible(False)
        ax[1].spines["top"].set_visible(False)
        ax[1].spines["bottom"].set_visible(False)
        ax[1].spines["right"].set_visible(False)
        ax[1].tick_params(axis='both',  # changes apply to the x-axis
                          which='both',  # both major and minor ticks are affected
                          bottom=False,  # ticks along the bottom edge are off
                          top=False,  # ticks along the top edge are off
                          left=False,
                          labelbottom=False,
                          labelleft=False)  # labels along the bottom edge are off
        ax[1].set_xlim(0, 1)
        ax[1].set_ylim(0, 1)

        # plot table
        ax[1].text(x=0, y=1 - 0.01, s="Top10\nKinase")
        ax[1].text(x=0.1, y=1 - 0.01, s="#Subs")

        ax[1].plot([0, 0.2], [.975, .975], color="black")
        ax[1].plot([0.1, 0.1], [0, .975], color="black")

        # get top 10 kinases for annotation
        text = self.koi.sort_values(by="#Subs", ascending=False).iloc[:10].values
        for j, i in enumerate(text):
            j += 1
            ax[1].text(x=0, y=1 - j / 10, s=i[0])
            ax[1].text(x=0.125, y=1 - j / 10, s=i[1])
        # plot some descriptive stats
        tot = self.koi.shape[0]
        s = f"Substrates for {tot} kinases found in data."
        ax[1].text(0.3, 0.975, s)
        med = round(self.koi['#Subs'].median(), 2)
        s = f"Median #Sub: {med}"
        ax[1].text(0.3, 0.925, s)
        mea = round(self.koi['#Subs'].mean(), 2)
        s = f"Mean #Sub: {mea}"
        ax[1].text(0.3, 0.875, s)
        # if kois are provided plot those
        if kois is not None:
            pos = .8
            for k in kois:
                try:
                    s = self.koi[self.koi["Kinase"].apply(lambda x: x.upper()) == k.upper()]["#Subs"].values[0]
                except Exception:
                    s = 0
                ss = f"{k} has {s} substrates."
                ax[1].text(0.3, pos, ss)
                pos -= 0.055

    def ksea(self, col, min_subs=5, simplify=None):
        r"""
        Calculate Kinase Enrichment Score.

        Parameters
        ----------
        col : str
            Column used for the analysis containing the kinase substrate
            enrichments.
        min_subs : int, optional
            Minimum number of substrates a kinase must have to be considered.
            The default is 5.
        simplify : None, "auto" or dict, optional
            Merge multiple kinases during analysis.
            Using "auto" a predefined set of kinase isoforms is merged.
            If provided with a dict, the dict has to contain a list of kinases
            to merge as values and the name of the merged kinases as key.
            The default is None.

        Notes
        -----
        The enrichment score is calculated as

        .. math::
            \frac{(\langle FC_{kinase} \rangle - \langle FC_{all} \rangle)\sqrt{N_{kinase}}}{\sigma_{all}}

        i.e. the difference in mean fold change between kinase and all substrates
        multiplied by the square root of number of kinase substrates and divided
        by the standard deviation of the fold change of all substrates (see [1]).

        References
        ----------
        [1] https://academic.oup.com/bioinformatics/article/33/21/3489/3892392

        Returns
        -------
        None.

        """
        # TODO wouldn't it make more sense to perform simplification in the
        # Annotate function?
        copy_annot_df = self.annotDf.copy(deep=True)
        if simplify is not None:
            if simplify == "auto":
                simplify = {"AKT": ["Akt1", "Akt2", "Akt3"],
                            "PKC": ["PKCA", "PKCD", "PKCE"],
                            "ERK": ["ERK1", "ERK2"],
                            "GSK3": ["GSK3B", "GSK3A"],
                            "JNK": ["JNK1", "JNK2", "JNK3"],
                            "FAK": ["FAK iso2"],
                            "p70S6K": ["p70S6K", "p70SKB"],
                            "RSK": ["p90RSK", "RSK2"],
                            "P38": ["P38A", "P38B", "P38C", "P338D"]}
            for key in simplify:
                copy_annot_df["KINASE"].replace(simplify[key], [key] * len(simplify[key]), inplace=True)

            # drop rows which are now duplicates
            if "Multiplicity" in copy_annot_df.columns:
                idx = copy_annot_df[["ucGene", "MOD_RSD", "Multiplicity", "KINASE"]].drop_duplicates().index
            else:
                idx = copy_annot_df[["ucGene", "MOD_RSD", "KINASE"]].drop_duplicates().index
            copy_annot_df = copy_annot_df.loc[idx]
            self.simpleDf = copy_annot_df

            # repeat annotation with the simplified dataset
            self.koi = self._extract_kois(self.simpleDf)

        # filter kinases with at least min_subs number of substrates
        koi = self.koi[self.koi["#Subs"] >= min_subs]["Kinase"]

        # init empty list to collect sub-dfs
        ksea_results_dfs = []
        # add the enrichment column back to the annotation df using the mergeID
        copy_annot_df = copy_annot_df.merge(self.data[[col, "mergeID"]], on="mergeID", how="left")
        for kinase in koi:
            # calculate the enrichment score
            k, s = self._enrichment(copy_annot_df[copy_annot_df[col].notnull()], col, kinase)
            # new dataframe containing kinase names and scores
            temp = pd.DataFrame(data={"kinase": k, "score": s}, index=[0])
            # add the new df to the pre-initialised list
            ksea_results_dfs.append(temp)

        # generate a single large df from the collected temp dfs
        self.kseaResults = pd.concat(ksea_results_dfs, ignore_index=True)
        # sort the concatenated dfs by kinase enrichment score
        self.kseaResults = self.kseaResults.sort_values(by="score", ascending=False)

    def return_enrichment(self):
        """Return a dataframe of kinase:score pairs."""
        if self.kseaResults is None:
            print("First perform the enrichment")
        else:
            # dropna in case of multiple columns in data
            # sometimes there are otherwise nan
            # nans are dropped in ksea enrichment
            return self.kseaResults.dropna()

    def plot_enrichment(self, up_col="orange", down_col="blue", bg_col="lightgray",
                        plot_bg=True, ret_fig=False, title="", figsize=(5, 10), ax=None):
        """
        Plot the KSEA results.

        Parameters
        ----------
        up_col : str, optional
            Color for enriched/upregulated kinases.
            The default is "orange".
        down_col : str, optional
            Colour for deriched/downregulated kinases.
            The default is "blue".
        bg_col : str, optional
            Colour for not kinases that did not change significantly.
            The default is "lightgray".
        plot_bg : bool, optional
            Whether to plot the unaffected kinases.
            The default is True.
        ret_fig : bool, optional
            Whether to return the figure object.
            The default is False.
        title : str, optional
            Title of the figure. The default is "".
        figsize : tuple of int, optional
            Figure size. The default is (5,10).
        ax : matplotlib.axis, optional
            The axis to plot on. Default is None.

        Returns
        -------
        fig : matplotlib figure.
            Only returned in ret is True.

        """
        if self.kseaResults is None:
            print("First perform the enrichment")
        else:
            # set all proteins to bg_col
            self.kseaResults["color"] = bg_col
            # highlight up and down regulated
            self.kseaResults.loc[self.kseaResults["score"] > 2, "color"] = up_col
            self.kseaResults.loc[self.kseaResults["score"] < -2, "color"] = down_col

            # init figure
            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=figsize)
            else:
                fig = plt.gcf()

            # only plot the unaffected substrates if plot_bg is True
            if plot_bg:
                sns.barplot(data=self.kseaResults.dropna(), x="score", y="kinase",
                            palette=self.kseaResults.dropna()["color"], ax=ax)
            else:
                # else remove the unaffected substrates from the plotting df
                sns.barplot(data=self.kseaResults[self.kseaResults["color"] != bg_col].dropna(), x="score", y="kinase",
                            palette=self.kseaResults[self.kseaResults["color"] != bg_col].dropna()["color"], ax=ax)

            # remove top and right spines/plot lines
            sns.despine()
            plt.legend([], [], frameon=False)
            plt.yticks(fontsize=10)
            ax.set_title(title)
            plt.axvline(0, 0, 1, ls="dashed", color="lightgray")
            # return the figure object only if demanded
            if ret_fig:
                plt.tight_layout()
                return fig
            else:
                return None

    def plot_volcano(self, log_fc, p_colname, kinases=None, ret_fig=False, **kwargs):
        """
        Plot volcano plots highlighting substrates of a given kinase.

        Parameters
        ----------
        log_fc : str
            Column name of column containing the log fold changes.
            Must be present in the dataframe KSEA was initialised with.
        p_colname : str
            Column name of column containing the p values.
            Must be present in the dataframe KSEA was initialised with.
        kinases : list of str, optional
            Limit the analysis to these kinases. The default is [].
        ret_fig : bool
            Whether to return a list of figures for every kinase.
        **kwargs :
            passed to autoprot.visualisation.volcano.

        Returns
        -------
        volcano_returns : list
            list of all returned figure objects. Only if ret_fig is True.

        """
        # generate a df containing only the kinases of interest
        if kinases is None:
            kinases = self.koi.sort_values('#Subs', ascending=False).head(5)['Kinase']
            print("No Kinase supplied, plotting the top 5 kinases.")
        df = self.annotate_df(kinases=kinases)

        volcano_returns = []
        for k in kinases:
            # index for highlighting the selected kinase substrates
            idx = df[df[k] == 1].index
            fig = vis.volcano(df, log_fc, p_colname=p_colname, highlight=idx,
                              kwargs_highlight={"label": f"{k} substrate"},
                              kwargs_both_sig={"alpha": .5},
                              **kwargs
                              )

            # add to the return list
            volcano_returns.append(fig)

        if ret_fig:
            return volcano_returns

    def return_kinase_substrate(self, kinase):
        """
        Return new dataframe with substrates of one or multiple kinase(s).

        Parameters
        ----------
        kinase : str or list of str
            Kinase(s) to analyse.

        Raises
        ------
        ValueError
            If kinase is neither list of str nor str.

        Returns
        -------
        df_filter : pd.Dataframe
            Dataframe containing detailed information on kinase-substrate pairs
            including reference literature.

        """
        # use the simplified dataset if it is present
        if self.simpleDf is not None:
            df = self.simpleDf.copy(deep=True)
        # otherwise use the complete dataset including kinase isoforms
        else:
            df = self.annotDf.copy(deep=True)

        # if a list of kinases is provided, iterate through the list and
        # collect corresponding indices
        if isinstance(kinase, list):
            idx = [df[df["KINASE"].fillna("NA").apply(lambda x: x.upper()) == k.upper()].index for k in kinase]

            # merge all row indices and use them to create a sub-df containing
            # only the kinases of interest
            df_filter = df.loc[pl.flatten(idx)]
        elif isinstance(kinase, str):
            df_filter = df[df["KINASE"].fillna("NA").apply(lambda x: x.upper()) == kinase.upper()]
        else:
            raise ValueError("Please provide either a string or a list of strings representing kinases of interest.")

        # data are merged implicitly on common column nnames i.e. on SITE_GRP_ID
        # only entries present in the filtered annotDfare retained
        df_filter = pd.merge(df_filter[['GENE', 'KINASE', 'KIN_ACC_ID', 'SUBSTRATE', 'SUB_ACC_ID',
                                        'SUB_GENE', 'SUB_MOD_RSD', 'SITE_GRP_ID', 'SITE_+/-7_AA', 'DOMAIN',
                                        'IN_VIVO_RXN', 'IN_VITRO_RXN', 'CST_CAT#', 'source', "mergeID"]],
                             self.PSP_regSits[['SITE_GRP_ID', 'ON_FUNCTION', 'ON_PROCESS', 'ON_PROT_INTERACT',
                                               'ON_OTHER_INTERACT', 'PMIDs', 'LT_LIT', 'MS_LIT', 'MS_CST',
                                               'NOTES']],
                             how="left")
        return df_filter

    def annotate_df(self, kinases=None):
        """
        Annotate the provided dataframe with boolean columns for given kinases.

        Parameters
        ----------
        kinases : list of str, optional
            List of kinases. The default is [].

        Returns
        -------
        pd.DataFrame
            annotated dataframe containing a column for each provided kinase
            with boolean values representing a row/protein being a kinase
            substrate or not.

        """
        if kinases is None:
            raise ValueError('Please provide at least one kinase for annotation')
        if len(kinases) > 0:
            # remove the two columns from the returned df
            df = self.data.drop(["MOD_RSD", "ucGene"], axis=1)
            for kinase in kinases:
                # find substrates for the given kinase in the dataset
                ids = self.return_kinase_substrate(kinase)["mergeID"]
                # init the boolean column with zeros
                df[kinase] = 0
                # check if the unique ID for each protein is present in the
                # returnKinaseSubstrate df. If so set the column value to 1.
                df.loc[df["mergeID"].isin(ids), kinase] = 1
            # remove also the mergeID column before returning the df
            return df.drop("mergeID", axis=1)
        else:
            raise ValueError('Please provide at least one kinase for annotation')
