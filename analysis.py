# -*- coding: utf-8 -*-
"""
Autoprot Analysis Functions.

@author: Wignand

@documentation: Julian
"""
import os
from subprocess import run, PIPE, STDOUT
from importlib import resources
from typing import Iterable, List, Any

from statsmodels.stats import multitest as mt
import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp, ttest_ind, wilcoxon
from sklearn.decomposition import PCA
import matplotlib.pylab as plt
import pylab as pl
import seaborn as sns
from operator import itemgetter
from Bio import Entrez
import time

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
    >>> prot_tt = autoprot.analysis.ttest(df=protLog, reps=twitchVsmild, cond="_TvM", return_fc=True, adjust_p_vals=True)
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
        prot = pd.read_csv("_static/testdata/proteinGroups.zip", sep='\\t', low_memory=False)
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
        return np.ma.filled(ttest_1samp(x, nan_policy="omit", alternative=alternative, popmean=0)[1], np.nan)

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
    >>> prot = pd.read_csv("_static/testdata/proteinGroups.zip", sep='\t', low_memory=False)
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


class AutoPCA:
    r"""
    Conduct principal component analyses.

    The class encompasses a set of helpful visualizations
    for further investigating the results of the PCA
    It needs the matrix on which the PCA is performed
    as well as row labels (rlabels)
    and column labels (clabels) corresponding to the
    provided matrix.

    Notes
    -----
    PCA is a method which allows you to visually investigate the underlying structure
    in your data after reduction of the dimensionality.
    With the .autoPCA() you can easily perform a PCA and also generate exploratory figures.
    Intro to PCA: https://learnche.org/pid/latent-variable-modelling/principal-component-analysis/index

    Examples
    --------
    for PCA no missing values are allowed
    filter those and store complete dataframe

    >>> temp = prot[~prot.filter(regex="log2.*norm").isnull().any(1)]

    get the matrix of quantitative values corresponding to conditions of interest
    Here we only use the first replicate for clarity

    >>> dataframe = temp.filter(regex="log2.*norm.*_1$")

    generate appropiate names for the columns and rows of the matrix
    for example here the columns represent the conditions and we are not interested in the rows (which are the genes)

    >>> clabels = dataframe.columns
    >>> rlabels = np.nan

    generate autopca object

    >>> autopca = autoprot.analysis.AutoPCA(dataframe, rlabels, clabels)

    The scree plots describe how much of the total variance of the dataset is
    explained ba the first n components. As you want to explain as variance as
    possible with as little variables as possible, chosing the number of components
    directly right to the steep descend of the plot is usually a good idea.

    >>> autopca.scree()

    .. plot::
        :context: close-figs

        import autoprot.analysis as ana
        import autoprot.preprocessing as pp
        import pandas as pd

        prot = pd.read_csv("_static/testdata/proteinGroups.zip", sep="\t", low_memory=False)
        protRatio = prot.filter(regex="Ratio .\/. normalized")
        protLog = pp.log(prot, protRatio, base=2)
        temp = protLog[~protLog.filter(regex="log2.*norm").isnull().any(1)]
        dataframe = temp.filter(regex="log2.*norm.*_1$")
        clabels = dataframe.columns
        rlabels = np.nan
        autopca = ana.AutoPCA(dataframe, rlabels, clabels)
        autopca.scree()

    The corrComp heatmap shows the PCA loads (i.e. how much a principal component is
    influenced by a change in that variable) relative to the variables (i.e. the
    experiment conditions). If a weight (colorbar) is close to zero, the corresponding
    PC is barely influenced by it.

    >>> autopca.corr_comp(annot=False)

    .. plot::
        :context: close-figs

        autopca.corr_comp(annot=False)

    The bar loading plot is a different way to represent the weights/loads for each
    condition and principal component. High values indicate a high influence of the
    variable/condition on the PC.

    >>> autopca.bar_load(pc=1)
    >>> autopca.bar_load(pc=2)

    .. plot::
        :context: close-figs

        autopca.bar_load(pc=1)
        autopca.bar_load(pc=2)

    The score plot shows how the different data points (i.e. proteins) are positioned
    with respect to two principal components.
    In more detail, the scores are the original data values multiplied by the
    weights of each value for each principal component.
    Usually they will separate more in the direction of PC1 as this component
    explains the largest share of the data variance

    >>> autopca.score_plot(pc1=1, pc2=2)

    .. plot::
        :context: close-figs

        autopca.score_plot(pc1=1, pc2=2)

    The loading plot is the 2D representation of the barLoading plots and shows
    the weights how each variable influences the two PCs.

    >>> autopca.loading_plot(pc1=1, pc2=2, labeling=True)

    .. plot::
        :context: close-figs

        autopca.loading_plot(pc1=1, pc2=2, labeling=True)

    The Biplot is a combination of loading plot and score plot as it shows the
    scores for each protein as point and the weights for each variable as
    vectors.
    >>> autopca.bi_plot(pc1=1, pc2=2)

    .. plot::
        :context: close-figs

        autopca.bi_plot(pc1=1, pc2=2)
    """

    # =========================================================================
    # TODO
    # - Add interactive 3D scatter plot
    # - Facilitate naming of columns and rows
    # - Allow further customization of plots (e.g. figsize)
    # - Implement pair plot for multiple dimensions
    # =========================================================================
    def __init__(self, dataframe: pd.DataFrame, rlabels: list, clabels: list, batch: list = None):
        """
        Initialise PCA class.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Input dataframe.
        rlabels : list
            Row labels.
        clabels : list
            Column labels.
        batch : list, optional
            Labels for distinct conditions used to colour dots in score plot.
            Must be the length of rlabels.
            The default is None.

        Returns
        -------
        None.

        """
        if not isinstance(dataframe, pd.DataFrame):
            dataframe = pd.DataFrame(dataframe)
        # drop any rows in the dataframe containing missing values
        self.X = dataframe.dropna()
        self.label = clabels
        self.rlabel = rlabels
        self.batch = batch
        # PCA is performed with the df containing missing values
        self.pca, self.forVis = self._perform_pca(dataframe, clabels)
        # generate scores from loadings
        self.Xt = self.pca.transform(self.X)
        self.expVar = self.pca.explained_variance_ratio_

    @staticmethod
    def _perform_pca(dataframe, label):
        """Perform pca and generate for_vis dataframe."""
        pca = PCA().fit(dataframe.dropna())
        # components_ is and ndarray of shape (n_components, n_features)
        # and contains the loadings/weights of each PCA eigenvector
        for_vis = pd.DataFrame(pca.components_.T)
        for_vis.columns = [f"PC{i}" for i in range(1, min(dataframe.shape[0], dataframe.T.shape[0]) + 1)]
        for_vis["label"] = label
        return pca, for_vis

    def scree(self, figsize=(15, 5)):
        """
        Plot Scree plot and Explained variance vs number of components.

        Parameters
        ----------
        figsize : tuple of int, optional
            The size of the figure object.
            The default is (15,5).

        Raises
        ------
        TypeError
            No PCA object was initialised in the class.

        Returns
        -------
        None.

        """
        if not isinstance(self.pca, PCA):
            raise TypeError("This is a function to plot Scree plots. Provide fitted sklearn PCA object.")

        eig_val = self.pca.explained_variance_
        cum_var = np.append(np.array([0]), np.cumsum(self.expVar))

        def _set_labels(ylabel, title):
            plt.ylabel(ylabel)
            plt.xlabel("# Component")
            plt.title(title)
            sns.despine()

        plt.figure(figsize=figsize)
        plt.subplot(121)
        plt.plot(range(1, len(eig_val) + 1), eig_val, marker="o", color="teal",
                 markerfacecolor='purple')
        _set_labels("Eigenvalues", "Scree plot")
        plt.subplot(122)
        plt.plot(range(1, len(cum_var) + 1), cum_var, ds="steps", color="teal")
        plt.xticks(range(1, len(eig_val) + 1))
        _set_labels("explained cumulative variance", "Explained variance")

    def corr_comp(self, annot=False):
        """
        Plot heatmap of PCA weights vs. variables.

        Parameters
        ----------
        annot : bool, optional
            If True, write the data value in each cell.
            If an array-like with the same shape as data, then use this
            to annotate the heatmap instead of the data.
            Note that DataFrames will match on position, not index.
            The default is False.

        Notes
        -----
        2D representation how strong each observation (e.g. log protein ratio)
        weights for each principal component.

        Returns
        -------
        None.

        """
        plt.figure()
        sns.heatmap(self.forVis.filter(regex="^PC"), cmap=sns.color_palette("PuOr", 10), annot=annot)
        yp = [i + 0.5 for i in range(len(self.label))]
        plt.yticks(yp, self.forVis["label"], rotation=0)
        plt.title("")

    def bar_load(self, pc=1, n=25):
        """
        Plot the loadings of a given component in a barplot.

        Parameters
        ----------
        pc : int, optional
            Component to draw. The default is 1.
        n : int, optional
            Plot only the n first rows.
            The default is 25.

        Returns
        -------
        None.

        """
        pc = f"PC{pc}"
        for_vis = self.forVis.copy()
        for_vis[f"{pc}_abs"] = abs(for_vis[pc])
        for_vis["color"] = "negative"
        for_vis.loc[for_vis[pc] > 0, "color"] = "positive"
        for_vis = for_vis.sort_values(by=f"{pc}_abs", ascending=False)[:n]
        plt.figure()
        ax = plt.subplot()
        sns.barplot(x=for_vis[pc], y=for_vis["label"], hue=for_vis["color"], alpha=.5,
                    hue_order=["negative", "positive"], palette=["teal", "purple"])
        ax.get_legend().remove()
        sns.despine()

    def return_load(self, pc=1, n=25):
        """
        Return the load for a given principal component.

        Parameters
        ----------
        pc : int, optional
            Component to draw. The default is 1.
        n : int, optional
            Plot only the n first rows.
            The default is 25.

        Returns
        -------
        pd.DataFrame
            Dataframe containing load vs. condition.

        """
        pc = f"PC{pc}"
        for_vis = self.forVis.copy()
        for_vis[f"{pc}_abs"] = abs(for_vis[pc])
        for_vis = for_vis.sort_values(by=f"{pc}_abs", ascending=False)[:n]
        return for_vis[[pc, "label"]]

    def return_score(self):
        """
        Return a dataframe of all scorings for all principal components.

        Returns
        -------
        scores : pd.DataFrame
            Dataframe holding the principal components as colnames and
            the scores for each protein on that PC as values.
        """
        columns = [f"PC{i + 1}" for i in range(self.Xt.shape[1])]
        scores = pd.DataFrame(self.Xt, columns=columns)
        if self.batch is not None:
            scores["batch"] = self.batch
        return scores

    def score_plot(self, pc1=1, pc2=2, labeling=False, file=None, figsize=(5, 5)):
        """
        Generate a PCA score plot.

        Parameters
        ----------
        pc1 : int, optional
            Number of the first PC to plot. The default is 1.
        pc2 : int, optional
            Number of the second PC to plot. The default is 2.
        labeling : bool, optional
            If True, points are labelled with the corresponding
            column labels. The default is False.
        file : str, optional
            Path to save the plot. The default is None.
        figsize : tuple of int, optional
            Figure size. The default is (5,5).

        Notes
        -----
        This will return a scatterplot with as many points as there are
        entries (i.e. protein IDs).
        The scores for each PC are the original protein ratios multiplied with
        the loading weights.
        The score plot corresponds to the individual positions of of each protein
        on a hyperplane generated by the pc1 and pc2 vectors.

        Returns
        -------
        None.

        """
        x = self.Xt[::, pc1 - 1]
        y = self.Xt[::, pc2 - 1]
        plt.figure(figsize=figsize)
        if self.batch is None:
            for_vis = pd.DataFrame({"x": x, "y": y})
            sns.scatterplot(data=for_vis, x="x", y="y")
        else:
            for_vis = pd.DataFrame({"x": x, "y": y, "batch": self.batch})
            sns.scatterplot(data=for_vis, x="x", y="y", hue=for_vis["batch"])
        for_vis["label"] = self.rlabel

        plt.title("Score plot")
        plt.xlabel(f"PC{pc1}\n{round(self.expVar[pc1 - 1] * 100, 2)} %")
        plt.ylabel(f"PC{pc2}\n{round(self.expVar[pc2 - 1] * 100, 2)} %")

        if labeling is True:
            ss = for_vis["label"]
            xx = for_vis["x"]
            yy = for_vis["y"]
            for x, y, s in zip(xx, yy, ss):
                plt.text(x, y, s)
        sns.despine()

        if file is not None:
            plt.savefig(fr"{file}/ScorePlot.pdf")

    def loading_plot(self, pc1=1, pc2=2, labeling=False, figsize=(5, 5)):
        """
        Generate a PCA loading plot.

        Parameters
        ----------
        pc1 : int, optional
            Number of the first PC to plot. The default is 1.
        pc2 : int, optional
            Number of the second PC to plot. The default is 2.
        labeling : bool, optional
            If True, points are labelled with the corresponding
            column labels. The default is False.
        figsize : tuple of int, optional
            The size of the figure object.
            The default is (5,5).

        Notes
        -----
        This will return a scatterplot with as many points as there are
        components (i.e. conditions) in the dataset.
        For each component a load magnitude for two PCs will be printedd
        that describes how much each condition influences the magnitude
        of the respective PC.

        Returns
        -------
        None.

        """
        plt.figure(figsize=figsize)
        if self.batch is None or len(self.batch) != self.forVis.shape[0]:
            sns.scatterplot(data=self.forVis, x=f"PC{pc1}",
                            y=f"PC{pc2}", edgecolor=None)
        else:
            sns.scatterplot(data=self.forVis, x=f"PC{pc1}",
                            y=f"PC{pc2}", edgecolor=None, hue=self.batch)
        sns.despine()

        plt.title("Loadings plot")
        plt.xlabel(f"PC{pc1}\n{round(self.expVar[pc1 - 1] * 100, 2)} %")
        plt.ylabel(f"PC{pc2}\n{round(self.expVar[pc2 - 1] * 100, 2)} %")

        if labeling is True:
            ss = self.forVis["label"]
            xx = self.forVis[f"PC{pc1}"]
            yy = self.forVis[f"PC{pc2}"]
            for x, y, s in zip(xx, yy, ss):
                plt.text(x, y, s)

    def bi_plot(self, pc1=1, pc2=2, num_load="all", figsize=(5, 5), **kwargs):
        """
        Generate a biplot, a combined loadings and score plot.

        Parameters
        ----------
        pc1 : int, optional
            Number of the first PC to plot. The default is 1.
        pc2 : int, optional
            Number of the second PC to plot. The default is 2.
        num_load : 'all' or int, optional
            Plot only the n first rows.
            The default is "all".
        figsize : tuple of int, optional
            Figure size. The default is (3,3).
        **kwargs :
            Passed to plt.scatter.

        Notes
        -----
        In the biplot, scores are shown as points and loadings as
        vectors.

        Returns
        -------
        None.

        """
        x = self.Xt[::, pc1 - 1]
        y = self.Xt[::, pc2 - 1]
        plt.figure(figsize=figsize)
        plt.scatter(x, y, color="lightgray", alpha=0.5, linewidth=0, **kwargs)

        temp = self.forVis[[f"PC{pc1}", f"PC{pc2}"]]
        temp["label"] = self.label
        temp = temp.sort_values(by=f"PC{pc1}")

        if num_load == "all":
            loadings = temp[[f"PC{pc1}", f"PC{pc2}"]].values
            labels = temp["label"].values
        else:
            loadings = temp[[f"PC{pc1}", f"PC{pc2}"]].iloc[:num_load].values
            labels = temp["label"].iloc[:num_load].values

        xscale = 1.0 / (self.Xt[::, pc1 - 1].max() - self.Xt[::, pc1 - 1].min())
        yscale = 1.0 / (self.Xt[::, pc2 - 1].max() - self.Xt[::, pc2 - 1].min())
        xmina = 0
        xmaxa = 0
        ymina = 0

        for load, lab in zip(loadings, labels):
            # plt.plot([0,load[0]/xscale], (0, load[1]/yscale), color="purple")
            plt.arrow(x=0, y=0, dx=load[0] / xscale, dy=load[1] / yscale, color="purple",
                      head_width=.2)
            plt.text(x=load[0] / xscale, y=load[1] / yscale, s=lab)

            if load[0] / xscale < xmina:
                xmina = load[0] / xscale
            elif load[0] / xscale > xmaxa:
                xmaxa = load[0] / xscale

            if load[1] / yscale < ymina or load[1] / yscale > ymina:
                ymina = load[1] / yscale

        plt.xlabel(f"PC{pc1}\n{round(self.expVar[pc1 - 1] * 100, 2)} %")
        plt.ylabel(f"PC{pc2}\n{round(self.expVar[pc2 - 1] * 100, 2)} %")
        sns.despine()

    def pair_plot(self, n=0):
        """
        Draw a pair plot of for pca for the given number of dimensions.

        Parameters
        ----------
        n : int, optional
            Plot only the n first rows. The default is 0.

        Notes
        -----
        Be careful for large data this might crash you PC -> better specify n!

        Returns
        -------
        None.

        """
        # TODO must be prettyfied quite a bit....
        if n == 0:
            n = self.Xt.shape[0]

        for_vis = pd.DataFrame(self.Xt[:, :n])
        i = np.argmin(self.Xt.shape)
        pcs = self.Xt.shape[i]
        for_vis.columns = [f"PC {i}" for i in range(1, pcs + 1)]
        if self.batch is not None:
            for_vis["batch"] = self.batch
            sns.pairplot(for_vis, hue="batch")
        else:
            sns.pairplot(for_vis)


class KSEA:
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
    toggle whether or not to screen for only in vivo determined substrate
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

    >>> ksea.volcanos(log_fc="logFC_TvC", p="pValue_TvC", kinases=["Akt1", "MKK4"],
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
                        plot_bg=True, ret=False, title="", figsize=(5, 10)):
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
        ret : bool, optional
            Whether to return the figure object.
            The default is False.
        title : str, optional
            Title of the figure. The default is "".
        figsize : tuple of int, optional
            Figure size. The default is (5,10).

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
            fig = plt.figure(figsize=figsize)
            plt.yticks(fontsize=10)
            plt.title(title)
            # only plot the unaffected substrates if plot_bg is True
            if plot_bg:
                sns.barplot(data=self.kseaResults.dropna(), x="score", y="kinase",
                            palette=self.kseaResults.dropna()["color"])
            else:
                # else remove the unaffected substrates from the plotting df
                sns.barplot(data=self.kseaResults[self.kseaResults["color"] != bg_col].dropna(), x="score", y="kinase",
                            palette=self.kseaResults[self.kseaResults["color"] != bg_col].dropna()["color"])

            # remove top and right spines/plot lines
            sns.despine()
            plt.legend([], [], frameon=False)
            plt.axvline(0, 0, 1, ls="dashed", color="lightgray")
            # return the figure object only if demanded
            if ret:
                plt.tight_layout()
                return fig

    def volcanos(self, log_fc, p, kinases=None, **kwargs):
        """
        Plot volcano plots highlighting substrates of a given kinase.

        Parameters
        ----------
        log_fc : str
            Column name of column containing the log fold changes.
            Must be present in the dataframe KSEA was initialised with.
        p : str
            Column name of column containing the p values.
            Must be present in the dataframe KSEA was initialised with.
        kinases : list of str, optional
            Limit the analysis to these kinases. The default is [].
        **kwargs :
            passed to autoprot.visualisation.volcano.

        Returns
        -------
        None.

        """
        # generate a df containing only the kinases of interest
        if kinases is None:
            kinases = []
        df = self.annotate_df(kinases=kinases)
        for k in kinases:
            # index for highlighting the selected kinase substrates
            idx = df[df[k] == 1].index
            vis.volcano(df, log_fc, p=p, highlight=idx,
                        custom_hl={"label": k},
                        custom_fg={"alpha": .5},
                        **kwargs)

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
            kinases = []
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
            print("Please provide kinase(s) for annotation.")


def miss_analysis(df, cols, n=None, sort='ascending', text=True, vis=True,
                  extra_vis=False, save_dir=None):
    r"""
    Print missing statistics for a dataframe.

    Parameters
    ----------
    df : pd.Dataframe
        Input dataframe with missing values.
    cols : list of str
        Columns to perform missing values analysis on.
    n : int, optional
        How many rows of the dataframe to displayed.
        The default is None (uses all rows).
    sort : str, optional
        "ascending" or "descending".
        The default is 'ascending'.
    text : bool, optional
        Whether to output text summaryMap.
        The default is True.
    vis : bool, optional
        whether to return barplot showing missingness.
        The default is True.
    extra_vis : bool, optional
        Whether to return matrix plot showing missingness.
        The default is False.
    save_dir : str, optional
        Path to folder where the results should be saved.
        The default is None.

    Raises
    ------
    ValueError
        If n_entries is incorrectly specified.

    Returns
    -------
    None.

    Examples
    --------
    miss_analysis gives a quick overview of the missingness of the provided
    dataframe. You can provide the complete or prefiltered dataframe as input.
    Providing n_entries allows you to specify how many of the entries of the dataframe
    (sorted by missingness) are displayed (i.e. only display the n_entries columns with
    most (or least) missing values) With the sort argument you can define
    whether the dataframe is sorted by least to most missing values or vice versa
    (using "descending" and "ascending", respectively). The vis and extra_vis
    arguments can be used to toggle the graphical output.
    In case of large data (a lot of columns) those might be better turned off.

    >>> autoprot.analysis.miss_analysis(phos_expanded,
    ...                                twitchVsctrl+twitchVsmild+mildVsctrl,
    ...                                sort="descending",
    ...                                extra_vis = True)

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
        mildVsctrl = ["log2_Ratio M/L normalized R1","log2_Ratio H/L normalized R2","log2_Ratio M/L normalized R3",
                      "log2_Ratio H/M normalized R4","log2_Ratio M/L normalized R5","log2_Ratio H/L normalized R6"]
        phos = ana.ttest(df=phos_expanded, reps=twitchVsmild, cond="TvM")
        phos = ana.ttest(df=phos_expanded, reps=twitchVsctrl, cond="TvC")

        ana.miss_analysis(phos_expanded,
                         twitchVsctrl+twitchVsmild+mildVsctrl,
                         text=False,
                         sort="descending",
                         extra_vis = True)
    """
    # only analyse subset of cols
    df = df[cols]
    # sorted list of lists with every sublist containing
    # [colname,total_n, n_missing, percentage, rank]

    # calculate summary missing statistics
    data = []
    # implicitly iterate over dataframe cols
    for i in df:
        # len dataframe
        n_entries = df.shape[0]
        # how many are missing
        m = df[i].isnull().sum()
        # percentage
        p = m / n_entries * 100
        data.append([i, n_entries, m, p])

    # Sort data by the percentage of missingness
    data = sorted(data, key=itemgetter(3))
    # inverse dataframe if required
    if sort == 'descending':
        data = data[::-1]

    # add a number corresponding to the position in the ranking
    # to every condition aka column.
    for idx, col in enumerate(data):
        col.append(idx)

    # determine number of entries to show
    if n is None:
        n = len(data)
    elif n > len(data):
        print("'n_entries' is larger than dataframe!\nDisplaying complete dataframe.")
        n = len(data)
    if n < 0:
        raise ValueError("'n_entries' has to be a positive integer!")

    if text:  # print summary statistics and saves them to file
        allines = ''
        for i in range(n):
            allines += f"{data[i][0]} has {data[i][2]} of {data[i][1]} entries missing ({round(data[i][3], 2)}%)."
            allines += '\n'
            # line separator
            allines += '-' * 80

        if save_dir:
            with open(f"{save_dir}/missAnalysis_text.txt", 'w') as f:
                for _ in range(n):
                    f.write(allines)

        # write all lines at once
        print(allines)

    if vis:  # Visualize the % missingness of first n entries of dataframe as a bar plot.
        data = pd.DataFrame(data=data,
                            columns=["Name", "tot_values", "tot_miss", "perc_miss", "rank"])

        plt.figure(figsize=(7, 7))
        ax = plt.subplot()
        # plot colname against total missing values
        splot = sns.barplot(x=data["tot_miss"].iloc[:n],
                            y=data["Name"].iloc[:n])

        # add the percentage of missingness to every bar of the plot
        for idx, p in enumerate(splot.patches):
            s = f'{str(round(data.iloc[idx, 3], 2))}%'
            x = p.get_width() + p.get_width() * .01
            y = p.get_y() + p.get_height() / 2
            splot.annotate(s, (x, y))

        plt.title("Missing values of dataframe columns.")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel("")

        if save_dir:
            plt.savefig(f"{save_dir}/missAnalysis_vis1.pdf")

    if extra_vis:  # Visualize the missingness in the dataset using missingno.
        # plots are generated with missingno.matrix.
        # See https://github.com/ResidentMario/missingno

        fig, ax = plt.subplots(1)
        msn.matrix(df, sort="ascending", ax=ax)
        if save_dir:
            plt.savefig(save_dir + "/missAnalysis_vis2.pdf")
        return True


def get_pub_abstracts(text=None, title_or_abstract=None, author=None, phrase=None,
                      exclusion=None, custom_search_term=None, make_word_cloud=False,
                      sort='pub+date', retmax=20, output="print"):
    """
    Get Pubmed abstracts.

    Notes
    -----
    Provide the search terms and pubmed will be searched for paper matching those
    The exclusion list can be used to exclude certain words to be contained in a paper
    If you want to perform more complicated searches use the possibility of using
    a custom term.

    If you search for a not excact term you can use *
    For example opto* will also return optogenetic.

    Parameters
    ----------
    text : list of str, optional
        List of words which must occur in text.
        The default is None.
    title_or_abstract : list of str, optional
        List of words which must occur in Titel or Abstract.
        The default is None.
    author : list of str, optional
        List of authors which must occor in Author list.
        The default is None.
    phrase : list of str, optional
        List of phrases which should occur in text.
        Seperate word in phrases with hyphen (e.g. akt-signaling).
        The default is None.
    exclusion : list of str, optional
        List of words which must not occur in text.
        The default is None.
    custom_search_term : str, optional
        Enter a custom search term (make use of advanced pubmed search functionality).
        If this is supplied all other search terms will be ignored.
        The default is None.
    make_word_cloud : bool, optional
        Whether to draw a wordcloud of the words in retrieved abstrace.
        The default is False.
    sort : str, optional
        How to sort the results.
        The default is 'pub+date'.
    retmax : int, optional
        Maximum number of found articles to return.
        The default is 20.
    output : str, optional
        How to handle the output.
        Possible values are 'SOMEPATH.txt',
        'SOMEPATH.html' or 'SOMEPATH'.
        If no extension is given, the output will be html.
        The default is "print".

    Returns
    -------
    None.

    Examples
    --------
    To generate a wordcloud and print the results of the found articles to the
    prompt use the following command

    >>> autoprot.analysis.get_pub_abstracts(titel_or_abstract=["p38", "JNK", "ERK"],
                                          make_word_cloud=True)

    .. plot::
        :context: close-figs

        import autoprot.analysis as ana
        ana.get_pub_abstracts(title_or_abstract=["p38", "JNK", "ERK"],
                              make_word_cloud=True)

    Even more comfortably, you can also save the results incl. the wordcloud
    as html file

    >>>  autoprot.analysis.get_pub_abstracts(titel_or_abstract=["p38", "JNK", "ERK"],
                                          make_word_cloud=True,
                                          output='./MyPubmedSearch.html')

    """

    if exclusion is None:
        exclusion = [""]
    if phrase is None:
        phrase = [""]
    if author is None:
        author = [""]
    if title_or_abstract is None:
        title_or_abstract = [""]
    if text is None:
        text = [""]

    # STEP 1: Generate the search term
    if custom_search_term is None:  # generate a pubmed search term from user input
        # Generate long list of concatenated search terms
        term = [i + "[Title/Abstract]" if i != "" else "" for i in title_or_abstract] + \
               [i + "[Text Word]" if i != "" else "" for i in text] + \
               [i + "[Author]" if i != "" else "" for i in author] + \
               [i if i != "" else "" for i in phrase]
        term = " AND ".join([i for i in term if i != ""])

        # add exclusions joined by NOT
        if exclusion != [""]:
            exclusion = " NOT ".join(exclusion)
            term = " NOT ".join([term] + [exclusion])
    else:  # if the user provides a custom search term, ignore all remaining input
        term = custom_search_term

    # STEP 2: Perform the PubMed search and return the PubMed IDs of the found articles
    Entrez.email = 'your.email@example.com'  # seemingly no true mail address is required.
    handle = Entrez.esearch(db='pubmed',
                            sort=sort,
                            retmax=str(retmax),
                            retmode='xml',
                            term=term)
    time.sleep(0.5)  # if function is executed in a loop: a small delay to ensure pubmed is responding
    results = Entrez.read(handle)["IdList"]

    # if the search was not successful, leave the function
    if len(results) == 0:
        print("No results for " + term)
        return

    # STEP 3: get more detailed results including abstracts, authors etc
    handle = Entrez.efetch(db='pubmed',
                           retmode='xml',
                           id=','.join(set(results)))
    papers = Entrez.read(handle)["PubmedArticle"]

    titles = []
    abstracts = []
    authors = []
    dois = []
    links = []
    pids = []
    dates = []
    journals = []
    final_dict = dict()

    # iterate through the papers
    for paper in papers:
        # get titles
        titles.append(paper["MedlineCitation"]["Article"]["ArticleTitle"])
        # get abstract
        if "Abstract" in paper["MedlineCitation"]["Article"].keys():
            abstracts.append(paper["MedlineCitation"]["Article"]["Abstract"])
        else:
            abstracts.append("No abstract")
        # get authors
        al = paper['MedlineCitation']['Article']["AuthorList"]
        temp_authors = []
        for a in al:
            if "ForeName" in a and "LastName" in a:
                temp_authors.append([f"{a['ForeName']} {a['LastName']}"])
            elif "LastName" in a:
                temp_authors.append([a['LastName']])
            else:
                temp_authors.append(a)
        authors.append(list(pl.flatten(temp_authors)))
        # get dois and make link
        doi = [i for i in paper['PubmedData']['ArticleIdList'] if i.attributes["IdType"] == "doi"]
        if len(doi) > 0:
            doi = str(doi[0])
        else:
            doi = "NaN"
        dois.append(doi)
        links.append(f" https://doi.org/{doi}")
        # get pids
        pids.append(str([i for i in paper['PubmedData']['ArticleIdList'] if i.attributes["IdType"] == "pubmed"][0]))
        # get dates

        raw_date = paper['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']
        try:
            date = f"{raw_date['Year']}-{raw_date['Month']}-{raw_date['Day']}"
        except KeyError:
            date = raw_date
        dates.append(date)
        # get journal
        journals.append(paper['MedlineCitation']['Article']['Journal']['Title'])

    # sum up the results in a dict containing the search term as key
    final_dict[term] = (titles, authors, abstracts, dois, links, pids, dates, journals)

    # STEP 4: Make a word cloud picture from Pubmed search abstracts.
    if make_word_cloud:  # plot a picture of words in the found abstracts
        abstracts = []
        # for every search term (there usually is only one)
        for i in final_dict:
            # the abstract is the third element in the list
            abstracts.extend(
                abstract["AbstractText"][0] for abstract in final_dict[i][2] if not isinstance(abstract, str))
        # create a long string of abstract words
        abstracts = " ".join(abstracts)

        # when exlusion list is added in wordcloud add those
        # TODO properly implement the inclusion of exclusions
        el = ["sup"]

        fig = vis.wordcloud(text=abstracts, exlusion_words=el)
        if output != "print":
            if output[:-4] == ".txt":
                output = output[:-4]
            fig.to_file(output + "/WordCloud.png")

    # STEP 5: generate a txt or html output or print to the prompt
    if output == "print":
        for i in final_dict:
            for title, author, abstract, doi, link, pid, date, journal in \
                    zip(final_dict[i][0], final_dict[i][1], final_dict[i][2], final_dict[i][3], final_dict[i][4],
                        final_dict[i][5], final_dict[i][6],
                        final_dict[i][7]):
                print(title)
                print('-' * 100)
                print("; ".join(author))
                print('-' * 100)
                print(journal)
                print(date)
                print(f"doi: {doi}, PubMedID: {pid}")
                print(link)
                print('-' * 100)
                if isinstance(abstract, str):
                    print(abstract)
                else:
                    print(abstract["AbstractText"][0])
                print('-' * 100)
                print('*' * 100)
                print('-' * 100)

    elif output[-4:] == ".txt":
        with open(output, 'w', encoding="utf-8") as f:
            for i in final_dict:
                for title, author, abstract, doi, link, pid, date, journal in \
                        zip(final_dict[i][0], final_dict[i][1], final_dict[i][2], final_dict[i][3],
                            final_dict[i][4], final_dict[i][5],
                            final_dict[i][6], final_dict[i][7]):
                    f.write(title)
                    f.write('\n')
                    f.write('-' * 100)
                    f.write('\n')
                    f.write("; ".join(author))
                    f.write('\n')
                    f.write('-' * 100)
                    f.write('\n')
                    f.write(journal)
                    f.write('\n')
                    f.write(str(date))
                    f.write('\n')
                    f.write(f"doi: {doi}, PubMedID: {pid}")
                    f.write('\n')
                    f.write(link)
                    f.write('\n')
                    f.write('-' * 100)
                    f.write('\n')

                    # write the abstract in a structured manner with defined line breaks
                    if not isinstance(abstract, str):
                        abstract = abstract["AbstractText"][0]
                    abstract = abstract.split(" ")
                    for idx, word in enumerate(abstract):
                        f.write(word)
                        f.write(' ')
                        if idx % 20 == 0 and idx > 0:
                            f.write('\n')

                    f.write('\n')
                    f.write('-' * 100)
                    f.write('\n')
                    f.write('*' * 100)
                    f.write('\n')
                    f.write('-' * 100)
                    f.write('\n')

    # if the output is a html file or a folder, write html
    elif output[-5:] == ".html" or os.path.isdir(output):
        if os.path.isdir(output):
            output = os.path.join(output, "PubCrawlerResults.html")
        with open(output, 'w', encoding="utf-8") as f:
            f.write("<!DOC html>")
            f.write("<html>")
            f.write("<head>")
            f.write("<style>")
            f.write(".center {")
            f.write("display: block;")
            f.write("margin-left: auto;")
            f.write("margin-right: auto;")
            f.write("width: 80%;}")
            f.write("</style>")
            f.write("</head>")
            f.write('<body style="background-color:#FFE5B4">')
            if make_word_cloud:
                f.write('<img src="WordCloud.png" alt="WordCloud" class="center">')
            for i in final_dict:
                for title, author, abstract, doi, link, pid, date, journal in \
                        zip(final_dict[i][0], final_dict[i][1], final_dict[i][2], final_dict[i][3],
                            final_dict[i][4], final_dict[i][5],
                            final_dict[i][6], final_dict[i][7]):
                    f.write(f"<h2>{title}</h2>")
                    f.write('<br>')
                    ta = "; ".join(author)
                    f.write(f'<p style="color:gray; font-size:16px">{ta}</p>')
                    f.write('<br>')
                    f.write('-' * 200)
                    f.write('<br>')
                    f.write(f"<i>{journal}</i>")
                    f.write('<br>')
                    f.write(str(date))
                    f.write('<br>')
                    f.write(f"<i>doi:</i> {doi}, <i>PubMedID:</i> {pid}")
                    f.write('<br>')
                    f.write(f'<a href={link}><i>Link to journal</i></a>')
                    f.write('<br>')
                    f.write('-' * 200)
                    f.write('<br>')

                    # write the abstract in a structured manner with defined line breaks
                    if not isinstance(abstract, str):
                        abstract = abstract["AbstractText"][0]
                    abstract = abstract.split(" ")

                    f.write('<p style="color:#202020">')
                    for idx, word in enumerate(abstract):
                        f.write(word)
                        f.write(' ')
                        if idx % 20 == 0 and idx > 0:
                            f.write('</p>')
                            f.write('<p style="color:#202020">')
                    f.write('</p>')

                    f.write('<br>')
                    f.write('-' * 200)
                    f.write('<br>')
                    f.write('*' * 150)
                    f.write('<br>')
                    f.write('-' * 200)
                    f.write('<br>')
            f.write("</body>")
            f.write("</html>")


def loess(data, xvals, yvals, alpha, poly_degree=2):
    r"""
    Calculate a LOcally-Weighted Scatterplot Smoothing Fit.

    See: https://medium.com/@langen.mu/creating-powerfull-lowess-graphs-in-python-e0ea7a30b17a

    Parameters
    ----------
    data : pd.Dataframe
        Input dataframe.
    xvals : str
        Colname of x values.
    yvals : str
        Colname of y values.
    alpha : float
        Sensitivity of the estimation.
        Controls how much of the total number of values is used during weighing.
        0 <= alpha <= 1.
    poly_degree : int, optional
        Degree of the fitted polynomial. The default is 2.

    Returns
    -------
    None.

    Notes
    -----
    Loess normalisation (also referred to as Savitzky-Golay filter) locally approximates
    the data around every point using low-order functions and giving less weight to distant
    data points.

    Examples
    --------
    >>> np.random.seed(10)
    >>> x_values = np.random.randint(-50,110,size=250)
    >>> y_values = np.square(x_values)/1.5 + np.random.randint(-1000,1000, size=len(x_values))
    >>> df = pd.DataFrame({"Xvalue" : x_values,
                           "Yvalue" : y_values
                           })

    >>> evalDF = autoprot.analysis.loess(df, "Xvalue", "Yvalue", alpha=0.7, poly_degree=2)
    >>> fig, ax = plt.subplots(1,1)
    >>> sns.scatterplot(df["Xvalue"], df["Yvalue"], ax=ax)
    >>> ax.plot(eval_df['v'], eval_df['g'], color='red', linewidth= 3, label="Test")

    .. plot::
        :context: close-figs

        import autoprot.analysis as ana
        import seaborn as sns

        x_values = np.random.randint(-50,110,size=(250))
        y_values = np.square(x_values)/1.5 + np.random.randint(-1000,1000, size=len(x_values))
        df = pd.DataFrame({"Xvalue" : x_values,
                           "Yvalue" : y_values
                           })
        evalDF = ana.loess(df, "Xvalue", "Yvalue", alpha=0.7, poly_degree=2)
        fig, ax = plt.subplots(1,1)
        sns.scatterplot(df["Xvalue"], df["Yvalue"], ax=ax)
        ax.plot(evalDF['v'], evalDF['g'], color='red', linewidth= 3, label="Test")
        plt.show()
    """
    # generate x,y value pairs and sort them according to x
    all_data = sorted(zip(data[xvals].tolist(), data[yvals].tolist()), key=lambda x: x[0])
    # separate the values again into x and y cols
    xvals, yvals = zip(*all_data)
    # generate empty df for final fit
    eval_df = pd.DataFrame(columns=['v', 'g'])

    n = len(xvals)
    m = n + 1
    # how many data points to include in the weighing
    # alpha determines the relative proportion of values considered during weighing
    q = int(np.floor(n * alpha) if alpha <= 1.0 else n)
    # the average point to point distance in x direction
    avg_interval = ((max(xvals) - min(xvals)) / len(xvals))
    # calculate upper on lower boundaries
    v_lb = min(xvals) - (.5 * avg_interval)
    v_ub = (max(xvals) + (.5 * avg_interval))
    # coordinates for the fitting points
    v = enumerate(np.linspace(start=v_lb, stop=v_ub, num=m), start=1)
    # create an array of ones of the same length as xvals
    xcols = [np.ones_like(xvals)]

    for j in range(1, (poly_degree + 1)):
        xcols.append([i ** j for i in xvals])
    x_mtx = np.vstack(xcols).T
    for i in v:
        iterval = i[1]
        iterdists = sorted([(j, np.abs(j - iterval)) for j in xvals], key=lambda x: x[1])
        _, raw_dists = zip(*iterdists)
        scale_fact = raw_dists[q - 1]
        scaled_dists = [(j[0], (j[1] / scale_fact)) for j in iterdists]
        weights = [(j[0], ((1 - np.abs(j[1] ** 3)) ** 3 if j[1] <= 1 else 0)) for j in scaled_dists]
        _, weights = zip(*sorted(weights, key=lambda x: x[0]))
        _, raw_dists = zip(*sorted(iterdists, key=lambda x: x[0]))
        _, scaled_dists = zip(*sorted(scaled_dists, key=lambda x: x[0]))
        w = np.diag(weights)
        b = np.linalg.inv(x_mtx.T @ w @ x_mtx) @ (x_mtx.T @ w @ yvals)
        # loc_eval
        local_est = sum(i[1] * (iterval ** i[0]) for i in enumerate(b))
        iter_df2 = pd.DataFrame({
            'v': [iterval],
            'g': [local_est]
        })
        eval_df = pd.concat([eval_df, iter_df2])
    eval_df = eval_df[['v', 'g']]
    return eval_df


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


def limma(df, reps, cond="", custom_design=None, calc_contrasts=None, print_r=False):
    # sourcery skip: extract-method, inline-immediately-returned-variable
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
    calc_contrasts : str, optional
        The contrasts to be calculated by limma. Must refer to the column names
        in the data file. Differences are indicated e.g. by "CondA-CondB".
        The autoprot implementation only calculates contrasts based on the
        column names specified in the design matrix.
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
    matrix_a custom design representing the design matrix of the microarray experiment,
    with rows corresponding to arrays and columns to coefficients to be estimated
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

    command = [R, '--vanilla', RFUNCTIONS, "limma", data_loc, output_loc, test, design_loc, calc_contrasts or ""]

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


def go_analysis(gene_list, organism="hsapiens"):
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


def make_psm(seq, seq_len):
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

    m = pd.DataFrame(m, columns=aa_dic.keys())

    return m.T
