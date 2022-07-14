# -*- coding: utf-8 -*-
"""
Autoprot Analysis Functions.

@author: Wignand

@documentation: Julian
"""
import os
from subprocess import run, PIPE, STDOUT
from importlib import resources
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
from autoprot import visualization as vis
from autoprot import RHelper
import warnings
import missingno as msn
from gprofiler import GProfiler

gp = GProfiler(
    user_agent="autoprot",
    return_dataframe=True)
from autoprot import preprocessing as pp

# might want to enable warnings for debugging
# disabled them for copy vs view pandas warnings (pretty annoying)
warnings.filterwarnings('ignore')

RSCRIPT, R = RHelper.returnRPath()

# check where this is actually used and make it local
cmap = sns.diverging_palette(150, 275, s=80, l=55, n=9)


def ttest(df, reps, cond="", returnFC=True, adjustPVals=True,
          alternative='two-sided', logged=True):
    r"""
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
    returnFC : bool, optional
        Whether to calculate the fold-change of the provided data.
        The processing of the fold-change can be controlled by the returnLogFC
        switch.
        The default is True.
    adjustPVals : bool, optional
        Whether to adjust P-values. The default is True.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.
        The following options are available (default is 'two-sided'):

        * 'two-sided': the mean of the underlying distribution of the sample
          is different than the given population mean (`popmean`)
        * 'less': the mean of the underlying distribution of the sample is
          less than the given population mean (`popmean`)
        * 'greater': the mean of the underlying distribution of the sample is
          greater than the given population mean (`popmean`)
    logged: bool, optional
        Set to True if input values are log-transformed (or VSN normalised).
        This returns the difference between values as logFC, otherwise
        values are log2 transformed to gain logFC.
        Default is true.


    Returns
    -------
    df : pd.DataFrame
        Input dataframe with additional cols.

    Examples
    --------
    >>> twitchVsmild = ['log2_Ratio H/M normalized BC18_1','log2_Ratio M/L normalized BC18_2','log2_Ratio H/M normalized BC18_3',
    ...                 'log2_Ratio H/L normalized BC36_1','log2_Ratio H/M normalized BC36_2','log2_Ratio M/L normalized BC36_2']
    >>> protRatio = prot.filter(regex="Ratio .\/. normalized")
    >>> protLog = autoprot.preprocessing.log(prot, protRatio, base=2)
    >>> prot_tt = autoprot.analysis.ttest(df=protLog, reps=twitchVsmild, cond="TvM", returnFC=True, adjustPVals=True)
    >>> prot_tt["pValue_TvM"].hist(bins=50)
    >>> plt.show()

    .. plot::
        :context: close-figs

        import autoprot.analysis as ana
        import autoprot.preprocessing as pp
        import pandas as pd
        twitchVsmild = ['log2_Ratio H/M normalized BC18_1','log2_Ratio M/L normalized BC18_2','log2_Ratio H/M normalized BC18_3',
                        'log2_Ratio H/L normalized BC36_1','log2_Ratio H/M normalized BC36_2','log2_Ratio M/L normalized BC36_2']
        prot = pd.read_csv("_static/testdata/proteinGroups.zip", sep='\t', low_memory=False)
        protRatio = prot.filter(regex="Ratio .\/. normalized")
        protLog = pp.log(prot, protRatio, base=2)
        prot_tt = ana.ttest(df=protLog, reps=twitchVsmild, cond="TvM", returnFC=True, adjustPVals=True)
        prot_tt["pValue_TvM"].hist(bins=50)
        plt.show()

    >>> df = pd.DataFrame({"a1":np.random.normal(loc=0, size=4000),
    ...                    "a2":np.random.normal(loc=0, size=4000),
    ...                    "a3":np.random.normal(loc=0, size=4000),
    ...                    "b1":np.random.normal(loc=0.5, size=4000),
    ...                    "b2":np.random.normal(loc=0.5, size=4000),
    ...                    "b3":np.random.normal(loc=0.5, size=4000),})
    >>> autoprot.analysis.ttest(df=df,
                                reps=[["a1","a2", "a3"],["b1","b2", "b3"]])["pValue_"].hist(bins=50)
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
        ana.ttest(df=df,
                  reps=[["a1","a2", "a3"],["b1","b2", "b3"]])["pValue_"].hist(bins=50)
        plt.show()

    """

    def oneSamp_ttest(x):
        return np.ma.filled(ttest_1samp(x,
                                        nan_policy="omit",
                                        alternative=alternative,
                                        popmean=0)[1], np.nan)

    def twoSamp_ttest(x):
        return np.ma.filled(ttest_ind(x[:len(reps[0])],
                                      x[len(reps[0]):],
                                      alternative=alternative,
                                      nan_policy="omit")[1], np.nan)

    if isinstance(reps[0], list) and len(reps) == 2:
        print("Performing two-sample t-Test")
        df[f"pValue{cond}"] = df[reps[0] + reps[1]].apply(lambda x: twoSamp_ttest(x), 1).astype(float)
        df[f"score{cond}"] = -np.log10(df[f"pValue{cond}"])
        if returnFC == True:
            if logged:
                df[f"logFC{cond}"] = pd.DataFrame(df[reps[0]].values - df[reps[1]].values).mean(1).values
            else:
                df[f"logFC{cond}"] = np.log2(pd.DataFrame(df[reps[0]].values / df[reps[1]].values).mean(1)).values

    else:
        print("Performing one-sample t-Test")
        df[f"pValue{cond}"] = df[reps].apply(lambda x: oneSamp_ttest(x), 1).astype(float)
        df[f"score{cond}"] = -np.log10(df[f"pValue{cond}"])
        if returnFC == True:
            if logged:
                df[f"logFC{cond}"] = df[reps].mean(1)
            else:
                df[f"logFC{cond}"] = np.log2(df[reps].mean(1))

    if adjustPVals == True:
        adjustP(df, f"pValue{cond}")

    return df


def adjustP(df, pCol, method="fdr_bh"):
    r"""
    Use statsmodels.multitest on dataframes.

    Note: when nan in p-value this function will return only nan.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    pCol : str
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
    >>> twitchVsmild = ['log2_Ratio H/M normalized BC18_1','log2_Ratio M/L normalized BC18_2','log2_Ratio H/M normalized BC18_3',
    ...                 'log2_Ratio H/L normalized BC36_1','log2_Ratio H/M normalized BC36_2','log2_Ratio M/L normalized BC36_2']
    >>> prot = pd.read_csv("_static/testdata/proteinGroups.zip", sep='\t', low_memory=False)
    >>> protRatio = prot.filter(regex="Ratio .\/. normalized")
    >>> protLog = pp.log(prot, protRatio, base=2)
    >>> prot_tt = ana.ttest(df=protLog, reps=twitchVsmild, cond="TvM", mean=True, adjustPVals=False)
    >>> prot_tt_adj = ana.adjustP(prot_tt, pCol="pValue_TvM")
    >>> prot_tt_adj.filter(regex='pValue').head()
       pValue_TvM  adj.pValue_TvM
    0         NaN             NaN
    1    0.947334        0.966514
    2         NaN             NaN
    3         NaN             NaN
    4    0.031292        0.206977
    """
    # indices of rows containing values
    idx = df[df[pCol].notnull()].index
    # init new col with for adjusted p-values
    df[f"adj.{pCol}"] = np.nan
    # apply correction for selected rows
    df.loc[idx, f"adj.{pCol}"] = mt.multipletests(df[pCol].loc[idx], method=method)[1]
    return df


def cohenD(df, group1, group2):
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
    cohen_d = (abs(mean1 - mean2)) / sd_pooled
    df["cohenD"] = cohen_d
    return df


class autoPCA:
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

    >>> X = temp.filter(regex="log2.*norm.*_1$")

    generate appropiate names for the columns and rows of the matrix
    for example here the columns represent the conditions and we are not interested in the rows (which are the genes)

    >>> clabels = X.columns
    >>> rlabels = np.nan

    generate autopca object

    >>> autopca = autoprot.analysis.autoPCA(X, rlabels, clabels)

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
        X = temp.filter(regex="log2.*norm.*_1$")
        clabels = X.columns
        rlabels = np.nan
        autopca = ana.autoPCA(X, rlabels, clabels)
        autopca.scree()

    The corrComp heatmap shows the PCA loads (i.e. how much a principal component is
    influenced by a change in that variable) relative to the variables (i.e. the
    experiment conditions). If a weight (colorbar) is close to zero, the corresponding
    PC is barely influenced by it.

    >>> autopca.corrComp(annot=False)

    .. plot::
        :context: close-figs

        autopca.corrComp(annot=False)

    The bar loading plot is a different way to represent the weights/loads for each
    condition and principle component. High values indicate a high influence of the
    variable/condition on the PC.

    >>> autopca.barLoad(1)
    >>> autopca.barLoad(2)

    .. plot::
        :context: close-figs

        autopca.barLoad(1)
        autopca.barLoad(2)

    The score plot shows how the different data points (i.e. proteins) are positioned
    with respect to two principal components.
    In more detail, the scores are the original data values multiplied by the
    weights of each value for each principal component.
    Usually they will separate more in the direction of PC1 as this component
    explains the largest share of the data variance

    >>> autopca.scorePlot(pc1=1, pc2=2)

    .. plot::
        :context: close-figs

        autopca.scorePlot(pc1=1, pc2=2)

    The loading plot is the 2D representation of the barLoading plots and shows
    the weights how each variable influences the two PCs.

    >>> autopca.loadingPlot(pc1=1, pc2=2, labeling=True)

    .. plot::
        :context: close-figs

        autopca.loadingPlot(pc1=1, pc2=2, labeling=True)

    The Biplot is a combination of loading plot and score plot as it shows the
    scores for each protein as point and the weights for each variable as
    vectors.
    >>> autopca.biPlot(pc1=1, pc2=2)

    .. plot::
        :context: close-figs

        autopca.biPlot(pc1=1, pc2=2)
    """

    # =========================================================================
    # TODO
    # - Add interactive 3D scatter plot
    # - Facilitate naming of columns and rows
    # - Allow further customization of plots (e.g. figsize)
    # - Implement pair plot for multiple dimensions
    # =========================================================================
    def __init__(self, X, rlabels, clabels, batch=None):
        """
        Initialise PCA class.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe.
        rlabels : list of str
            Row labels.
        clabels : list of str
            Column labels.
        batch : list of str, optional
            Labels for distinct conditions used to colour dots in score plot.
            Must be the length of rlabels.
            The default is None.

        Returns
        -------
        None.

        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        # drop any rows in the dataframe containing missing values
        self.X = X.dropna()
        self.label = clabels
        self.rlabel = rlabels
        self.batch = batch
        # PCA is performed with the df containing missing values
        self.pca, self.forVis = self._performPCA(X, clabels)
        # generate scores from loadings
        self.Xt = self.pca.transform(self.X)
        self.expVar = self.pca.explained_variance_ratio_

    @staticmethod
    def _performPCA(X, label):
        """Perform pca and generate forVis dataframe."""
        pca = PCA().fit(X.dropna())
        # components_ is and ndarray of shape (n_components, n_features)
        # and contains the loadings/weights of each PCA eigenvector
        forVis = pd.DataFrame(pca.components_.T)
        forVis.columns = [f"PC{i}" for i in range(1, min(X.shape[0], X.T.shape[0]) + 1)]
        forVis["label"] = label
        return pca, forVis

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

        eigVal = self.pca.explained_variance_
        cumVar = np.append(np.array([0]), np.cumsum(self.expVar))

        plt.figure(figsize=figsize)
        plt.subplot(121)
        plt.plot(range(1, len(eigVal) + 1), eigVal, marker="o", color="teal",
                 markerfacecolor='purple')
        plt.ylabel("Eigenvalues")
        plt.xlabel("# Component")
        plt.title("Scree plot")
        sns.despine()

        plt.subplot(122)
        plt.plot(range(1, len(cumVar) + 1), cumVar, ds="steps", color="teal")
        plt.xticks(range(1, len(eigVal) + 1))
        plt.ylabel("explained cumulative variance")
        plt.xlabel("# Component")
        plt.title("Explained variance")
        sns.despine()

    def corrComp(self, annot=False):
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
        sns.heatmap(self.forVis.filter(regex="PC"), cmap=sns.color_palette("PuOr", 10), annot=annot)
        yp = [i + 0.5 for i in range(len(self.label))]
        plt.yticks(yp, self.forVis["label"], rotation=0);
        plt.title("")

    def barLoad(self, pc=1, n=25):
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
        PC = f"PC{pc}"
        forVis = self.forVis.copy()
        forVis[f"{PC}_abs"] = abs(forVis[PC])
        forVis["color"] = "negative"
        forVis.loc[forVis[PC] > 0, "color"] = "positive"
        forVis = forVis.sort_values(by=f"{PC}_abs", ascending=False)[:n]
        plt.figure()
        ax = plt.subplot()
        sns.barplot(x=forVis[PC], y=forVis["label"], hue=forVis["color"], alpha=.5,
                    hue_order=["negative", "positive"], palette=["teal", "purple"])
        ax.get_legend().remove()
        sns.despine()

    def returnLoad(self, pc=1, n=25):
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
        PC = f"PC{pc}"
        forVis = self.forVis.copy()
        forVis[f"{PC}_abs"] = abs(forVis[PC])
        forVis = forVis.sort_values(by=f"{PC}_abs", ascending=False)[:n]
        return forVis[[PC, "label"]]

    def returnScore(self):
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

    def scorePlot(self, pc1=1, pc2=2, labeling=False, file=None, figsize=(5, 5)):
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
            forVis = pd.DataFrame({"x": x, "y": y})
            sns.scatterplot(data=forVis, x="x", y="y")
        else:
            forVis = pd.DataFrame({"x": x, "y": y, "batch": self.batch})
            sns.scatterplot(data=forVis, x="x", y="y", hue=forVis["batch"])
        forVis["label"] = self.rlabel

        plt.title("Score plot")
        plt.xlabel(f"PC{pc1}\n{round(self.expVar[pc1 - 1] * 100, 2)} %")
        plt.ylabel(f"PC{pc2}\n{round(self.expVar[pc2 - 1] * 100, 2)} %")

        if labeling is True:
            ss = forVis["label"]
            xx = forVis["x"]
            yy = forVis["y"]
            for x, y, s in zip(xx, yy, ss):
                plt.text(x, y, s)
        sns.despine()

        if file is not None:
            plt.savefig(fr"{file}/ScorePlot.pdf")

    def loadingPlot(self, pc1=1, pc2=2, labeling=False, figsize=(5, 5)):
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

    def biPlot(self, pc1=1, pc2=2, numLoad="all", figsize=(5, 5), **kwargs):
        """
        Generate a biplot, a combined loadings and score plot.

        Parameters
        ----------
        pc1 : int, optional
            Number of the first PC to plot. The default is 1.
        pc2 : int, optional
            Number of the second PC to plot. The default is 2.
        numLoad : int, optional
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

        if numLoad == "all":
            loadings = temp[[f"PC{pc1}", f"PC{pc2}"]].values
            labels = temp["label"].values
        else:
            loadings = temp[[f"PC{pc1}", f"PC{pc2}"]].iloc[:numLoad].values
            labels = temp["label"].iloc[:numLoad].values

        xscale = 1.0 / (self.Xt[::, pc1 - 1].max() - self.Xt[::, pc1 - 1].min())
        yscale = 1.0 / (self.Xt[::, pc2 - 1].max() - self.Xt[::, pc2 - 1].min())
        xmina = 0
        xmaxa = 0
        ymina = 0

        for l, lab in zip(loadings, labels):
            # plt.plot([0,l[0]/xscale], (0, l[1]/yscale), color="purple")
            plt.arrow(x=0, y=0, dx=l[0] / xscale, dy=l[1] / yscale, color="purple",
                      head_width=.2)
            plt.text(x=l[0] / xscale, y=l[1] / yscale, s=lab)

            if l[0] / xscale < xmina:
                xmina = l[0] / xscale
            elif l[0] / xscale > xmaxa:
                xmaxa = l[0] / xscale

            if l[1] / yscale < ymina:
                ymina = l[1] / yscale
            elif l[1] / yscale > ymina:
                ymina = l[1] / yscale

        plt.xlabel(f"PC{pc1}\n{round(self.expVar[pc1 - 1] * 100, 2)} %")
        plt.ylabel(f"PC{pc2}\n{round(self.expVar[pc2 - 1] * 100, 2)} %")
        sns.despine()

    def pairPlot(self, n=0):
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

        forVis = pd.DataFrame(self.Xt[:, :n])
        i = np.argmin(self.Xt.shape)
        pcs = self.Xt.shape[i]
        forVis.columns = [f"PC {i}" for i in range(1, pcs + 1)]
        if self.batch is not None:
            forVis["batch"] = self.batch
            sns.pairplot(forVis, hue="batch")
        else:
            sns.pairplot(forVis)


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

    >>> ksea.annotate(organism="mouse", onlyInVivo=True)

    After the annotation it is always a good idea to get an overview of the
    kinases in the data an how many substrates the have. Based on this you
    might want to adjust a cutoff specifying the minimum number of substrates
    per kinase.

    >>> ksea.getKinaseOverview(kois=["Akt1","MKK4", "P38A", "Erk1"])

    Next, you can perform the actual kinase substrate enrichment analysis.
    The analysis is based on the log fold change of your data.
    Therefore, you have to provide the function with the appropiate column of
    your data and the minimum number of substrates per kinase.

    >>> ksea.ksea(col="logFC_TvC", minSubs=5)

    After the ksea has finished, you can get information for further analysis
    such as the substrates of a specific kinase (or a list of kinases)

    >>> ksea.returnKinaseSubstrate(kinase=["Akt1", "MKK4"]).sample() # doctest: +SKIP

    or a new dataframe with additional columns for every kinase showing if the
    protein is a substrate of that kinase or not

    >>> ksea.annotateDf(kinases=["Akt1", "MKK4"]).iloc[:2,-5:]

    Eventually, you can also generate plots of the enrichment analysis.

    >>> ksea.plotEnrichment(up_col="salmon",
    ...                     bg_col="pink",
    ...                     down_col="hotpink")

    .. plot::
        :context: close-figs

        import autoprot.preprocessing as pp
        import autoprot.analysis as ana
        import pandas as pd

        phos = pd.read_csv("_static/testdata/Phospho (STY)Sites_mod.zip", sep="\t", low_memory=False)
        phos = pp.cleaning(phos, file = "Phospho (STY)")
        phosRatio = phos.filter(regex="^Ratio .\/.( | normalized )R.___").columns
        phos = pp.log(phos, phosRatio, base=2)
        phos = pp.filterLocProb(phos, thresh=.75)
        phosRatio = phos.filter(regex="log2_Ratio .\/.( | normalized )R.___").columns
        phos = pp.removeNonQuant(phos, phosRatio)

        phosRatio = phos.filter(regex="log2_Ratio .\/. normalized R.___")
        phos_expanded = pp.expandSiteTable(phos, phosRatio)

        twitchVsmild = ['log2_Ratio H/M normalized R1','log2_Ratio M/L normalized R2','log2_Ratio H/M normalized R3',
                        'log2_Ratio H/L normalized R4','log2_Ratio H/M normalized R5','log2_Ratio M/L normalized R6']
        twitchVsctrl = ["log2_Ratio H/L normalized R1","log2_Ratio H/M normalized R2","log2_Ratio H/L normalized R3",
                        "log2_Ratio M/L normalized R4", "log2_Ratio H/L normalized R5","log2_Ratio H/M normalized R6"]

        phos = ana.ttest(df=phos_expanded, reps=twitchVsmild, cond="TvM", mean=True)
        phos = ana.ttest(df=phos_expanded, reps=twitchVsctrl, cond="TvC", mean=True)

        ksea = ana.KSEA(phos)
        ksea.annotate(organism="mouse", onlyInVivo=True)
        ksea.getKinaseOverview(kois=["Akt1","MKK4", "P38A", "Erk1"])
        ksea.ksea(col="logFC_TvC", minSubs=5)

        ksea.plotEnrichment(up_col="salmon",
                            bg_col="pink",
                            down_col="hotpink")

    You can also highlight a list of kinases in volcano plots.
    This is based on the autoprot volcano function.
    You can pass all the common parameters to this function.

    >>> ksea.volcanos(logFC="logFC_TvC", p="pValue_TvC", kinases=["Akt1", "MKK4"],
    ...               annot="Gene names", sig_col="gray")

    .. plot::
        :context: close-figs

        ksea.volcanos(logFC="logFC_TvC", p="pValue_TvC", kinases=["Akt1", "MKK4"],
                      annot="Gene names", sig_col="gray")

    Sometimes the enrichment is crowded by various kinase isoforms.
    In such cases it makes sense to simplify the annotation by grouping those
    isoforms together.

    >>> simplify = {"ERK":["ERK1","ERK2"],
    ...             "GSK3":["GSK3A", "GSK3B"]}
    >>> ksea.ksea(col="logFC_TvC", minSubs=5, simplify=simplify)
    >>> ksea.plotEnrichment()

    .. plot::
        :context: close-figs

        simplify = {"ERK":["ERK1","ERK2"],
                    "GSK3":["GSK3A", "GSK3B"]}
        ksea.ksea(col="logFC_TvC", minSubs=5, simplify=simplify)
        ksea.plotEnrichment()

    Of course you can also get the ksea results as a dataframe to save or to further customize.

    >>> ksea.returnEnrichment()

    Of course is the database not exhaustive and you might want to add additional
    substrates manually. This can be done the following way.
    Manually added substrates are always added irrespective of the species used
    for the annotation.

    >>> ksea = ana.KSEA(phos)
    >>> genes = ["RPGR"]
    >>> modRsds = ["S564"]
    >>> kinases = ["mTOR"]
    >>> ksea.addSubstrate(kinase=kinases, substrate=genes, subModRsd=modRsds)

    >>> ksea.annotate(organism="mouse", onlyInVivo=True)
    >>> ksea.ksea(col="logFC_TvC", minSubs=5)
    >>> ksea.plotEnrichment(plotBg=False)

    >>> ksea.removeManualSubs()
    >>> ksea.annotate(organism="mouse", onlyInVivo=True)
    >>> ksea.ksea(col="logFC_TvC", minSubs=5)
    >>> ksea.plotEnrichment(plotBg=False)
    """

    def __init__(self, data):
        """
        Initialise the KSEA object.

        Parameters
        ----------
        data : pd.DataFrame
            A phosphoproteomics datasaet.
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
        self.data = self.__preprocess(data.copy(deep=True))
        # init other class objects
        self.annotDf = None
        self.kseaResults = None
        self.koi = None
        self.simpleDf = None

    @staticmethod
    def __preprocess(data):
        """Define MOD_RSD, ucGene and mergeID cols in the input dataset."""
        # New column containing the modified residue as Ser201
        data["MOD_RSD"] = data["Amino acid"] + data["Position"].fillna(0).astype(int).astype(str)
        # The Gene names as defined for the Kinase substrate dataset
        data["ucGene"] = data["Gene names"].fillna("NA").apply(lambda x: x.upper())
        # an index column
        data["mergeID"] = range(data.shape[0])
        return data

    def __enrichment(self, df, col, kinase):
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
        KS = df[col][df["KINASE"].fillna('').apply(lambda x: kinase in x)]
        s = KS.mean()  # mean FC of kinase subs
        p = df[col].mean()  # mean FC of all substrates
        m = KS.shape[0]  # number of kinase substrates
        sig = df[col].std()  # standard dev of FC of all
        score = ((s - p) * np.sqrt(m)) / sig
        return [kinase, score]

    def __extractKois(self, df):
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
        # stirngs
        koi = [i for i in list(df["KINASE"].values.flatten()) if isinstance(i, str)]
        # remove duplicates
        ks = set(koi)
        # empty list to take on sets of kinase:count pairs
        temp = []
        for k in ks:
            # count the number of appearances of each kinase name in the list of kinase names
            temp.append((k, koi.count(k)))
        return pd.DataFrame(temp, columns=["Kinase", "#Subs"])

    def addSubstrate(self, kinase, substrate, subModRsd):
        """
        Manually add a substrate to the database.

        Parameters
        ----------
        kinase : list of str
            Name of the kinase e.g. PAK2.
        substrate : list of str
            Name of the substrate e.g. Prkd1.
        subModRsd : list of str
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
        it = iter([kinase, substrate, subModRsd])
        the_len = len(next(it))
        if not all(len(l) == the_len for l in it):
            raise ValueError('not all lists have same length!')

        # generate new empty df to fill in the new kinases
        temp = pd.DataFrame(columns=self.PSP_KS.columns)
        for i in range(len(kinase)):
            temp.loc[i, "KINASE"] = kinase[i]
            temp.loc[i, "SUB_GENE"] = substrate[i]
            temp.loc[i, "SUB_MOD_RSD"] = subModRsd[i]
            temp.loc[i, "source"] = "manual"
        # append to the original database from PSP
        self.PSP_KS = self.PSP_KS.append(temp, ignore_index=True)

    # TODO find a better name
    def removeManualSubs(self):
        """Remove all manual entries from the PSP database."""
        self.PSP_KS = self.PSP_KS[self.PSP_KS["source"] == "PSP"]

    def annotate(self, organism="human", onlyInVivo=False):
        """
        Annotate with known kinase substrate pairs.

        Parameters
        ----------
        organism : str, optional
            The target organism. The default is "human".
        onlyInVivo : bool, optional
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
        if onlyInVivo == True:
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
        self.koi = self.__extractKois(self.annotDf)

    def getKinaseOverview(self, kois=None):
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
            for j, k in enumerate(kois):
                try:
                    s = self.koi[self.koi["Kinase"].apply(lambda x: x.upper()) == k.upper()]["#Subs"].values[0]
                except:
                    s = 0
                ss = f"{k} has {s} substrates."
                ax[1].text(0.3, pos, ss)
                pos -= 0.055

    def ksea(self, col, minSubs=5, simplify=None):
        r"""
        Calculate Kinase Enrichment Score.

        Parameters
        ----------
        col : str
            Column used for the analysis containing the kinase substrate
            enrichments.
        minSubs : int, optional
            Minumum number of substrates a kinase must have to be considered.
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
            \frac{(\langle FC_{kinase} \rangle - \langle FC_{all} \rangle)\sqrt{N_{all}}}{\sigma_{all}}

        i.e. the difference in mean fold change between kinase and all substrates
        multiplied by the square root of number of kinase substrates and divided
        by the standard deviation of the fold change of all substrates.

        Returns
        -------
        None.

        """
        # TODO wouldnt it make more sense to perform simplification in the
        # Annotate function?
        copyAnnotDf = self.annotDf.copy(deep=True)
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
                copyAnnotDf["KINASE"].replace(simplify[key], [key] * len(simplify[key]), inplace=True)

            # drop rows which are now duplicates
            if "Multiplicity" in copyAnnotDf.columns:
                idx = copyAnnotDf[["ucGene", "MOD_RSD", "Multiplicity", "KINASE"]].drop_duplicates().index
            else:
                idx = copyAnnotDf[["ucGene", "MOD_RSD", "KINASE"]].drop_duplicates().index
            copyAnnotDf = copyAnnotDf.loc[idx]
            self.simpleDf = copyAnnotDf

            # repeat annotation with the simplified dataset
            self.koi = self.__extractKois(self.simpleDf)

        # filter kinases with at least minSubs number of substrates
        koi = self.koi[self.koi["#Subs"] >= minSubs]["Kinase"]

        # init empty df
        self.kseaResults = pd.DataFrame(columns=["kinase", "score"])
        # add the enrichment column back to the annotation df using the mergeID
        copyAnnotDf = copyAnnotDf.merge(self.data[[col, "mergeID"]], on="mergeID", how="left")
        for kinase in koi:
            # calculate the enrichment score
            k, s = self.__enrichment(copyAnnotDf[copyAnnotDf[col].notnull()], col, kinase)
            # new dataframe containing kinase names and scores
            temp = pd.DataFrame(data={"kinase": k, "score": s}, index=[0])
            # add the new df to the pre-initialised df
            self.kseaResults = self.kseaResults.append(temp, ignore_index=True)
        # sort the concatenated dfs by kinase enrichment score
        self.kseaResults = self.kseaResults.sort_values(by="score", ascending=False)

    def returnEnrichment(self):
        """Return a dataframe of kinase:score pairs."""
        if self.kseaResults is None:
            print("First perform the enrichment")
        else:
            # dropna in case of multiple columns in data
            # sometimes there are otherwise nan
            # nans are dropped in ksea enrichment
            return self.kseaResults.dropna()

    def plotEnrichment(self, up_col="orange", down_col="blue", bg_col="lightgray",
                       plotBg=True, ret=False, title="", figsize=(5, 10)):
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
        plotBg : bool, optional
            Whether or not to plot the unaffected kinases.
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
            # highlight up and downregulated
            self.kseaResults.loc[self.kseaResults["score"] > 2, "color"] = up_col
            self.kseaResults.loc[self.kseaResults["score"] < -2, "color"] = down_col
            # init figure
            fig = plt.figure(figsize=figsize)
            plt.yticks(fontsize=10)
            plt.title(title)
            # only plot the unaffected substrates if plotBg is True
            if plotBg == True:
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
            if ret == True:
                plt.tight_layout()
                return fig

    def volcanos(self, logFC, p, kinases=[], **kwargs):
        """
        Plot volcano plots highlighting substrates of a given kinase.

        Parameters
        ----------
        logFC : str
            Colname of column containing the log fold changes.
            Must be present in the dataframe KSEA was initialised with.
        p : str
            Colname of column containing the p values.
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
        df = self.annotateDf(kinases=kinases)
        for k in kinases:
            # index for highlighting the selected kinase substrates
            idx = df[df[k] == 1].index
            vis.volcano(df, logFC, p=p, highlight=idx,
                        custom_hl={"label": k},
                        custom_fg={"alpha": .5},
                        **kwargs)

    def returnKinaseSubstrate(self, kinase):
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
        dfFilter : pd.Dataframe
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
            idx = []
            # find the rows corresponding to each kinase
            for k in kinase:
                idx.append(df[df["KINASE"].fillna("NA").apply(lambda x: x.upper()) == k.upper()].index)
            # merge all row indices and use them to create a sub-df containing
            # only the kinases of interest
            dfFilter = df.loc[pl.flatten(idx)]
        # if only a single kinase is provided, filter the input df directly
        elif isinstance(kinase, str):
            dfFilter = df[df["KINASE"].fillna("NA").apply(lambda x: x.upper()) == kinase.upper()]
        else:
            raise ValueError("Please provide either a string or a list of strings representing kinases of interest.")

        # data are merged implicitely on common colnames i.e. on SITE_GRP_ID
        # only entries present in the filtered annotDfare retained
        dfFilter = pd.merge(dfFilter[['GENE', 'KINASE', 'KIN_ACC_ID', 'SUBSTRATE', 'SUB_ACC_ID',
                                      'SUB_GENE', 'SUB_MOD_RSD', 'SITE_GRP_ID', 'SITE_+/-7_AA', 'DOMAIN', 'IN_VIVO_RXN',
                                      'IN_VITRO_RXN', 'CST_CAT#',
                                      'source', "mergeID"]],
                            self.PSP_regSits[['SITE_GRP_ID', 'ON_FUNCTION', 'ON_PROCESS', 'ON_PROT_INTERACT',
                                              'ON_OTHER_INTERACT', 'PMIDs', 'LT_LIT', 'MS_LIT', 'MS_CST',
                                              'NOTES']],
                            how="left")
        return dfFilter

    def annotateDf(self, kinases=[]):
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
        if len(kinases) > 0:
            # remove the two columns from the returned df
            df = self.data.drop(["MOD_RSD", "ucGene"], axis=1)
            for kinase in kinases:
                # find substrates for the given kinase in the dataset
                ids = self.returnKinaseSubstrate(kinase)["mergeID"]
                # init the boolean column with zeros
                df[kinase] = 0
                # check if the unique ID for each protein is present in the
                # returnKinaseSubstrate df. If so set the column value to 1.
                df.loc[df["mergeID"].isin(ids), kinase] = 1
            # remnove also the mergeID column before returning the df
            return df.drop("mergeID", axis=1)
        else:
            print("Please provide kinase(s) for annotation.")


def missAnalysis(df, cols, n=None, sort='ascending', text=True, vis=True,
                 extraVis=False, saveDir=None):
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
    extraVis : bool, optional
        Whether to return matrix plot showing missingness.
        The default is False.
    saveDir : str, optional
        Path to folder where the results should be saved.
        The default is None.

    Raises
    ------
    ValueError
        If n is incorrectly specified.

    Returns
    -------
    None.

    Examples
    --------
    miss_analysis gives a quick overview of the missingness of the provided
    dataframe. You can provide the complete or prefiltered dataframe as input.
    Providing n allows you to specify how many of the entries of the dataframe
    (sorted by missingness) are displayed (i.e. only display the n columns with
    most (or least) missing values) With the sort argument you can define
    whether the dataframe is sorted by least to most missing values or vice versa
    (using "descending" and "ascending", respectively). The vis and extra_vis
    arguments can be used to toggle the graphical output.
    In case of large data (a lot of columns) those might be better turned off.

    >>> autoprot.analysis.missAnalysis(phos_expanded,
    ...                                twitchVsctrl+twitchVsmild+mildVsctrl,
    ...                                sort="descending",
    ...                                extraVis = True)

    .. plot::
        :context: close-figs

        import autoprot.preprocessing as pp
        import autoprot.analysis as ana
        import pandas as pd

        phos = pd.read_csv("_static/testdata/Phospho (STY)Sites_mod.zip", sep="\t", low_memory=False)
        phos = pp.cleaning(phos, file = "Phospho (STY)")
        phosRatio = phos.filter(regex="^Ratio .\/.( | normalized )R.___").columns
        phos = pp.log(phos, phosRatio, base=2)
        phos = pp.filterLocProb(phos, thresh=.75)
        phosRatio = phos.filter(regex="log2_Ratio .\/.( | normalized )R.___").columns
        phos = pp.removeNonQuant(phos, phosRatio)

        phosRatio = phos.filter(regex="log2_Ratio .\/. normalized R.___")
        phos_expanded = pp.expandSiteTable(phos, phosRatio)

        twitchVsmild = ['log2_Ratio H/M normalized R1','log2_Ratio M/L normalized R2','log2_Ratio H/M normalized R3',
                        'log2_Ratio H/L normalized R4','log2_Ratio H/M normalized R5','log2_Ratio M/L normalized R6']
        twitchVsctrl = ["log2_Ratio H/L normalized R1","log2_Ratio H/M normalized R2","log2_Ratio H/L normalized R3",
                        "log2_Ratio M/L normalized R4", "log2_Ratio H/L normalized R5","log2_Ratio H/M normalized R6"]
        mildVsctrl = ["log2_Ratio M/L normalized R1","log2_Ratio H/L normalized R2","log2_Ratio M/L normalized R3",
                      "log2_Ratio H/M normalized R4","log2_Ratio M/L normalized R5","log2_Ratio H/L normalized R6"]
        phos = ana.ttest(df=phos_expanded, reps=twitchVsmild, cond="TvM", mean=True)
        phos = ana.ttest(df=phos_expanded, reps=twitchVsctrl, cond="TvC", mean=True)

        ana.missAnalysis(phos_expanded,
                         twitchVsctrl+twitchVsmild+mildVsctrl,
                         text=False,
                         sort="descending",
                         extraVis = True)
    """

    def create_data(df):
        """Calculate summary missing statistics."""
        data = []
        # implicitly iterate over df cols
        for i in df:
            # len df
            n = df.shape[0]
            # how many are missing
            m = df[i].isnull().sum()
            # percentage
            p = m / n * 100
            data.append([i, n, m, p])
        data = add_ranking(data)
        return data

    def add_ranking(data):
        """Sort data by the percentage of missingness."""
        data = sorted(data, key=itemgetter(3))
        # add a number corresponding to the position in the ranking
        # to every condition aka column.
        for idx, col in enumerate(data):
            col.append(idx)
        return data

    def describe(data, n, saveDir, sort='ascending'):
        """
        Print summary statistics as text based on data df.

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe with columns [colname,total_n, n_missing, percentage, rank]
        n : int
            How many entries are displayed.
        saveDir : str
            Path to dump txt output.
        sort : str, optional
            How to sort data.
            Possible values are 'ascending' and 'descending'.
            The default is 'ascending'.

        Raises
        ------
        ValueError
            If n is not properly chosen.

        Returns
        -------
        bool
            True if successful.

        """
        if n == None:
            n = len(data)
        elif n > len(data):
            print("'n' is larger than dataframe!\nDisplaying complete dataframe.")
            n = len(data)
        if n < 0:
            raise ValueError("'n' has to be a positive integer!")

        if sort == 'ascending':
            pass
        elif sort == 'descending':
            data = data[::-1]

        allines = ''
        for i in range(n):
            allines += "{} has {} of {} entries missing ({}%).".format(data[i][0],
                                                                       data[i][2],
                                                                       data[i][1],
                                                                       round(data[i][3], 2))
            allines += '\n'
            # line separator
            allines += '-' * 80

        if saveDir:
            with open(saveDir + "/missAnalysis_text.txt", 'w') as f:
                for i in range(n):
                    f.write(allines)

        # write all lines at once
        print(allines)

        return True

    def visualize(data, n, saveDir, sort='ascending'):
        """
        Visualize the % missingness of first n entries of dataframe as a bar plot.

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe with columns [colname,total_n, n_missing, percentage, rank]
        n : int
            How many entries are displayed.
            If None, all entries are displayed.
            Default is None.
        saveDir : str
            Path to dump txt output.
        sort : str, optional
            How to sort data.
            Possible values are 'ascending' and 'descending'.
            The default is 'ascending'.

        Raises
        ------
        ValueError
            If the n is incorrect.

        Returns
        -------
        bool
            True if function successful.

        """
        if n == None:
            n = len(data)
        elif n > len(data):
            print("'n' is larger than dataframe!\nDisplaying complete dataframe.")
            n = len(data)
        if n < 0:
            raise ValueError("'n' has to be a positive integer!")

        if sort == 'ascending':
            pass
        elif sort == 'descending':
            data = data[::-1]

        data = pd.DataFrame(data=data,
                            columns=["Name", "tot_values", "tot_miss", "perc_miss", "rank"])

        plt.figure(figsize=(7, 7))
        ax = plt.subplot()
        # plot colname against total missing values
        splot = sns.barplot(x=data["tot_miss"].iloc[:n],
                            y=data["Name"].iloc[:n])

        # add the percentage of missingness to every bar of the plot
        for idx, p in enumerate(splot.patches):
            s = str(round(data.iloc[idx, 3], 2)) + '%'
            x = p.get_width() + p.get_width() * .01
            y = p.get_y() + p.get_height() / 2
            splot.annotate(s, (x, y))

        plt.title("Missing values of dataframe columns.")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel("")

        if saveDir:
            plt.savefig(saveDir + "/missAnalysis_vis1.pdf")

        return True

    def visualize_extra(df, saveDir):
        """
        Visualize the missingness in the dataset using missingno.

        Notes
        -----
        Plots are generated with missingno.matrix.
        See https://github.com/ResidentMario/missingno

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to check for missing values.
        saveDir : str
            Path to save the figures.

        Returns
        -------
        bool
            True if successful.

        """
        fig, ax = plt.subplots(1)
        msn.matrix(df, sort="ascending", ax=ax)
        if saveDir:
            plt.savefig(saveDir + "/missAnalysis_vis2.pdf")
        return True

    # only analyse subset of cols
    df = df[cols]
    # sorted list of lists with every sublist containing
    # [colname,total_n, n_missing, percentage, rank]
    data = create_data(df)
    if text == True:
        # print summary statistics and saves them to file
        describe(data, n, saveDir, sort)
    if vis == True:
        # generate a bar plot of percent missingness vs colname
        visualize(data, n, saveDir, sort)
    if extraVis == True:
        # plot a fancy missingness matrix for the original dataframe
        visualize_extra(df, saveDir)


def getPubAbstracts(text=[""], ToA=[""], author=[""], phrase=[""],
                    exclusion=[""], customSearchTerm=None, makeWordCloud=False,
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
        The default is [""].
    ToA : list of str, optional
        List of words which must occur in Titel or Abstract.
        The default is [""].
    author : list of str, optional
        List of authors which must occor in Author list.
        The default is [""].
    phrase : list of str, optional
        List of phrases which should occur in text.
        Seperate word in phrases with hyphen (e.g. akt-signaling).
        The default is [""].
    exclusion : list of str, optional
        List of words which must not occur in text.
        The default is [""].
    customSearchTerm : str, optional
        Enter a custom search term (make use of advanced pubmed search functionality).
        If this is supplied all other search terms will be ignored.
        The default is None.
    makeWordCloud : bool, optional
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

    >>> autoprot.analysis.getPubAbstracts(ToA=["p38", "JNK", "ERK"],
                                          makeWordCloud=True)

    .. plot::
        :context: close-figs

        import autoprot.analysis as ana
        ana.getPubAbstracts(ToA=["p38", "JNK", "ERK"],
                            makeWordCloud=True)

    Even more comfortably, you can also save the results incl. the wordcloud
    as html file

    >>>  autoprot.analysis.getPubAbstracts(ToA=["p38", "JNK", "ERK"],
                                          makeWordCloud=True,
                                          output='./MyPubmedSearch.html')

    """

    def search(query, retmax, sort):
        """
        Perform a PubMed search.

        Parameters
        ----------
        query : str
            PubMed search term.
        retmax : int
            Maximum number of items to return.
        sort : str
            By which to sort the results.

        Returns
        -------
        results : Bio.Entrez.Parser.DictionaryElement
            A dictionary holding a summary of the search results.

        """
        # seemingly no true mail address is required.
        Entrez.email = 'your.email@example.com'
        handle = Entrez.esearch(db='pubmed',
                                sort=sort,
                                retmax=str(retmax),
                                retmode='xml',
                                term=query)
        results = Entrez.read(handle)
        return results

    def fetchDetails(id_list):
        """
        Get detailed information on articles from their PubMed IDs.

        Parameters
        ----------
        id_list : list of str
            List of PubMed IDs.

        Returns
        -------
        results : Bio.Entrez.Parser.DictionaryElement
            A dictionary holding detailed infos on the articles.

        """
        ids = ','.join(id_list)
        Entrez.email = 'your.email@example.com'
        handle = Entrez.efetch(db='pubmed',
                               retmode='xml',
                               id=ids)
        results = Entrez.read(handle)
        return results

    def makeSearchTerm(text, ToA, author, phrase, exclusion):
        """
        Generate PubMed search term.

        Parameters
        ----------
        text : list of str, optional
            List of words which must occur in text.
        ToA : list of str, optional
            List of words which must occur in Titel or Abstract.
        author : list of str, optional
            List of authors which must occor in Author list.
        phrase : list of str, optional
            List of phrases which should occur in text.
            Seperate word in phrases with hyphen (e.g. akt-signaling).
        exclusion : list of str, optional
            List of words which must not occur in text.

        Returns
        -------
        term : str
            A PubMed search term.

        """
        # Generate long list of concatenated search terms
        term = [i + "[Title/Abstract]" if i != "" else "" for i in ToA] + \
               [i + "[Text Word]" if i != "" else "" for i in text] + \
               [i + "[Author]" if i != "" else "" for i in author] + \
               [i if i != "" else "" for i in phrase]
        term = (" AND ").join([i for i in term if i != ""])

        # add exclusions joined by NOT
        if exclusion != [""]:
            exclusion = (" NOT ").join(exclusion)
            term = (" NOT ").join([term] + [exclusion])
            return term
        else:
            return term

    def makewc(final, output):
        """
        Make a word cloud picture from Pubmed search abstracts.

        Parameters
        ----------
        final : dict
            Dictionary mapping a search term to detailed results.
        output : str
            Path to the dir for saving the images.

        Returns
        -------
        None.

        """
        abstracts = []
        # for every search term (there usually is only one)
        for i in final:
            # the abstract is the third element in the list
            for abstract in final[i][2]:
                # if there was no abstract found, the value will be 'no abstract'
                if not isinstance(abstract, str):
                    abstracts.append(abstract["AbstractText"][0])
        abstracts = (" ").join(abstracts)

        # when exlusion list is added in wordcloud add those
        # TODO properly implement the inclusion of exclusions
        el = ["sup"]

        fig = vis.wordcloud(text=abstracts, exlusionwords=el)
        if output != "print":
            if output[:-4] == ".txt":
                output = output[:-4]
            fig.to_file(output + "/WordCloud.png")

    def genOutput(final, output, makeWordCloud):
        """
        Return results summary.

        Parameters
        ----------
        final : dict
            Dictionary mapping a search term to detailed results.
        output : str
            If 'print' returns the results on the prompt.
            If path to txt-file: saves as txt file
            If path to html-files: saves as html file.
            If path is a folder: save as html inside that folder.
        makeWordCloud : bool
            Include wordcloud in HTML output.
            Note this only works for html output paths.

        Returns
        -------
        None.

        """
        if output == "print":
            for i in final:
                for title, author, abstract, doi, link, pid, date, journal in \
                        zip(final[i][0], final[i][1], final[i][2], final[i][3], final[i][4], final[i][5], final[i][6],
                            final[i][7]):
                    print(title)
                    print('-' * 100)
                    print(("; ").join(author))
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
                for i in final:
                    for title, author, abstract, doi, link, pid, date, journal in \
                            zip(final[i][0], final[i][1], final[i][2], final[i][3], final[i][4], final[i][5],
                                final[i][6], final[i][7]):
                        f.write(title)
                        f.write('\n')
                        f.write('-' * 100)
                        f.write('\n')
                        f.write(("; ").join(author))
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
                        writeAbstract(abstract, f, "txt")
                        f.write('\n')
                        f.write('-' * 100)
                        f.write('\n')
                        f.write('*' * 100)
                        f.write('\n')
                        f.write('-' * 100)
                        f.write('\n')

        # if the output is an html file or a folder, write html
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
                if makeWordCloud == True:
                    f.write('<img src="WordCloud.png" alt="WordCloud" class="center">')
                for i in final:
                    for title, author, abstract, doi, link, pid, date, journal in \
                            zip(final[i][0], final[i][1], final[i][2], final[i][3], final[i][4], final[i][5],
                                final[i][6], final[i][7]):
                        f.write(f"<h2>{title}</h2>")
                        f.write('<br>')
                        ta = ("; ").join(author)
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
                        writeAbstract(abstract, f, "html")
                        f.write('<br>')
                        f.write('-' * 200)
                        f.write('<br>')
                        f.write('*' * 150)
                        f.write('<br>')
                        f.write('-' * 200)
                        f.write('<br>')
                f.write("</body>")
                f.write("</html>")

    def writeAbstract(abstract, f, typ):
        """
        Write abstracts in blocks to avoid long lines.

        Parameters
        ----------
        abstract : list of str
            A list containing the abstract on position 0.
        f : filehandler
            filehandler holding the output file.
        typ : str
            'html' or 'txt.

        Returns
        -------
        None.

        """
        if not isinstance(abstract, str):
            abstract = abstract["AbstractText"][0]
        abstract = abstract.split(" ")

        if typ == "txt":
            for idx, word in enumerate(abstract):
                f.write(word)
                f.write(' ')
                if idx % 20 == 0 and idx > 0:
                    f.write('\n')

        elif typ == "html":
            f.write('<p style="color:#202020">')
            for idx, word in enumerate(abstract):
                f.write(word)
                f.write(' ')
                if idx % 20 == 0 and idx > 0:
                    f.write('</p>')
                    f.write('<p style="color:#202020">')
            f.write('</p>')

    # if function is executed in a loop a small deleay to ensure pubmed is responding
    time.sleep(0.5)

    if customSearchTerm is None:
        # generate a pubmed search term from user input
        term = makeSearchTerm(text, ToA, author, phrase, exclusion)
    else:
        # if the user provides a custom searhc term, ignore all other input
        term = customSearchTerm

    # Perform the PubMed search and return the PubMed IDs of the found articles
    results = search(term, retmax, sort)["IdList"]

    # if the search was successful
    if len(results) > 0:
        # get more detailed results including abstracts, authors etc
        results2 = fetchDetails(set(results))

        titles = []
        abstracts = []
        authors = []
        dois = []
        links = []
        pids = []
        dates = []
        journals = []
        final = dict()

        # iterate through the articles
        for paper in results2["PubmedArticle"]:
            # get titles
            titles.append(paper["MedlineCitation"]["Article"]["ArticleTitle"])
            # get abstract
            if "Abstract" in paper["MedlineCitation"]["Article"].keys():
                abstracts.append(paper["MedlineCitation"]["Article"]["Abstract"])
            else:
                abstracts.append("No abstract")
            # get authors
            al = paper['MedlineCitation']['Article']["AuthorList"]
            tempAuthors = []
            for a in al:
                if "ForeName" in a and "LastName" in a:
                    tempAuthors.append([f"{a['ForeName']} {a['LastName']}"])
                elif "LastName" in a:
                    tempAuthors.append([a['LastName']])
                else:
                    tempAuthors.append(a)
            authors.append(list(pl.flatten(tempAuthors)))
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

            rawDate = paper['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']
            try:
                date = f"{rawDate['Year']}-{rawDate['Month']}-{rawDate['Day']}"
            except:
                date = rawDate
            dates.append(date)
            # get journal
            journals.append(paper['MedlineCitation']['Article']['Journal']['Title'])

        # sum up the results in a dict containing the search term as key
        final[term] = (titles, authors, abstracts, dois, links, pids, dates, journals)

        if makeWordCloud == True:
            # plot a picture of words in the found abstracts
            makewc(final, output)

        # generate a txt or html output or print to the prompt
        genOutput(final, output, makeWordCloud)

    else:
        print("No results for " + term)


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
    >>> x_values = np.random.randint(-50,110,size=(250))
    >>> y_values = np.square(x_values)/1.5 + np.random.randint(-1000,1000, size=len(x_values))
    >>> df = pd.DataFrame({"Xvalue" : x_values,
                           "Yvalue" : y_values
                           })

    >>> evalDF = autoprot.analysis.loess(df, "Xvalue", "Yvalue", alpha=0.7, poly_degree=2)
    >>> fig, ax = plt.subplots(1,1)
    >>> sns.scatterplot(df["Xvalue"], df["Yvalue"], ax=ax)
    >>> ax.plot(evalDF['v'], evalDF['g'], color='red', linewidth= 3, label="Test")

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

    def loc_eval(x, b):
        loc_est = 0
        for i in enumerate(b): loc_est += i[1] * (x ** i[0])
        return (loc_est)

    # generate x,y value pairs and sort them according to x
    all_data = sorted(zip(data[xvals].tolist(), data[yvals].tolist()), key=lambda x: x[0])
    # separate the values again into x and y cols
    xvals, yvals = zip(*all_data)
    # generate empty df for final fit
    evalDF = pd.DataFrame(columns=['v', 'g'])

    n = len(xvals)
    m = n + 1
    # how many datapoints to include in the weighing
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
    X = np.vstack(xcols).T
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
        W = np.diag(weights)
        b = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ yvals)
        local_est = loc_eval(iterval, b)
        iterDF2 = pd.DataFrame({
            'v': [iterval],
            'g': [local_est]
        })
        evalDF = pd.concat([evalDF, iterDF2])
    evalDF = evalDF[['v', 'g']]
    return (evalDF)


def edm(A, B):
    """
    Calculate an euclidean distance matrix between two matrices.

    See:  https://medium.com/swlh/euclidean-distance-matrix-4c3e1378d87f

    Parameters
    ----------
    A : np.ndarray
        Matrix 1.
    B : np.ndarray
        Matrix 2.

    Returns
    -------
    np.ndarray
        Distance matrix.

    """
    p1 = np.sum(A ** 2, axis=1)[:, np.newaxis]
    p2 = np.sum(B ** 2, axis=1)
    p3 = -2 * np.dot(A, B.T)
    return np.sqrt(p1 + p2 + p3)


def limma(df, reps, cond="", customDesign=None, calcContrasts=None, print_r=False):
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
    customDesign : str, optional
        Path to custom design file.
        The default is None.
    calcContrasts : str, optional
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
    A custom design representing the design matrix of the microarray experiment,
    with rows corresponding to arrays and columns to coefficients to be estimated
    can be provided using customDesign.
    If customDesign is the unit vector meaning that the arrays are treated as replicates.
    See: https://www.rdocumentation.org/packages/limma/versions/3.28.14/topics/lmFit

    Examples
    --------
    >>> df = pd.DataFrame({"a1":np.random.normal(loc=0, size=4000),
    ...                    "a2":np.random.normal(loc=0, size=4000),
    ...                    "a3":np.random.normal(loc=0, size=4000),
    ...                    "b1":np.random.normal(loc=0.5, size=4000),
    ...                    "b2":np.random.normal(loc=0.5, size=4000),
    ...                    "b3":np.random.normal(loc=0.5, size=4000),})
    >>> testRes = ana.limma(df, reps=[["a1","a2", "a3"],["b1","b2", "b3"]], cond="_test")
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
    dataLoc = d + "/input.csv"
    outputLoc = d + "/output.csv"
    if customDesign is None:
        designLoc = d + "/design.csv"

    if "UID" in df.columns:
        pass
    else:
        df["UID"] = range(1, df.shape[0] + 1)

    # flatten in case of two sample
    pp.to_csv(df[["UID"] + list(pl.flatten(reps))], dataLoc)

    # Normally no customDesign is provided
    if customDesign is None:
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
            pp.to_csv(design, designLoc)
        else:
            test = "oneSample"
            print("LIMMA: Assuming a one sample test")
            # The R function will generate a design matrix corresponding to
            # ones
    # if there is a custom design matrix, use it
    else:
        print("LIMMA: Assuming a custom design test with:")
        print(f"Design specified at {customDesign}")
        print("Columns: {}".format('\n\t'.join(list(pl.flatten(reps)))))

        design = pd.read_csv(customDesign, sep='\t')
        print("Using design matrix:\n")
        print(design.to_markdown())

        test = "custom"
        designLoc = customDesign

    command = [R, '--vanilla',
               RSCRIPT,  # script location
               "limma",  # functionName
               dataLoc,  # data location
               outputLoc,  # output file,
               test,  # kind of test
               designLoc,  # design location
               calcContrasts if calcContrasts else ""  # whether to calculate contrasts
               ]

    p = run(command,
            stdout=PIPE,
            stderr=STDOUT,
            universal_newlines=True)

    if print_r:
        print(p.stdout)

    res = pp.read_csv(outputLoc)
    res.columns = [i + cond if i != "UID" else i for i in res.columns]
    # this keeps the index of the original df in the returned df
    df = df.reset_index().merge(res, on="UID").set_index('index')

    os.remove(dataLoc)
    os.remove(outputLoc)
    if customDesign is None:
        if isinstance(reps[0], list) and len(reps) == 2:
            os.remove(designLoc)

    return df


def rankProd(df, reps, cond="", print_r=False, correct_fc=True):
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
    dataLoc = d + "/input.csv"
    outputLoc = d + "/output.csv"

    if "UID" in df.columns:
        pass
    else:
        df["UID"] = range(1, df.shape[0] + 1)

    if isinstance(reps[0], list) and len(reps) == 2:
        class_labels = [0, ] * len(reps[0]) + [1, ] * len(reps[1])
        print("rankProd: Assuming a two sample test with:")
        print("Sample 1: {}".format(', '.join(['\n\t' + x for x in reps[0]])))
        print("Sample 2: {}".format(', '.join(['\n\t' + x for x in reps[1]])))
        print("Class labels: {}".format(', '.join([str(x) for x in class_labels])))

    else:
        print("rankProd: Assuming a one sample test")
        class_labels = [1, ] * len(reps)

    # flatten in case of two sample
    pp.to_csv(df[["UID"] + list(pl.flatten(reps))], dataLoc)

    command = [R, '--vanilla',
               RSCRIPT,  # script location
               "rankProd",  # functionName
               dataLoc,  # data location
               outputLoc,  # output file,
               ','.join([str(x) for x in class_labels]),
               ]

    p = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)

    if print_r:
        print(p.stdout)

    res = pp.read_csv(outputLoc)
    res.columns = [i + cond if i != "UID" else i for i in res.columns]
    df = df.reset_index().merge(res, on="UID").set_index('index')

    if isinstance(reps[0], list) and len(reps) == 2:
        df['logFC' + cond] = df[reps[0]].mean(axis=1, skipna=True) - df[reps[1]].mean(axis=1, skipna=True)
    else:
        df['logFC' + cond] = df[reps].mean(axis=1)

    os.remove(dataLoc)
    os.remove(outputLoc)

    return df


def annotatePS(df, ps, colsToKeep=[]):
    """
    Annotate phosphosites with information derived from PhosphositePlus.

    Parameters
    ----------
    df : pd.Dataframe
        dataframe containing PS of interst.
    ps : str
        Column containing info about the PS.
        Format: GeneName_AminoacidPositoin (e.g. AKT_T308).
    colsToKeep : list of str, optional
        Which columns from original dataframe (input df) to keep in output.
        The default is [].

    Returns
    -------
    pd.Dataframe
        The input dataframe with the kept columns and additional phosphosite cols.

    """

    def makeMergeCol(df, file="regSites"):
        """Format the phosphosite positions and gene names so that merging is possible."""
        if file == "regSites":
            return df["GENE"].fillna("").apply(lambda x: str(x).upper()) + '_' + df["MOD_RSD"].fillna("").apply(
                lambda x: x.split('-')[0])
        return df["SUB_GENE"].fillna("").apply(lambda x: str(x).upper()) + '_' + df["SUB_MOD_RSD"].fillna("")

    with resources.open_binary("autoprot.data", "Kinase_Substrate_Dataset.zip") as d:
        KS = pd.read_csv(d, sep='\t', compression='zip')
        KS["merge"] = makeMergeCol(KS, "KS")
    with resources.open_binary("autoprot.data", "Regulatory_sites.zip") as d:
        regSites = pd.read_csv(d, sep='\t', compression='zip')
        regSites["merge"] = makeMergeCol(regSites)

    KS_coi = ['KINASE', 'DOMAIN', 'IN_VIVO_RXN', 'IN_VITRO_RXN', 'CST_CAT#', 'merge']
    regSites_coi = ['ON_FUNCTION', 'ON_PROCESS', 'ON_PROT_INTERACT', 'ON_OTHER_INTERACT',
                    'PMIDs', 'NOTES', 'LT_LIT', 'MS_LIT', 'MS_CST', 'merge']

    df = df.copy(deep=True)
    df.rename(columns={ps: "merge"}, inplace=True)
    df = df[["merge"] + colsToKeep]
    df = df.merge(KS[KS_coi], on="merge", how="left")
    df = df.merge(regSites[regSites_coi], on="merge", how="left")

    return df


def goAnalysis(geneList, organism="hsapiens"):
    """
    Perform go Enrichment analysis (also KEGG and REAC).

    Parameters
    ----------
    geneList : list of str
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
    >>> autoprot.analysis.goAnalysis(['PEX14', 'PEX18']).iloc[:3,:3]
    source      native                                   name
    0  CORUM  CORUM:1984                 PEX14 homodimer complex
    1  GO:CC  GO:1990429          peroxisomal importomer complex
    2  GO:BP  GO:0036250  peroxisome transport along microtubule
    """
    if not isinstance(geneList, list):
        try:
            geneList = list(geneList)
        except:
            raise ValueError("Please provide a list of gene names")
    return gp.profile(organism=organism, query=geneList, no_evidences=False)


def makePSM(seq, seqLen):
    """
    Generate a position score matrix for a set of sequences.

    Returns the percentage of each amino acid for each position that
    can be further normalized using a PSM of unrelated/background sequences.

    Parameters
    ----------
    seq : list of str
        list of sequences.
    seqLen : int
        Length of the peptide sequences.
        Must match to the list provided.

    Returns
    -------
    pd.Dataframe
        Dataframe holding the prevalence for every amino acid per position in
        the input sequences.

    Examples
    --------
    >>> autoprot.analysis.makePSM(['PEPTIDE', 'PEGTIDE', 'GGGGGGG'], 7)
              0         1         2         3         4         5         6
    G  0.333333  0.333333  0.666667  0.333333  0.333333  0.333333  0.333333
    P  0.666667  0.000000  0.333333  0.000000  0.000000  0.000000  0.000000
    A  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
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
        'A': 0,
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

    seq = [i for i in seq if len(i) == seqLen]
    seqT = [''.join(s) for s in zip(*seq)]
    scoreMatrix = []
    for pos in seqT:
        d = aa_dic.copy()
        for aa in pos:
            aa = aa.upper()
            if aa == '.' or aa == '-' or aa == '_' or aa == "X":
                pass
            else:
                d[aa] += 1
        scoreMatrix.append(d)

    for pos in scoreMatrix:
        for k in pos.keys():
            pos[k] /= len(seq)

    # empty array -> (sequenceWindow, aa)
    m = np.empty((seqLen, 20))
    for i in range(m.shape[0]):
        x = [j for j in scoreMatrix[i].values()]
        m[i] = x

    m = pd.DataFrame(m, columns=aa_dic.keys())

    return m.T
