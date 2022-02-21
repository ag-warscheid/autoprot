# -*- coding: utf-8 -*-
"""
Autoprot Analysis Functions.

@author: Wignand

@documentation: Julian
"""
import os
from subprocess import run, PIPE
from importlib import resources
from pathlib import Path
from scipy.stats import ttest_1samp, ttest_ind, zscore
from scipy.spatial import distance
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import maxdists,fcluster
from statsmodels.stats import multitest as mt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score,calinski_harabasz_score, davies_bouldin_score
import matplotlib.pylab as plt
import pylab as pl
import colorsys
import seaborn as sns
from operator import itemgetter
import itertools
from Bio import Entrez
import time
from autoprot import visualization as vis
from autoprot import RHelper
import wordcloud as wc
import warnings
import missingno as msn
from gprofiler import GProfiler
gp = GProfiler(
    user_agent="autoprot",
    return_dataframe=True)
from autoprot import preprocessing as pp
#might want to enable warnings for debugging
#disabled them for copy vs view pandas warnings (pretty annoying)
warnings.filterwarnings('ignore')

RSCRIPT, R = RHelper.returnRPath()

#check where this is actually used and make it local
cmap = sns.diverging_palette(150, 275, s=80, l=55, n=9)


def ttest(df, reps, cond="", mean=True, adjustPVals=True):
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
    mean : bool, optional
        Whether to calculate the logFC of the provided data.
        For the one sample ttest log2 transformed ratios are expected.
        For the two sample ttest log transformed intensities are expected.
        If mean=True for two sample ttest, the log2 ratio of the provided
        replicates will be calculated.
        The default is True.
    adjustPVals : bool, optional
        Whether to adjust P-values. The default is True.

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
    >>> prot_tt = autoprot.analysis.ttest(df=protLog, reps=twitchVsmild, cond="TvM", mean=True, adjustPVals=True)
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
        prot_tt = ana.ttest(df=protLog, reps=twitchVsmild, cond="TvM", mean=True, adjustPVals=True)
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
    cond = '_' + cond
    if isinstance(reps[0], list) and len(reps) == 2:
        df[f"pValue{cond}"] = df[reps[0]+reps[1]].apply(lambda x: np.ma.filled(ttest_ind(x[:len(reps[0])], x[len(reps[0]):], nan_policy="omit")[1],np.nan),1).astype(float)
        df[f"score{cond}"] = -np.log10(df[f"pValue{cond}"])
        if mean == True:
            df[f"logFC{cond}"] = np.log2(pd.DataFrame(df[reps[0]].values / df[reps[1]].values).mean(1)).values

    else:
        df[f"pValue{cond}"] = df[reps].apply(lambda x: np.ma.filled(ttest_1samp(x, nan_policy="omit", popmean=0)[1],np.nan),1).astype(float)
        df[f"score{cond}"] = -np.log10(df[f"pValue{cond}"])
        if mean == True:
            df[f"logFC{cond}"] = df[reps].mean(1)

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
    std2= df[group2].std(1).values
    # TODO: the pooled sd here is calculated omitting the sample sizes n
    # This is not exactly what was proposed for cohens d: https://en.wikipedia.org/wiki/Effect_size
    sd_pooled =  np.sqrt((std1**2 + std2**2) / 2)
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
    
    Generate plots
    
    >>> autopca.scree()
    >>> autopca.corrComp(annot=False)
    >>> autopca.barLoad(1)
    >>> autopca.barLoad(2)
    >>> autopca.scorePlot(pc1=1, pc2=2)
    >>> autopca.loadingPlot(pc1=1, pc2=2, labeling=True)
    >>> autopca.biPlot(pc1=1, pc2=2)
    
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
        autopca.corrComp(annot=False)
        autopca.barLoad(1)
        autopca.barLoad(2)
        autopca.scorePlot(pc1=1, pc2=2)
        autopca.loadingPlot(pc1=1, pc2=2, labeling=True)
        autopca.biPlot(pc1=1, pc2=2)
        plt.show()
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
        forVis.columns = [f"PC{i}" for i in range(1, min(X.shape[0], X.T.shape[0])+1)]
        forVis["label"] = label
        return pca, forVis

    def scree(self, figsize=(15,5)):
        """
        Plot Scree plot and Explained variance vs number of components.

        Parameters
        ----------
        figsize : TYPE, optional
            DESCRIPTION. The default is (15,5).

        Raises
        ------
        TypeError
            No PCA object was initialised in the class.

        Returns
        -------
        None.

        """
        if not isinstance(self.pca,PCA):
            raise TypeError("This is a function to plot Scree plots. Provide fitted sklearn PCA object.")

        eigVal = self.pca.explained_variance_
        cumVar = np.append(np.array([0]), np.cumsum(self.expVar))

        plt.figure(figsize=figsize)
        plt.subplot(121)
        plt.plot(range(1,len(eigVal)+1), eigVal, marker="o", color="teal",
                markerfacecolor='purple')
        plt.ylabel("Eigenvalues")
        plt.xlabel("# Component")
        plt.title("Scree plot")
        sns.despine()

        plt.subplot(122)
        plt.plot(range(1, len(cumVar)+1), cumVar, ds="steps", color="teal")
        plt.xticks(range(1,len(eigVal)+1))
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
        yp = [i+0.5 for i in range(len(self.label))]
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
        PC =  f"PC{pc}"
        forVis = self.forVis.copy()
        forVis[f"{PC}_abs"] = abs(forVis[PC])
        forVis["color"] = "negative"
        forVis.loc[forVis[PC]>0, "color"] = "positive"
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
        PC =  f"PC{pc}"
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
        columns = [f"PC{i+1}" for i in range(self.Xt.shape[1])]
        scores = pd.DataFrame(self.Xt, columns=columns)
        if self.batch is not None:
            scores["batch"] = self.batch
        return scores

    def scorePlot(self, pc1=1, pc2=2, labeling=False, file=None, figsize=(5,5)):
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
        x = self.Xt[::,pc1-1]
        y = self.Xt[::,pc2-1]
        plt.figure(figsize=figsize)
        if self.batch is None:
            forVis = pd.DataFrame({"x":x,"y":y})
            sns.scatterplot(data=forVis, x="x", y="y")
        else:
            forVis = pd.DataFrame({"x":x,"y":y, "batch":self.batch})
            sns.scatterplot(data=forVis, x="x", y="y", hue=forVis["batch"])
        forVis["label"] = self.rlabel

        plt.title("Score plot")
        plt.xlabel(f"PC{pc1}\n{round(self.expVar[pc1-1]*100,2)} %")
        plt.ylabel(f"PC{pc2}\n{round(self.expVar[pc2-1]*100,2)} %")

        if labeling is True:
            ss = forVis["label"]
            xx = forVis["x"]
            yy = forVis["y"]
            for x,y,s in zip(xx,yy,ss):
                plt.text(x,y,s)
        sns.despine()

        if file is not None:
            plt.savefig(fr"{file}/ScorePlot.pdf")

    def loadingPlot(self, pc1=1, pc2=2, labeling=False, figsize=(5,5)):
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
        figsize : TYPE, optional
            DESCRIPTION. The default is (5,5).

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
            y=f"PC{pc2}",edgecolor=None)
        else:
            sns.scatterplot(data=self.forVis, x=f"PC{pc1}",
            y=f"PC{pc2}",edgecolor=None, hue=self.batch)
        sns.despine()

        plt.title("Loadings plot")
        plt.xlabel(f"PC{pc1}\n{round(self.expVar[pc1-1]*100,2)} %")
        plt.ylabel(f"PC{pc2}\n{round(self.expVar[pc2-1]*100,2)} %")

        if labeling is True:
            ss = self.forVis["label"]
            xx = self.forVis[f"PC{pc1}"]
            yy = self.forVis[f"PC{pc2}"]
            for x,y,s in zip(xx,yy,ss):
                plt.text(x,y,s)


    def biPlot(self, pc1=1, pc2=2, numLoad="all", figsize=(5,5), **kwargs):
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
        x = self.Xt[::,pc1-1]
        y = self.Xt[::,pc2-1]
        plt.figure(figsize=figsize)
        plt.scatter(x,y, color="lightgray", alpha=0.5, linewidth=0, **kwargs)

        temp = self.forVis[[f"PC{pc1}", f"PC{pc2}"]]
        temp["label"] = self.label
        temp = temp.sort_values(by=f"PC{pc1}")

        if numLoad == "all":
            loadings = temp[[f"PC{pc1}", f"PC{pc2}"]].values
            labels = temp["label"].values
        else:
            loadings = temp[[f"PC{pc1}", f"PC{pc2}"]].iloc[:numLoad].values
            labels = temp["label"].iloc[:numLoad].values

        xscale = 1.0 / (self.Xt[::,pc1-1].max() - self.Xt[::,pc1-1].min())
        yscale = 1.0 / (self.Xt[::,pc2-1].max() - self.Xt[::,pc2-1].min())
        xmina = 0
        xmaxa = 0
        ymina = 0

        for l, lab in zip(loadings, labels):
            #plt.plot([0,l[0]/xscale], (0, l[1]/yscale), color="purple")
            plt.arrow(x=0, y=0,dx=l[0]/xscale, dy= l[1]/yscale, color="purple",
                     head_width=.2)
            plt.text(x=l[0]/xscale, y=l[1]/yscale, s=lab)

            if l[0]/xscale < xmina:
                xmina = l[0]/xscale
            elif l[0]/xscale > xmaxa:
                xmaxa = l[0]/xscale

            if l[1]/yscale < ymina:
                ymina = l[1]/yscale
            elif l[1]/yscale > ymina:
                ymina = l[1]/yscale

        plt.xlabel(f"PC{pc1}\n{round(self.expVar[pc1-1]*100,2)} %")
        plt.ylabel(f"PC{pc2}\n{round(self.expVar[pc2-1]*100,2)} %")
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

        forVis = pd.DataFrame(self.Xt[:,:n])
        i = np.argmin(self.Xt.shape)
        pcs = self.Xt.shape[i]
        forVis.columns = [f"PC {i}" for i in range(1,pcs+1)]
        if self.batch is not None:
            forVis["batch"] = self.batch
            sns.pairplot(forVis, hue="batch")
        else:
            sns.pairplot(forVis)

class autoHCA:
    """
    Conduct hierarchical cluster analysis.
    
    Usesr provides dataframe and can afterwards
    use various metrics and methods to perfom and evaluate
    clustering.
    StandarWorkflow:
    makeLnkage() -> evalClustering() -> clusterMap() -> writeClusterFiles()
    
    Examples
    --------
    autoProt provides a class which allows the easy implementation and
    evaluation of a hierarchical cluster analysis.
    You need provide the data.
    You may also want to provide appropriate row and column labels.
    Depending on what the aim of your cluster analysis is you might want to
    perform a zscore transformation.
    
    
    """

    def __init__(self, data, clabels=None, rlabels=None, zscore=None, linkage=None):
        """
        Initialise the class.
        
        Parameters
        ----------
        data : np.array
            The data to be clustered.
        clabels : np.array, optional
            Column labels. The default is None.
        rlabels : np.array, optional
            Row labels. The default is None.
        zscore : int or None, optional
            Axis along which to calculate the zscore.
            The default is None.
        linkage : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        self.data = self._checkData(data,clabels, rlabels, zscore)
        self.rlabels = rlabels
        if clabels is not None:
            self.clabels = clabels
        else:
            self.clabels = range(1, self.data.shape[1]+1)
        self.linkage = linkage
        self.cluster = None
        self.clusterCol = None
        self.cmap = sns.diverging_palette(150, 275, s=80, l=55, n=9)

    @staticmethod
    def _checkData(data, clabels, rlabels, zs):
        """
        Check if data contains missing values and remove them.

        Parameters
        ----------
        data : np.array
            The data to be clustered.
        clabels : np.array, optional
            Column labels.
        rlabels : np.array, optional
            Row labels.
        zs : int or None
            Axis to calculate zscore on. If None, not zscore is calculated.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        data : pd.DataFrame
            Data transferred into Dataframe with colnames and rownames.

        """
        # drop all rows containing missing values
        if rlabels is None and clabels is None:
            data = pd.DataFrame(data).dropna()

        # drop rows with values and assign colnames
        elif rlabels is None:
            if data.shape[1] != len(clabels):
                    raise ValueError("Data and labels must be same size!")
            data = pd.DataFrame(data).dropna()
            data.columns = clabels

        # find columns with missing values and drop these
        elif clabels is None:
            if data.shape[0] != len(rlabels):
                raise ValueError("Data and labels must be same size!")
            temp = pd.DataFrame(data)
            # column indices without missing data
            to_keep = temp[temp.notnull().all(1)].index
            temp = temp.loc[to_keep]
            # temp["labels"] = rlabels
            #rlabels = temp["labels"]
            # data = temp.drop("labels", axis=1)
            data = temp
            data.index = rlabels

        # clabels and rlabels are provided
        # check if row and col dimensions match and drop columns with
        # missing values
        else:
            if data.shape[0] != len(rlabels) or data.shape[1] != len(clabels):
                raise ValueError("Data and labels must be same size!")
            temp = pd.DataFrame(data)
            # column indices without missing data
            to_keep = temp[temp.notnull().all(1)].index
            # temp["labels"] = rlabels
            temp = temp.loc[to_keep]
            # rlabels = temp["labels"]
            # data = temp.drop("labels", axis=1)
            data = temp
            data.index = rlabels
            data.columns = clabels

        # if the zscore is calculate (i.e. if zs != None)
        # a dataframe with zscores instead of values is calculated
        if zs is not None:
            temp = data.copy(deep=True).values
            tc = data.columns
            ti = data.index
            X = zscore(temp, axis=zs)
            data = pd.DataFrame(X)
            data.columns = tc
            data.index = ti

        return data

    def _get_N_HexCol(self, n):
        """Return RGB values for colouring a number n of values."""
        HSV_tuples = [(x*1/n, 0.75, 0.6) for x in range(n)]
        RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
        return list(RGB_tuples)


    def _makeCluster(self, n, colors):
        """
        Form flat clusters from the hierarchical clustering of linkage.

        Parameters
        ----------
        n : int
            Max number of clusters.
        colors : None or list of RGB_tuples
            Colors for the clusters.
            If none, new colors are generated.

        Returns
        -------
        None.

        """
        # self.cluster is an ndarray of length x
        # with x = number of original data points
        self.cluster = fcluster(self.linkage, # the hierarchical clustering
                                t=n, # max number of clusters
                                criterion="maxclust") # forms maximumum n=t clusters
        # there should be as many colours as clusters
        if colors is None or len(colors) != n:
            colors = self._get_N_HexCol(n)
        # produce n * x colours (with n = number of clusters and
        # x = number of datapoints)
        # TODO Check hat happens here: self.ClusterCol is overwritten n-1 times
        # from seaborn:
        # List of colors to label for either the rows or columns.
        # Useful to evaluate whether samples within a group are clustered together.
        # Can use nested lists or DataFrame for multiple color levels of labeling.
        # If given as a pandas.DataFrame or pandas.Series,
        # labels for the colors are extracted from the DataFrames column names
        # or from the name of the Series.
        # DataFrame/Series colors are also matched to the data by their index,
        # ensuring colors are drawn in the correct order.
        for i in range(n):
            self.clusterCol = [colors[i] if j == i+1 else j for j in self.cluster]


    def _makeClusterTraces(self,n,file,colors, z_score=None):
        """
        Plot RMSD vs colname line plots.

        Shaded areas representing groups of RMSDs are plotted.

        Parameters
        ----------
        n : int
            Number of Clusters to visualise.
        file : str
            Filename with extension to save file to.
            Will be extended by FNAME_traces.EXT.
        colors : list of str or None.
            Colours for the traces. If none, the same predefined colours will
            be used for all n traces.
        z_score : int or None, optional
            Axis along which to standardise the data by z-score transformation.
            The default is None.

        Returns
        -------
        None.

        """
        plt.figure(figsize=(5,3*n))
        temp = pd.DataFrame(self.data.copy(deep=True))
        if z_score is not None:
            # calculate the z-score using scipy using the other axis (i.e. axis=0 if
            # 1 was provided and vice versa)
            temp = pd.DataFrame(zscore(temp, axis=1-z_score)) #seaborn and scipy work opposite
        # ndarray containing the cluster numbers for each data point
        temp["cluster"] = self.cluster

        # iterate over the generated clusters
        for i in range(n):

            ax = plt.subplot(n, 1, i+1)
            # slice data points belonging to a certain cluster number
            temp2 = temp[temp["cluster"]==i+1].drop("cluster", axis=1)

            # compute the root mean square deviation of the z-scores or the protein log fold changes
            # as a helper we take the -log of the rmsd in order to plot in the proper sequence
            # i.e. low RMSDs take on large -log values and thereofore are plotted
            # last and on the top
            temp2["distance"] = temp2.apply(lambda x: -np.log(np.sqrt(sum((x-temp2.mean())**2))),1)

            if temp2.shape[0] == 1:
                # if cluster contains only 1 entry i.e. one condition
                ax.set_title("Cluster {}".format(i+1))
                ax.set_ylabel("")
                ax.set_xlabel("")
                # ax.plot(range(temp2.shape[1]-1),temp2.drop("distance", axis=1).values.reshape(3), color=color[idx], alpha=alpha[idx])
                ax.plot(range(temp2.shape[1]-1),
                        temp2.drop("distance", axis=1).values.reshape(3))
                plt.xticks(range(len(self.clabels)), self.clabels)
                continue

            # bin the RMSDs into five groups
            temp2["distance"] = pd.cut(temp2["distance"],5)

            #get aestethics for traces
            if colors is None:
                color=["#C72119","#D67155","#FFC288", "#FFE59E","#FFFDBF"]
            else:
                color = [colors[i]]*5
            color=color[::-1]
            alpha=[0.1, 0.2, 0.25, 0.4, 0.6]

            # group into the five RMSD bins
            grouped = temp2.groupby("distance")

            ax.set_title("Cluster {}".format(i+1))
            if z_score == None:
                ax.set_ylabel("-ln RMSD(value)")
            else:
                ax.set_ylabel("-ln RMSD(z-score)")
            ax.set_xlabel("Condition")
            
            # for every RMSD group
            for idx, (i, group) in enumerate(grouped):
                # for every condition (i.e. colname)
                for j in range(group.shape[0]):
                    ax.plot(range(temp2.shape[1]-1),
                            group.drop("distance", axis=1).iloc[j],
                            color=color[idx],
                            alpha=alpha[idx])
            # set the tick labels as the colnames
            plt.xticks(range(len(self.clabels)), self.clabels)

            # save to file if asked
            if file is not None:
                name, ext = file.split('.')
                filet = f"{name}_traces.{ext}"
                plt.savefig(filet)


    def _makeSummary(self, n, file):
        """
        n: number of cluster
        ToDo: Maybe enable switch between heatmap and trace?
        """
        temp = self.data.copy(deep=True)
        temp["cluster"] = self.cluster
        grouped = temp.groupby("cluster")[self.data.columns].mean()
        lim = abs(grouped).max().max()

        ylabel = [f"Cluster{i+1}\n{j} member" for i,j in enumerate(temp.groupby("cluster").count().iloc[:,0].values)]
        plt.figure()
        plt.title("Summary Of Clustering")
        sns.heatmap(grouped,
                    cmap=self.cmap, vmin=lim, vmax=-lim)
        plt.yticks([i+0.5 for i in range(len(ylabel))], ylabel, rotation=0)
        if file is not None:
            name, ext = file.split('.')
            filet = f"{name}_summary.{ext}"
            plt.tight_layout()
            plt.savefig(filet)


    def asDist(self, c):
        """
        Convert a matrix (i.e. correlation matrix) into a distance matrix for hierachical clustering.

        Parameters
        ----------
        c : np.ndarray
            Input matrix.

        Returns
        -------
        list
            List corresponsding to left off-diagonal elememnts of the
            correlation matrix.
            
        Example
        -------
        >>> a = [
        ...     [0.1, .32, .2,  0.4, 0.8], 
        ...     [.23, .18, .56, .61, .12], 
        ...     [.9,   .3,  .6,  .5,  .3], 
        ...     [.34, .75, .91, .19, .21]
        ...      ]
        >>> c = np.corrcoef(a)
        >>> c
        array([[ 1.        , -0.35153114, -0.74736506, -0.48917666],
               [-0.35153114,  1.        ,  0.23810227,  0.15958285],
               [-0.74736506,  0.23810227,  1.        , -0.03960706],
               [-0.48917666,  0.15958285, -0.03960706,  1.        ]])
        >>> autoprot.autoHCA.asDist(c)
        [-0.3515311393849671,
         -0.7473650573493561,
         -0.4891766567441463,
         0.23810227412143423,
         0.15958285448266604,
         -0.03960705975653923]
        """
        return [c[i][j] for i in (range(c.shape[0])) for j in (range(c.shape[1])) if i<j]

    def makeLinkage(self, method, metric):
        """
        Perform hierarchical clustering on the data.

        Parameters
        ----------
        method : str
            Which method is used for the clustering.
            Possible are 'single', 'average' and 'complete' and all values
            for method of scipy.cluster.hierarchy.linkage
            See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
        metric : str
            Which metric is used to calculate distance.
            Possible values are 'pearson', 'spearman' and all metrics
            implemented in scipy.spatial.distance.pdist
            See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html

        Returns
        -------
        None.

        """
        # First calculate a distance metric between the points
        if metric in ["pearson", "spearman"]:
            corr = pd.DataFrame(self.data).T.corr(metric).values
            dist = self.asDist(1 - corr)
        else:
            dist = distance.pdist(self.data, metric=metric)
        # perform hierarchical clustering using the distance metric
        # the returned matrix self-linkage contains n-1 x 4 elements
        # with each row representing
        # cluster1, cluster2, distance_between_1_and_2,
        # number_of_observations_in_the_cluster
        self.linkage = hierarchy.linkage(dist, method=method)


    def evalClustering(self,start=2, upTo=20, figsize=(15,5)):
        """
        Evaluate number of clusters.

        Parameters
        ----------
        start : int, optional
            The minimum number of clusters to plot. The default is 2.
        upTo : int, optional
            The maximum nbumber of clusters to plot. The default is 20.
        figsize : tuple of float or int, optional
            The size of the plotted figure.
            The default is (15,5).

        Notes
        -----
        Davies-Bouldin score:
            The score is defined as the average similarity measure of each 
            cluster with its most similar cluster, where similarity is the
            ratio of within-cluster distances to between-cluster distances.
            Thus, clusters which are farther apart and less dispersed will
            result in a better score.
            The minimum score is zero, with lower values indicating better
            clustering.
        Silhouette score:
            The Silhouette Coefficient is calculated using the mean
            intra-cluster distance (a) and the mean nearest-cluster
            distance (b) for each sample. The Silhouette Coefficient for a
            sample is (b - a) / max(a, b). To clarify, b is the distance
            between a sample and the nearest cluster that the sample is not a
            part of. Note that Silhouette Coefficient is only defined if
            number of labels is 2 <= n_labels <= n_samples - 1.
            The best value is 1 and the worst value is -1. Values near 0
            indicate overlapping clusters. Negative values generally indicate
            that a sample has been assigned to the wrong cluster, as a
            different cluster is more similar.
        Harabasz score:
            It is also known as the Variance Ratio Criterion.
            The score is defined as ratio between the within-cluster dispersion
            and the between-cluster dispersion.

        Returns
        -------
        None.

        """
        upTo += 1
        pred = []
        for i in range(start,upTo):
            # return the assigned cluster labels for each data point
            cluster = fcluster(self.linkage,t=i, criterion='maxclust')
            # calculate scores based on assigned cluster labels and
            # the original data points
            pred.append((davies_bouldin_score(self.data, cluster),
                       silhouette_score(self.data, cluster),
                       calinski_harabasz_score(self.data, cluster)))

        pred = np.array(pred)
        plt.figure(figsize=figsize)
        plt.subplot(131)
        plt.title("Davies_boulding_score")
        plt.plot(pred[::,0])
        plt.xticks(range(upTo-start),range(start,upTo), rotation=90);
        plt.grid(axis='x')
        plt.subplot(132)
        plt.title("Silhouoette_score")
        plt.plot(pred[::,1])
        plt.xticks(range(upTo-start),range(start,upTo), rotation=90);
        plt.grid(axis='x')
        plt.subplot(133)
        plt.title("Harabasz score")
        plt.plot(pred[::,2])
        plt.xticks(range(upTo-start),range(start,upTo), rotation=90);
        plt.grid(axis='x')
        print(f"Best Davies Boulding at {start + list(pred[::,0]).index(min(pred[::,0]))} with {min(pred[::,0])}")
        print(f"Best Silhouoette_score at {start + list(pred[::,1]).index(max(pred[::,1]))} with {max(pred[::,1])}")
        print(f"Best Harabasz/Calinski at {start + list(pred[::,2]).index(max(pred[::,2]))} with {max(pred[::,2])}")


    def clusterMap(self, nCluster=None, colCluster=False, makeTraces=False,
                   summary=False, file=None, rowColors=None,
                   colors=None,yticklabels="", **kwargs):
        """
        Visualise the clustering.

        Parameters
        ----------
        nCluster : int, optional
            How many clusters to annotate. The default is None.
        colCluster : bool, optional
            Whether to cluster the columns. The default is False.
        makeTraces : bool, optional
            Whether to generate traces of each cluster. The default is False.
        summary : bool, optional
            Whether to generate a summery.
            NOT IMPLEMENTED.
            The default is False.
        file : str, optional
            Path to the output plot file. The default is None.
        rowColors : dict, optional
            dictionary of title and colors for row coloring.
            Generates an additional column in the heatmeap showing
            the indicated columns values as colors.
            Has to be same length as provided data.
            The default is None.
        colors : list of str, optional
            Colors for the annotated clusters.
            Has to be the same size as nCluster.
            The default is None.
        yticklabels : TYPE, optional
            DESCRIPTION. The default is "".
        **kwargs :
            passed to seaborn.clustermap.
            See https://seaborn.pydata.org/generated/seaborn.clustermap.html
            May also contain 'z-score' that is used during making of
            cluster traces.

        Returns
        -------
        None.

        """
        # summaryMap: report heatmap with means of each cluster
        # -> also provide summary for each cluster like number of entries?
        #savemode just preliminary have to be overworked here

        # generates self.cluster with cluster labels and
        # self.clusterCol with some matrix of colours
        if nCluster is not None:
            self._makeCluster(nCluster, colors)

        if "cmap" not in kwargs.keys():
            kwargs["cmap"] = self.cmap

        if rowColors is not None:
            # add the cluster color information
            rowColors["cluster"] = self.clusterCol
            # convert dict to dataframe to use as multi-index labeling col in sns
            temp = pd.DataFrame(rowColors)
            # 
            cols = ["cluster"] + temp.drop("cluster", axis=1).columns.to_list()
            temp = temp[cols]
            temp.index = self.data.index
        else:
            temp = self.clusterCol

        sns.clustermap(self.data, row_linkage=self.linkage,
                   row_colors=temp, col_cluster=colCluster,
                   yticklabels=yticklabels, **kwargs)

        if file is not None:
            plt.savefig(file)

        if makeTraces == True:
            if "z_score" in kwargs.keys():
                self._makeClusterTraces(nCluster, file, z_score=kwargs["z_score"], colors=colors)
            else:
                self._makeClusterTraces(nCluster, file, colors=colors)

        if summary == True:
            self._makeSummary(nCluster, file)



    def returnCluster(self):
        """
        return df with clustered data
        """
        temp = self.data
        temp["cluster"] = self.cluster
        return temp


    def writeClusterFiles(self, rootdir):
        """
        generates folder with text files for each
        cluster at provided rootdir
        """
        path = os.path.join(rootdir, "clusterResults")
        if "clusterResults" in os.listdir(rootdir):
            pass
        else:
            os.mkdir(path)

        temp = self.data
        temp["cluster"] = self.cluster
        for cluster in temp["cluster"].unique():
            pd.DataFrame(temp[temp["cluster"]==cluster].index).to_csv(f"{path}/cluster_{cluster}.tsv", header=None, index=False)


class KSEA:
    """
    Perform kinase substrate enrichment analysis.

    You have to provide phosphoproteomic data.
    This data has to contain information about Gene name, position and amino acid of the peptides with
    "Gene names", "Position" and "Amino acid" as the respective column names.
    Optionally you can provide a "Multiplicity" column.

    Methods
    -------
    addSubstrate(kinase, substrate, subModRsd)
        Function that allows user to manually add substrates.
    removeManualSubs
    annotate
    getKinaseOverview
    ksea
    returnEnrichment
    plotEnrichment
    volcanos
    returnKinaseSubstrate
    annotateDf
    """

    def __init__(self, data):
        with resources.open_text("autoprot.data","Kinase_Substrate_Dataset") as d:
            self.PSP_KS = pd.read_csv(d, sep='\t')
        self.PSP_KS["SUB_GENE"] = self.PSP_KS["SUB_GENE"].fillna("NA").apply(lambda x: x.upper())
        self.PSP_KS["source"] = "PSP"
        with resources.open_text("autoprot.data","Regulatory_sites") as d:
            self.PSP_regSits = pd.read_csv(d, sep='\t')
        self.data = self.__preprocess(data.copy(deep=True))
        self.annotDf = None
        self.kseaResults = None
        self.koi = None
        self.simpleDf = None


    @staticmethod
    def __preprocess(data):
        data["MOD_RSD"] = data["Amino acid"] + data["Position"].fillna(0).astype(int).astype(str)
        data["ucGene"] = data["Gene names"].fillna("NA").apply(lambda x: x.upper())
        data["mergeID"] = range(data.shape[0])
        return data


    def __enrichment(self,df, col, kinase):
        KS = df[col][df["KINASE"].fillna('').apply(lambda x: kinase in x)]
        s = KS.mean()#mean FC of kinase subs
        p = df[col].mean()#mean FC of all
        m = KS.shape[0]#no of kinase subs
        sig = df[col].std()#standard dev of FC of all
        score = ((s-p)*np.sqrt(m))/sig
        return [kinase,score]


    def __extractKois(self,df):
        koi = [i for i in list(df["KINASE"].values.flatten()) if isinstance(i,str) ]
        ks = set(koi)
        temp = []
        for k in ks:
            temp.append((k,koi.count(k)))
        return pd.DataFrame(temp, columns=["Kinase", "#Subs"])


    def addSubstrate(self, kinase, substrate, subModRsd):
        """
        function that allows user to manually add substrates

        """
        it = iter([kinase, substrate, subModRsd])
        the_len = len(next(it))
        if not all(len(l) == the_len for l in it):
             raise ValueError('not all lists have same length!')

        temp = pd.DataFrame(columns=self.PSP_KS.columns)
        for i in range(len(kinase)):
            temp.loc[i, "KINASE"] = kinase[i]
            temp.loc[i, "SUB_GENE"] = substrate[i]
            temp.loc[i, "SUB_MOD_RSD"] = subModRsd[i]
            temp.loc[i, "source"] = "manual"
        self.PSP_KS = self.PSP_KS.append(temp, ignore_index=True)


    def removeManualSubs(self):
        self.PSP_KS = self.PSP_KS[self.PSP_KS["source"]=="PSP"]


    def annotate(self,
        organism="human",
        onlyInVivo=False):
        """
        do we need crossspecies annotation?
        """


        if onlyInVivo==True:
            temp = self.PSP_KS[((self.PSP_KS["KIN_ORGANISM"] == organism) &
            (self.PSP_KS["SUB_ORGANISM"] == organism) &
            (self.PSP_KS["IN_VIVO_RXN"]=="X")) | (self.PSP_KS["source"]=="manual")]
        else:
            temp = self.PSP_KS[((self.PSP_KS["KIN_ORGANISM"] == organism) &
            (self.PSP_KS["SUB_ORGANISM"] == organism)) | (self.PSP_KS["source"]=="manual")]

        if "Multiplicity" in self.data.columns:
            self.annotDf = pd.merge(self.data[["ucGene", "MOD_RSD", "Multiplicity","mergeID"]],
            temp,
            left_on=["ucGene", "MOD_RSD"],
            right_on=["SUB_GENE", "SUB_MOD_RSD"],
            how="left")
        else:
            self.annotDf = pd.merge(self.data[["ucGene", "MOD_RSD", "mergeID"]],
            temp,
            left_on=["ucGene", "MOD_RSD"],
            right_on=["SUB_GENE", "SUB_MOD_RSD"],
            how="left")


        self.koi = self.__extractKois(self.annotDf)


    def getKinaseOverview(self, kois=None):

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))

        sns.histplot(self.koi["#Subs"], bins=50, ax= ax[0])
        sns.despine(ax=ax[0])
        ax[0].set_title("Overview of #Subs per kinase")

        #get axis[1] ready - basically remove everthing
        ax[1].spines["left"].set_visible(False)
        ax[1].spines["top"].set_visible(False)
        ax[1].spines["bottom"].set_visible(False)
        ax[1].spines["right"].set_visible(False)
        ax[1].tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False,
        labelbottom=False,
        labelleft=False) # labels along the bottom edge are off
        ax[1].set_xlim(0,1)
        ax[1].set_ylim(0,1)

        #plot table
        ax[1].text(x=0, y=1-0.01, s="Top10\nKinase")
        ax[1].text(x=0.1, y=1-0.01, s="#Subs")

        ax[1].plot([0,0.2], [.975,.975], color="black")
        ax[1].plot([0.1,0.1], [0,.975], color="black")

        #get top 10 kinases for annotation
        text = self.koi.sort_values(by="#Subs", ascending=False).iloc[:10].values
        for j,i in enumerate(text):
            j+=1
            ax[1].text(x=0, y=1-j/10, s=i[0])
            ax[1].text(x=0.125, y=1-j/10, s=i[1])
        #plot some descriptive stats
        tot = self.koi.shape[0]
        s = f"Substrates for {tot} kinases found in data."
        ax[1].text(0.3, 0.975, s)
        med = round(self.koi['#Subs'].median(), 2)
        s = f"Median #Sub: {med}"
        ax[1].text(0.3,0.925, s)
        mea = round(self.koi['#Subs'].mean(), 2)
        s = f"Mean #Sub: {mea}"
        ax[1].text(0.3,0.875, s)
        #if kois are provided plot those
        if kois is not None:
            pos = .8
            for j,k in enumerate(kois):
                try:
                    s = self.koi[self.koi["Kinase"].apply(lambda x: x.upper()) == k.upper()]["#Subs"].values[0]
                except:
                    s = 0
                ss = f"{k} has {s} substrates."
                ax[1].text(0.3,pos, ss)
                pos-=0.055


    def ksea(self, col, minSubs = 5, simplify=None):
        """
        function in which enrichment score is calculated
        """
        copyAnnotDf = self.annotDf.copy(deep=True)
        if simplify is not None:
            if simplify == "auto":
                simplify = {"AKT" : ["Akt1", "Akt2", "Akt3"],
                           "PKC" :  ["PKCA", "PKCD", "PKCE"],
                           "ERK" :  ["ERK1", "ERK2"],
                           "GSK3" : ["GSK3B", "GSK3A"],
                           "JNK" :  ["JNK1", "JNK2", "JNK3"],
                           "FAK" :  ["FAK iso2"],
                           "p70S6K":["p70S6K", "p70SKB"],
                           "RSK" : ["p90RSK", "RSK2"],
                           "P38" : ["P38A","P38B","P38C","P338D"]}
            for key in simplify:
                copyAnnotDf["KINASE"].replace(simplify[key], [key]*len(simplify[key]), inplace=True)

            #drop rows which are now duplicates
            if "Multiplicity" in copyAnnotDf.columns:
                idx = copyAnnotDf[["ucGene","MOD_RSD","Multiplicity", "KINASE"]].drop_duplicates().index
            else:
                idx = copyAnnotDf[["ucGene","MOD_RSD", "KINASE"]].drop_duplicates().indx
            copyAnnotDf = copyAnnotDf.loc[idx]
            self.simpleDf = copyAnnotDf

            self.koi = self.__extractKois(self.simpleDf)

        koi = self.koi[self.koi["#Subs"]>=minSubs]["Kinase"]

        self.kseaResults = pd.DataFrame(columns = ["kinase", "score"])
        copyAnnotDf = copyAnnotDf.merge(self.data[[col,"mergeID"]], on="mergeID", how="left")
        for kinase in koi:
            k, s = self.__enrichment(copyAnnotDf[copyAnnotDf[col].notnull()], col, kinase)
            temp = pd.DataFrame(data={"kinase":k, "score":s}, index=[0])
            self.kseaResults = self.kseaResults.append(temp, ignore_index=True)
        self.kseaResults = self.kseaResults.sort_values(by="score", ascending=False)


    def returnEnrichment(self):
        if self.kseaResults is None:
            print("First perform the enrichment")
        else:
        #dropna in case of multiple columns in data
        #sometimes there are otherwise nan
        #nans are dropped in ksea enrichment
            return self.kseaResults.dropna()


    def plotEnrichment(self, up_col="orange", down_col="blue",
    bg_col = "lightgray", plotBg = True, ret=False, title="",
    figsize=(5,10)):
        """
        function that can be used to plot the KSEA results
        """
        if self.kseaResults is None:
            print("First perform the enrichment")
        else:
            self.kseaResults["color"] = bg_col
            self.kseaResults.loc[self.kseaResults["score"]>2, "color"] = up_col
            self.kseaResults.loc[self.kseaResults["score"]<-2, "color"] = down_col
            fig = plt.figure(figsize=figsize)
            plt.yticks(fontsize=10)
            plt.title(title)
            if plotBg == True:
                sns.barplot(data= self.kseaResults.dropna(), x="score",y="kinase",
                palette=self.kseaResults.dropna()["color"])
            else:
                sns.barplot(data= self.kseaResults[self.kseaResults["color"]!=bg_col].dropna(), x="score",y="kinase",
                palette=self.kseaResults[self.kseaResults["color"]!=bg_col].dropna()["color"])
            sns.despine()
            plt.legend([],[], frameon=False)
            plt.axvline(0,0,1, ls="dashed", color="lightgray")
            if ret == True:
                plt.tight_layout()
                return fig


    def volcanos(self, logFC, p, kinases=[], **kwargs):
        """
        function that can be used to plot volcanos highlighting substrates
        of given kinase
        """
        df = self.annotateDf(kinases=kinases)
        for k in kinases:
            idx = df[df[k]==1].index
            vis.volcano(df, logFC, p=p, highlight=idx,
            custom_hl={"label":k},
            custom_fg={"alpha":.5}, **kwargs)


    def returnKinaseSubstrate(self, kinase):
        """
        returns new dataframe with respective substrates of kinase
        """
        if self.simpleDf is not None:
            df = self.simpleDf.copy(deep=True)
        else:
            df = self.annotDf.copy(deep=True)

        if isinstance(kinase, list):
            idx = []
            for k in kinase:
                idx.append(df[df["KINASE"].fillna("NA").apply(lambda x: x.upper())==k.upper()].index)
            dfFilter = df.loc[pl.flatten(idx)]
        elif isinstance(kinase, str):
            dfFilter = df[df["KINASE"].fillna("NA").apply(lambda x: x.upper())==kinase.upper()]
        else:
            raise ValueError("Please provide either a string or a list of strings representing kinases of interest.")

        dfFilter = pd.merge(dfFilter[['GENE', 'KINASE','KIN_ACC_ID','SUBSTRATE','SUB_ACC_ID',
                                       'SUB_GENE','SUB_MOD_RSD', 'SITE_GRP_ID','SITE_+/-7_AA', 'DOMAIN', 'IN_VIVO_RXN', 'IN_VITRO_RXN', 'CST_CAT#',
                                       'source', "mergeID"]],
                            self.PSP_regSits[['SITE_GRP_ID','ON_FUNCTION', 'ON_PROCESS', 'ON_PROT_INTERACT',
                                              'ON_OTHER_INTERACT', 'PMIDs', 'LT_LIT', 'MS_LIT', 'MS_CST',
                                              'NOTES']],
                                              how="left")
        return dfFilter


    def annotateDf(self, kinases=[]):
        """
        adds column to provided dataframe with given kinases and boolean value
        indicating whether or not peptide is kinase substrate
        """
        if len(kinases) > 0:
            df = self.data.drop(["MOD_RSD", "ucGene"], axis=1)
            for kinase in kinases:
                ids = self.returnKinaseSubstrate(kinase)["mergeID"]
                df[kinase] = 0
                df.loc[df["mergeID"].isin(ids),kinase] = 1
            return df.drop("mergeID", axis=1)
        else:
            print("Please provide Kinase for annotation.")


def missAnalysis(df,cols,n=999, sort='ascending',text=True, vis=True, extraVis=False,
                 saveDir=None):

    """
    function takes dataframe and
    prints missing stats
    :n:: int,  how much of the dataframe shall be displayed
    :sort:: (ascending, descending) which way to sort the df
    :text:: boolean, whether to output text summaryMap
    :vis:: boolean, whether to return barplot showing missingness
    :extraVis:: boolean, whether to return matrix plot showing missingness
    """

    def create_data(df):
        """
        takes df and output
        data for missing entrys
        """
        data = []
        for i in df:
            #len df
            n = df.shape[0]
            #how many are missing
            m = df[i].isnull().sum()
            #percentage
            p = m/n*100
            data.append([i,n,m,p])
        data = add_ranking(data)
        return data


    def add_ranking(data):
        """
        adds ranking
        """
        data = sorted(data,key=itemgetter(3))
        for idx, col in enumerate(data):
            col.append(idx)

        return data


    def describe(data, n,saveDir, sort='ascending'):
        """
        prints data
        :n how many entries are displayed
        :sort - determines which way data is sorted [ascending; descending]
        """
        if n == 999:
            n = len(data)
        elif n>len(data):
            print("'n' is larger than dataframe!\nDisplaying complete dataframe.")
            n = len(data)
        if n<0:
            raise ValueError("'n' has to be a positive integer!")

        if sort == 'ascending':
            pass
        elif sort == 'descending':
            data = data[::-1]

        for i in range(n):
            print("{} has {} of {} entries missing ({}%).".format(data[i][0], data[i][2],
                                                                               data[i][1], round(data[i][3],2)))
            print('-'*80)

        if saveDir:
            with open(saveDir + "/missAnalysis_text.txt", 'w') as f:
                for i in range(n):
                    f.write("{} has {} of {} entries missing ({}%).".format(data[i][0], data[i][2],
                                                                                       data[i][1], round(data[i][3],2)))
                    f.write('-'*80)

        return True


    def visualize(data, n, saveDir, sort='ascending'):
        """
        visualizes the n entries of dataframe
        :sort - determines which way data is sorted [ascending; descending]
        """
        if n == 999:
            n = len(data)
        elif n>len(data):
            print("'n' is larger than dataframe!\nDisplaying complete dataframe.")
            n = len(data)
        if n<0:
            raise ValueError("'n' has to be a positive integer!")

        if sort == 'ascending':
            pass
        elif sort == 'descending':
            data=data[::-1]

        data = pd.DataFrame(data=data, columns=["Name", "tot_values","tot_miss", "perc_miss", "rank"])
        plt.figure(figsize=(7,7))
        ax = plt.subplot()
        splot = sns.barplot(x=data["tot_miss"].iloc[:n], y=data["Name"].iloc[:n])
        for idx,p in enumerate(splot.patches):
            s=str(round(data.iloc[idx,3],2))+'%'
            x=p.get_width()+p.get_width()*.01
            y=p.get_y()+p.get_height()/2
            splot.annotate(s, (x,y))
        plt.title("Missing values of dataframe columns.")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel("")
        if saveDir:
            plt.savefig(saveDir + "/missAnalysis_vis1.pdf")

        return True


    def visualize_extra(df, saveDir):
        import missingno as msn
        plt.figure(figsize=(10,7))
        msn.matrix(df, sort="ascending")
        if saveDir:
            plt.savefig(saveDir + "/missAnalysis_vis2.pdf")
        return True


    df = df[cols]
    data = create_data(df)
    if text==True:
        describe(data,n, saveDir, sort)
    if vis == True:
        visualize(data,n, saveDir, sort)
    if extraVis == True:
        visualize_extra(df, saveDir)


def getPubAbstracts(text=[""], ToA=[""], author=[""], phrase=[""],
                  exclusion=[""], customSearchTerm=None, makeWordCloud=False,
                  sort='pub+date', retmax=20, output="print"):
    """
    Get Pubmed abstracts
    Provide the search terms and pubmed will be searched for paper matching those
    The exclusion list can be used to exclude certain words to be contained in a paper
    If you want to perform more complicated searches use the possibility of using
    a custom term.

    If you search for a not excact term you can use *
    For example opto* will also return optogenetic.

    @params:
    ::ToA: List of words which must occur in Titel or Abstract
    ::text: List of words which must occur in text
    ::author: List of authors which must occor in Author list
    ::phrase: List of phrases which should occur in text. Seperate word in phrases with hyphen (e.g. akt-signaling)
    ::exclusion: List of words which must not occur in text
    ::CustomSearchTerm: Enter a custom search term (make use of advanced pubmed search functionality)
    ::makeWordCloud: Boolen, toggles whether to draw a wordcloud of the words in retrieved abstrace
    """



    def search(query,retmax, sort):
        Entrez.email = 'your.email@example.com'
        handle = Entrez.esearch(db='pubmed',
                                sort=sort,
                                retmax=str(retmax),
                                retmode='xml',
                                term=query)
        results = Entrez.read(handle)
        return results


    def fetchDetails(id_list):
        ids = ','.join(id_list)
        Entrez.email = 'your.email@example.com'
        handle = Entrez.efetch(db='pubmed',
                               retmode='xml',
                               id=ids)
        results = Entrez.read(handle)
        return results


    def makeSearchTerm(text, ToA, author, phrase, exclusion):

        term = [i+"[Title/Abstract]" if i != "" else "" for i in ToA] +\
                [i+"[Text Word]" if i != "" else "" for i in text] +\
                [i+"[Author]" if i != "" else "" for i in author] + \
                [i if i != "" else "" for i in phrase]
        term = (" AND ").join([i for i in term if i != ""])

        if exclusion != [""]:
            exclusion = (" NOT ").join(exclusion)
            term = (" NOT ").join([term] + [exclusion])
            return term
        else:
            return term


    def makewc(final, output):
        abstracts = []
        for i in final:
            for abstract in final[i][2]:
                if not isinstance(abstract, str):
                    abstracts.append(abstract["AbstractText"][0])
        abstracts = (" ").join(abstracts)

        #when exlusion list is added in wordcloud add those
        el = ["sup"]

        fig = vis.wordcloud(text=abstracts, exlusionwords=el)
        if output != "print":
            if output[:-4] == ".txt":
                output = output[:-4]
            fig.to_file(output + "/WordCloud.png")


    def genOutput(final, output, makeWordCloud):

        if output == "print":
            for i in final:
                for title,author, abstract, doi, link, pid, date, journal in\
                zip(final[i][0],final[i][1],final[i][2],final[i][3],final[i][4],final[i][5],final[i][6],final[i][7]):
                    print(title)
                    print('-'*100)
                    print(("; ").join(author))
                    print('-'*100)
                    print(journal)
                    print(date)
                    print(f"doi: {doi}, PubMedID: {pid}")
                    print(link)
                    print('-'*100)
                    if isinstance(abstract, str):
                        print(abstract)
                    else:
                        print(abstract["AbstractText"][0])
                    print('-'*100)
                    print('*'*100)
                    print('-'*100)

        elif output[-4:] == ".txt":
            with open(output, 'w', encoding="utf-8") as f:
                for i in final:
                    for title,author, abstract, doi, link, pid, date, journal in\
                    zip(final[i][0],final[i][1],final[i][2],final[i][3],final[i][4],final[i][5],final[i][6],final[i][7]):
                        f.write(title)
                        f.write('\n')
                        f.write('-'*100)
                        f.write('\n')
                        f.write(("; ").join(author))
                        f.write('\n')
                        f.write('-'*100)
                        f.write('\n')
                        f.write(journal)
                        f.write('\n')
                        f.write(str(date))
                        f.write('\n')
                        f.write(f"doi: {doi}, PubMedID: {pid}")
                        f.write('\n')
                        f.write(link)
                        f.write('\n')
                        f.write('-'*100)
                        f.write('\n')
                        writeAbstract(abstract, f, "txt")
                        f.write('\n')
                        f.write('-'*100)
                        f.write('\n')
                        f.write('*'*100)
                        f.write('\n')
                        f.write('-'*100)
                        f.write('\n')

        # would be nice to have a nicer format -> idea write with html and save as such! :) noice!
        else:
            if output[-5:] == ".html":
                pass
            else:
                output += "/PubCrawlerResults.html"
            with open(output, 'w', encoding="utf-8") as f:
                f.write("<!DOCTYPE html>")
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
                    f.write(f'<img src="WordCloud.png" alt="WordCloud" class="center">')
                for i in final:
                    for title,author, abstract, doi, link, pid, date, journal in\
                    zip(final[i][0],final[i][1],final[i][2],final[i][3],final[i][4],final[i][5],final[i][6],final[i][7]):
                        f.write(f"<h2>{title}</h2>")
                        f.write('<br>')
                        ta = ("; ").join(author)
                        f.write(f'<p style="color:gray; font-size:16px">{ta}</p>')
                        f.write('<br>')
                        f.write('-'*200)
                        f.write('<br>')
                        f.write(f"<i>{journal}</i>")
                        f.write('<br>')
                        f.write(str(date))
                        f.write('<br>')
                        f.write(f"<i>doi:</i> {doi}, <i>PubMedID:</i> {pid}")
                        f.write('<br>')
                        f.write(f'<a href={link}><i>Link to journal</i></a>')
                        f.write('<br>')
                        f.write('-'*200)
                        f.write('<br>')
                        writeAbstract(abstract, f, "html")
                        f.write('<br>')
                        f.write('-'*200)
                        f.write('<br>')
                        f.write('*'*150)
                        f.write('<br>')
                        f.write('-'*200)
                        f.write('<br>')
                f.write("</body>")
                f.write("</html>")


    def writeAbstract(abstract, f, typ):
        if not isinstance(abstract, str):
            abstract = abstract["AbstractText"][0]
        abstract = abstract.split(" ")

        if typ == "txt":
            for idx, word in enumerate(abstract):
                f.write(word)
                f.write(' ')
                if idx%20==0 and idx>0:
                    f.write('\n')


        elif typ == "html":
            f.write('<p style="color:#202020">')
            for idx, word in enumerate(abstract):
                f.write(word)
                f.write(' ')
                if idx%20==0 and idx>0:
                    f.write('</p>')
                    f.write('<p style="color:#202020">')
            f.write('</p>')



    # if function is executed in a loop a small deleay to ensure pubmed is responding
    time.sleep(0.5)

    if customSearchTerm is None:
        term = makeSearchTerm(text, ToA, author, phrase,
                      exclusion)
    else:
        term = customSearchTerm

    results = search(term, retmax, sort)["IdList"]
    if len(results) > 0:
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

        for paper in results2["PubmedArticle"]:
            #get titles
            titles.append(paper["MedlineCitation"]["Article"]["ArticleTitle"])
            #get abstract
            if "Abstract" in paper["MedlineCitation"]["Article"].keys():
                abstracts.append(paper["MedlineCitation"]["Article"]["Abstract"])
            else:
                abstracts.append("No abstract")
            #get authors
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
            #get dois and make link
            doi = [i for i in paper['PubmedData']['ArticleIdList'] if i.attributes["IdType"]=="doi"]
            if len(doi) > 0:
                doi = str(doi[0])
            else: doi = "NaN"
            dois.append(doi)
            links.append(f" https://doi.org/{doi}")
            #get pids
            pids.append(str([i for i in paper['PubmedData']['ArticleIdList'] if i.attributes["IdType"]=="pubmed"][0]))
            #get dates

            rawDate = paper['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']
            try:
                date = f"{rawDate['Year']}-{rawDate['Month']}-{rawDate['Day']}"
            except:
                date = rawDate
            dates.append(date)
            #get journal
            journals.append(paper['MedlineCitation']['Article']['Journal']['Title'])

        final[term] = (titles,authors,abstracts,dois, links, pids, dates, journals)

        if makeWordCloud == True:
            makewc(final, output)
        genOutput(final, output, makeWordCloud)

    else:
        print("No results for " + term)


def loess(data, xvals, yvals, alpha, poly_degree=1):
    """
    https://medium.com/@langen.mu/creating-powerfull-lowess-graphs-in-python-e0ea7a30b17a
    example:

    evalDF = loess("t1/2_median_HeLa", "t1/2_median_Huh", data = comp, alpha=0.9, poly_degree=1)

    fig = plt.figure()
    ax = plt.subplot()
    sns.scatterplot(comp["t1/2_median_HeLa"], comp["t1/2_median_Huh"])

    ax.plot(evalDF['v'], evalDF['g'], color='red', linewidth= 3, label="Test")
    plt.xlim(1,12)
    plt.ylim(1,12)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes)
    """

    def loc_eval(x, b):
        loc_est = 0
        for i in enumerate(b): loc_est+=i[1]*(x**i[0])
        return(loc_est)

    all_data = sorted(zip(data[xvals].tolist(), data[yvals].tolist()), key=lambda x: x[0])
    xvals, yvals = zip(*all_data)
    evalDF = pd.DataFrame(columns=['v','g'])
    n = len(xvals)
    m = n + 1
    q = int(np.floor(n * alpha) if alpha <= 1.0 else n)
    avg_interval = ((max(xvals)-min(xvals))/len(xvals))
    v_lb = min(xvals)-(.5*avg_interval)
    v_ub = (max(xvals)+(.5*avg_interval))
    v = enumerate(np.linspace(start=v_lb, stop=v_ub, num=m), start=1)
    xcols = [np.ones_like(xvals)]
    for j in range(1, (poly_degree + 1)):
        xcols.append([i ** j for i in xvals])
    X = np.vstack(xcols).T
    for i in v:
        iterpos = i[0]
        iterval = i[1]
        iterdists = sorted([(j, np.abs(j-iterval)) for j in xvals], key=lambda x: x[1])
        _, raw_dists = zip(*iterdists)
        scale_fact = raw_dists[q-1]
        scaled_dists = [(j[0],(j[1]/scale_fact)) for j in iterdists]
        weights = [(j[0],((1-np.abs(j[1]**3))**3 if j[1]<=1 else 0)) for j in scaled_dists]
        _, weights      = zip(*sorted(weights,     key=lambda x: x[0]))
        _, raw_dists    = zip(*sorted(iterdists,   key=lambda x: x[0]))
        _, scaled_dists = zip(*sorted(scaled_dists,key=lambda x: x[0]))
        W         = np.diag(weights)
        b         = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ yvals)
        local_est = loc_eval(iterval, b)
        iterDF2   = pd.DataFrame({
                       'v'  :[iterval],
                       'g'  :[local_est]
                       })
        evalDF = pd.concat([evalDF, iterDF2])
    evalDF = evalDF[['v','g']]
    return(evalDF)


def edm(A,B):
    """
    Function that forms euclidean distrance matrix
    https://medium.com/swlh/euclidean-distance-matrix-4c3e1378d87f
    params:
    :a: matrix; matrix A
    :b: matrix; matrix B
    """
    p1 = np.sum(A**2, axis=1)[:, np.newaxis]
    p2 = np.sum(B**2, axis=1)
    p3 = -2 * np.dot(A,B.T)
    return np.sqrt(p1+p2+p3)


def limma(df, reps, cond="", customDesign=None):
    """
    Function that performs moderated ttest as implemented in limma

    TODO: better handle coefficient extraction in R
    """

    d = os.getcwd()
    dataLoc = d + "/input.csv"
    outputLoc = d + "/output.csv"
    if customDesign is None:
        designLoc = d + "/design.csv"

    if "UID" in df.columns:
        pass
    else:
        df["UID"] = range(1, df.shape[0]+1)

    #if not isinstance(reps, list):
    #    cols = cols.to_list()
    #flatten in case of two sample
    pp.to_csv(df[["UID"] + list(pl.flatten(reps))], dataLoc)

    if customDesign is None:
        if isinstance(reps[0], list) and len(reps) == 2:
            test = "twoSample"
            design = pd.DataFrame({"Intercept":[1]*(len(reps[0])+len(reps[1])),
                                   "coef":[0]*len(reps[0]) + [1]*len(reps[1])})
            pp.to_csv(design, designLoc)
        else:
            test = "oneSample"
    else:
        test = "custom"
        designLoc = customDesign

    command = [R, '--vanilla',
    RSCRIPT, #script location
    "limma", #functionName
    dataLoc, #data location
    outputLoc, #output file,
    test, #kind of test
    designLoc #design location
    ]
    run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)

    res = pp.read_csv(outputLoc)
    res.columns = [i+cond if i != "UID" else i for i in res.columns]
    df = df.merge(res, on="UID")

    os.remove(dataLoc)
    os.remove(outputLoc)
    if customDesign is None:
        if isinstance(reps[0], list) and len(reps) == 2:
            os.remove(designLoc)

    return df


def rankProd(df, reps, cond=""):
    """
    Function that performs RankProd test as in R RankProd package
    At the moment one sample test only
    Test for up and downregulated genes separatly therefore returns two p values
    """

    d = os.getcwd()
    dataLoc = d + "/input.csv"
    outputLoc = d + "/output.csv"

    if "UID" in df.columns:
        pass
    else:
        df["UID"] = range(1, df.shape[0]+1)

    #if not isinstance(reps, list):
    #    cols = cols.to_list()
    #flatten in case of two sample
    pp.to_csv(df[["UID"] + list(pl.flatten(reps))], dataLoc)


    command = [R, '--vanilla',
    RSCRIPT, #script location
    "rankProd", #functionName
    dataLoc, #data location
    outputLoc, #output file,
    ]
    run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)

    res = pp.read_csv(outputLoc)
    res.columns = [i+cond if i != "UID" else i for i in res.columns]
    df = df.merge(res, on="UID")

    os.remove(dataLoc)
    os.remove(outputLoc)

    return df


def annotatePS(df, ps, colsToKeep=[]):
    """
    Function that annotates phosphosites with information derived from PSP.
    @params:
    :df:: dataframe containing PS of interst
    :ps:: column containing info about the PS - format: GeneName_AminoacidPositoin (e.g. AKT_T308)
    :colsToKeep:: which columns from original dataframe (input df) to keep in output
    """
    def makeMergeCol(df, file="regSites"):
        if file == "regSites":
            return df["GENE"].fillna("").apply(lambda x: str(x).upper()) +'_' + df["MOD_RSD"].fillna("").apply(lambda x: x.split('-')[0])
        return df["SUB_GENE"].fillna("").apply(lambda x: str(x).upper()) +'_' + df["SUB_MOD_RSD"].fillna("")

    with resources.open_text("autoprot.data","Kinase_Substrate_Dataset") as d:
                KS = pd.read_csv(d, sep='\t')
                KS["merge"] = makeMergeCol(KS, "KS")
    with resources.open_text("autoprot.data","Regulatory_sites") as d:
                regSites = pd.read_csv(d, sep='\t')
                regSites["merge"] = makeMergeCol(regSites)

    KS_coi = ['KINASE', 'DOMAIN', 'IN_VIVO_RXN', 'IN_VITRO_RXN', 'CST_CAT#', 'merge']
    regSites_coi = ['ON_FUNCTION', 'ON_PROCESS', 'ON_PROT_INTERACT', 'ON_OTHER_INTERACT',
                   'PMIDs', 'NOTES', 'LT_LIT', 'MS_LIT', 'MS_CST', 'merge']

    df = df.copy(deep=True)
    df.rename(columns={ps:"merge"}, inplace=True)
    df = df[["merge"] + colsToKeep]
    df = df.merge(KS[KS_coi], on="merge", how="left")
    df = df.merge(regSites[regSites_coi], on="merge", how="left")

    return df


def goAnalysis(geneList, organism="hsapiens"):
    """
    performs go Enrichment analysis (also KEGG and REAC)
    """
    if not isinstance(geneList, list):
        try:
            geneList = list(geneList)
        except:
            raise ValueError("Please provide a list of gene names")
    return gp.profile(organism=organism, query=geneList,no_evidences=False)


def makePSM(seq, seqLen):
    """
    Function that generates a position score matrix for a set
    of given sequences
    Returns the percentage of each amino acid for each position
    could be further normalized using a PSM of unrelated/background sequences
    """
    aa_dic = {
        'G':0,
        'P':0,
        'A':0,
        'V':0,
        'L':0,
        'I':0,
        'M':0,
        'C':0,
        'F':0,
        'Y':0,
        'W':0,
        'H':0,
        'K':0,
        'R':0,
        'Q':0,
        'N':0,
        'E':0,
        'D':0,
        'S':0,
        'T':0,
    }

    seq = [i for i in seq if len(i)==seqLen]
    seqT = [''.join(s) for s in zip(*seq)]
    scoreMatrix = []
    for pos in seqT:
        d = aa_dic.copy()
        for aa in pos:
            aa = aa.upper()
            if aa == '.' or aa == '-' or aa == '_' or aa == "X":
                pass
            else:
                d[aa]+=1
        scoreMatrix.append(d)

    for pos in scoreMatrix:
        for k in pos.keys():
            pos[k] /= len(seq)

    #empty array -> (sequenceWindow, aa)
    m = np.empty((seqLen,20))
    for i in range(m.shape[0]):
        x = [j for j in scoreMatrix[i].values()]
        m[i] = x

    m = pd.DataFrame(m, columns=aa_dic.keys())

    return m.T