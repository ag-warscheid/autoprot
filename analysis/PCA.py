# -*- coding: utf-8 -*-
"""
Autoprot Analysis Functions.

@author: Wignand, Julian, Johannes

@documentation: Julian
"""
from typing import Union, Literal

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from sklearn.decomposition import PCA
from .. import r_helper

from gprofiler import GProfiler

gp = GProfiler(
    user_agent="autoprot",
    return_dataframe=True)
RFUNCTIONS, R = r_helper.return_r_path()

# check where this is actually used and make it local
cmap = sns.diverging_palette(150, 275, s=80, l=55, n=9)


class AutoPCA:
    # noinspection PyUnresolvedReferences
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

        prot = pd.read_csv("_static/testdata/03_proteinGroups.zip", sep="\t", low_memory=False)
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
    def __init__(self, dataframe: pd.DataFrame, clabels: Union[list[str],None], rlabels: Union[list[str], None] = None,
                 batch: Union[list[str], None] = None):
        """
        Initialise PCA class.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Input dataframe.
        clabels : list or None
            Column labels.
        rlabels : list or None, optional
            Row labels.
            The default is None.
        batch : list, optional
            Labels for distinct conditions used to colour dots in score plot.
            Must be the length of rlabels.
            The default is None.

        Returns
        -------
        None.

        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Provide a pandas dataframe.")
        # drop any rows in the dataframe containing missing values
        if clabels is not None:
            self.X = dataframe[clabels].dropna()
        else:  # use all columns provided
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
    def _perform_pca(dataframe: pd.DataFrame, label: list[str]) -> tuple:
        """Perform pca and generate for_vis dataframe."""
        pca = PCA().fit(dataframe.dropna())
        # components_ is and ndarray of shape (n_components, n_features)
        # and contains the loadings/weights of each PCA eigenvector
        for_vis = pd.DataFrame(pca.components_.T)
        for_vis.columns = [f"PC{i}" for i in range(1, min(dataframe.shape[0], dataframe.T.shape[0]) + 1)]
        for_vis["label"] = label
        return pca, for_vis

    def scree(self, figsize=(15, 5)) -> None:
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

    def corr_comp(self, annot=False, ax: plt.axis = None) -> None:
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
        ax: plt.axis, optional
            axis to plot on. Default is None.

        Notes
        -----
        2D representation how strong each observation (e.g. log protein ratio)
        weights for each principal component.

        Returns
        -------
        None.

        """
        if ax is None:
            fig, ax = plt.subplots(1)
        sns.heatmap(self.forVis.filter(regex="^PC"), cmap=sns.color_palette("PuOr", 10), annot=annot, ax=ax)
        yp = [i + 0.5 for i in range(len(self.label))]
        ax.set_yticks(yp, self.forVis["label"], rotation=0)
        ax.set_title("")

    def bar_load(self, pc: int = 1, n: int = 25) -> None:
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

    def return_load(self, pc: int = 1, n: int = 25) -> pd.DataFrame:
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

    def return_score(self) -> pd.DataFrame:
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

    def score_plot(self, pc1: int = 1, pc2: int = 2, labeling: bool = False, file: str = None,
                   figsize: tuple = (5, 5)) -> None:
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
        This will return a scatter plot with as many points as there are
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

    def loading_plot(self, pc1: int = 1, pc2: int = 2, labeling: bool = False, ax: plt.axis = None,
                     figsize: tuple[int] = (5, 5)):
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
            The size of the figure object. Will be ignored if ax is not None.
            The default is (5,5).
        ax: plt.axis, optional.
            The axis to plot on. Default is None.

        Notes
        -----
        This will return a scatter plot with as many points as there are
        components (i.e. conditions) in the dataset.
        For each component a load magnitude for two PCs will be printed
        that describes how much each condition influences the magnitude
        of the respective PC.

        Returns
        -------
        None.

        """
        if ax is None:
            fig, ax = plt.subplots(1, figsize=figsize)
        if self.batch is None or len(self.batch) != self.forVis.shape[0]:
            sns.scatterplot(data=self.forVis, x=f"PC{pc1}",
                            y=f"PC{pc2}", edgecolor=None, ax=ax)
        else:
            sns.scatterplot(data=self.forVis, x=f"PC{pc1}",
                            y=f"PC{pc2}", edgecolor=None, hue=self.batch, ax=ax)
        sns.despine()

        ax.set_title("Loadings plot")
        ax.set_xlabel(f"PC{pc1}\n{round(self.expVar[pc1 - 1] * 100, 2)} %")
        ax.set_ylabel(f"PC{pc2}\n{round(self.expVar[pc2 - 1] * 100, 2)} %")

        if labeling is True:
            ss = self.forVis["label"]
            xx = self.forVis[f"PC{pc1}"]
            yy = self.forVis[f"PC{pc2}"]
            for x, y, s in zip(xx, yy, ss):
                ax.text(x, y, s)

    def bi_plot(self, pc1: int = 1, pc2: int = 2, num_load: Union[Literal["all"], int] = "all",
                figsize: tuple[int, int] = (5, 5), **kwargs) -> None:
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

    def pair_plot(self, n: int = 0) -> None:
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
