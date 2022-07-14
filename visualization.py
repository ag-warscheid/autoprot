# -*- coding: utf-8 -*-
"""
Autoprot Visualisation Functions.

@author: Wignand

@documentation: Julian
"""
from scipy import stats
from scipy.stats import zscore
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import pylab as pl
from autoprot import venn
from matplotlib_venn import venn2
from matplotlib_venn import venn3
import logomaker
import colorsys
import matplotlib.patches as patches
from itertools import chain

from wordcloud import WordCloud
from wordcloud import STOPWORDS

from pdfminer3.layout import LAParams
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer3.converter import TextConverter

import io
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

plt.rcParams['pdf.fonttype'] = 42


# TODO: Add functionality of embedding all the plots as subplots in figures by providing ax parameter

def correlogram(df, columns=None, file="proteinGroups", log=True, save_dir=None,
                save_type="pdf", save_name="pairPlot", lower_triang="scatter",
                sample_frac=None, bins=100, ret_fig=False):
    r"""Plot a pair plot of the dataframe intensity columns in order to assess the reproducibility.

    Notes
    -----
    The lower half of the correlogram shows a scatter plot comparing pairs of
    conditions while the upper part shows you the color coded correlation
    coefficients as well as the intersection of hits between both conditions.
    In tiles corresponding to self-comparison (the same value on y and x axis)
    a histogram of intensities is plotted.

    Parameters
    ----------
    df : pd.df
        Dataframe from MaxQuant file.
    columns : list of strings, optional
        The columns to be visualized. The default is empty list.
    file : str, optional
        "proteinGroups" or "Phospho(STY)" (does only change annotation).
        The default is "proteinGroups".
    log : bool, optional
        Whether provided intensities are already log transformed.
        The default is True.
    save_dir : str, optional
        Where the plots are saved. The default is None.
    save_type : str, optional
        What format the saved plots have (pdf, png). The default is "pdf".
    save_name : str, optional
        The name of the saved file. The default is "pairPlot".
    lower_triang : "scatter", "hexBin" or "hist2d", optional
        The kind of plot displayed in the lower triang.
        The default is "scatter".
    sample_frac : float, optional
        Fraction between 0 and 1 to indicate fraction of entries to be shown in scatter.
        Might be useful for large correlograms in order to make it possible
        to work with those in illustrator.
        The default is None.
    bins : int, optional
        Number of bins for histograms.
        The default is 100.
    ret_fig : bool, optional
        Wether to return the seaborn figure object

    Raises
    ------
    ValueError
        If provided list of columns is not suitable.

    Returns
    -------
    None.

    Examples
    --------
    You may for example plot the protein intensitites of a single condition of
    your experiment .

    >>> autoprot.visualization.correlogram(prot,mildLogInt, file='proteinGroups', lower_triang="hist2d")

    .. plot::
        :context_: close-figs

        import pandas as pd
        import autoprot.preprocessing as pp
        import autoprot.visualization as vis

        twitchInt = ['Intensity H BC18_1','Intensity M BC18_2','Intensity H BC18_3',
                     'Intensity H BC36_1','Intensity H BC36_2','Intensity M BC36_2']
        ctrlInt = ["Intensity L BC18_1","Intensity L BC18_2","Intensity L BC18_3",
                   "Intensity L BC36_1", "Intensity L BC36_2","Intensity L BC36_2"]
        mildInt = ["Intensity M BC18_1","Intensity H BC18_2","Intensity M BC18_3",
                   "Intensity M BC36_1","Intensity M BC36_2","Intensity H BC36_2"]

        prot = pd.read_csv("_static/testdata/proteinGroups.zip", sep='\t', low_memory=False)
        prot = pp.log(prot, twitchInt+ctrlInt+mildInt, base=10)
        twitchLogInt = [f"log10_{i}" for i in twitchInt]
        mildLogInt = [f"log10_{i}" for i in mildInt]

        vis.correlogram(prot,mildLogInt, file='proteinGroups', lower_triang="hist2d")
        plt.show()

    You may want to change the plot type on the lower left triangle.

    >>> autoprot.visualization.correlogram(prot,mildLogInt, file='proteinGroups', lower_triang="hexBin")

    .. plot::
        :context_: close-figs

        vis.correlogram(prot,mildLogInt, file='proteinGroups', lower_triang="hexBin")

    """

    if columns is None:
        columns = list()
    if columns is None:
        columns = []

    def getColor(r):
        colors = {
            0.8: "#d67677",
            0.81: "#d7767c",
            0.82: "#d87681",
            0.83: "#da778c",
            0.84: "#dd7796",
            0.85: "#df78a1",
            0.86: "#e179ad",
            0.87: "#e379b8",
            0.88: "#e57ac4",
            0.89: "#e77ad0",
            0.90: "#ea7bdd",
            0.91: "#ec7bea",
            0.92: "#e57cee",
            0.93: "#dc7cf0",
            0.94: "#d27df2",
            0.95: "#c87df4",
            0.96: "#be7df6",
            0.97: "#b47ef9",
            0.98: "#a97efb",
            0.99: "#9e7ffd",
            1: "#927fff"
        }
        return "#D63D40" if r <= 0.8 else colors[np.round(r, 2)]

    def corrfunc(x, y, **kws):
        """Calculate correlation coefficient and add text to axis."""
        df = pd.DataFrame({"x": x, "y": y})
        df = df.dropna()
        x = df["x"].values
        y = df["y"].values
        r, _ = stats.pearsonr(x, y)
        ax = plt.gca()
        ax.annotate("r = {:.2f}".format(r),
                    xy=(.1, .9), xycoords=ax.transAxes)

    def heatmap(x, y, **kws):
        """Calculate correlation coefficient and add coloured tile to axis."""
        df = pd.DataFrame({"x": x, "y": y})
        df = df.replace(-np.inf, np.nan).dropna()
        x = df["x"].values
        y = df["y"].values
        r, _ = stats.pearsonr(x, y)
        ax = plt.gca()
        ax.add_patch(mpl.patches.Rectangle((0, 0), 5, 5,
                                           color=getColor(r),
                                           transform=ax.transAxes))
        ax.tick_params(axis="both", which="both", length=0)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    def lower_scatter(x, y, **kws):
        """Plot data points as scatter plot to axis."""
        data = pd.DataFrame({"x": x, "y": y})
        if sample_frac is not None:
            data = data.sample(int(data.shape[0] * sample_frac))
        ax = plt.gca()
        ax.scatter(data['x'], data['y'], linewidth=0)

    def lower_hex_bin(x, y, **kws):
        """Plot data points as hexBin plot to axis."""
        plt.hexbin(x, y, cmap="Blues", bins=bins,
                   gridsize=50)

    def lower_hist_2D(x, y, **kws):
        """Plot data points as hist2d plot to axis."""
        df = pd.DataFrame({"x": x, "y": y})
        df = df.dropna()
        x = df["x"].values
        y = df["y"].values
        plt.hist2d(x, y, bins=bins, cmap="Blues", vmin=0, vmax=1)

    def proteins_found(x, y, **kws):
        df = pd.DataFrame({"x": x, "y": y})
        df = df.dropna()
        x = df["x"].values
        y = df["y"].values
        r, _ = stats.pearsonr(x, y)
        ax = plt.gca()
        if file == "Phospho (STY)":
            ax.annotate(f"{len(y)} peptides identified", xy=(0.1, 0.9), xycoords=ax.transAxes)
            ax.annotate(f"R: {str(round(r, 2))}", xy=(0.25, 0.5), size=18, xycoords=ax.transAxes)

        elif file == "proteinGroups":
            ax.annotate(f"{len(y)} proteins identified", xy=(0.1, 0.9), xycoords=ax.transAxes)
            ax.annotate(f"R: {str(round(r, 2))}", xy=(0.25, 0.5), size=18, xycoords=ax.transAxes)

    if len(columns) == 0:
        raise ValueError("No columns provided!")
    else:
        # select columns for plotting
        temp_df = df[columns]

    # perform log transformation if not already done
    if not log:
        temp_df[columns] = np.log10(temp_df[columns])
    # avoid inf values from log transformation
    y = temp_df.replace(-np.inf, np.nan)

    # maps each pairwise combination of column onto an axis grid
    g = sns.PairGrid(y)
    # accesses the lower triangle
    g.map_lower(corrfunc)
    # plot the data points on the lower triangle
    if lower_triang == "scatter":
        g.map_lower(lower_scatter)
    elif lower_triang == "hexBin":
        g.map_lower(lower_hex_bin)
    elif lower_triang == "hist2d":
        g.map_lower(lower_hist_2D)
    # histograms on the diagonal
    g.map_diag(sns.histplot)
    # coloured tiles for the upper triangle
    g.map_upper(heatmap)
    # annotate the number of identified proteins
    g.map_upper(proteins_found)

    if save_dir is not None:
        if save_type == "pdf":
            plt.savefig(f"{save_dir}/{save_name}.pdf")
        elif save_type == "png":
            plt.savefig(f"{save_dir}/{save_name}.png")

    if ret_fig:
        return g


def corrMap(df, columns, cluster=False, annot=None, cmap="YlGn", figsize=(7, 7),
            saveDir=None, saveType="pdf", saveName="pairPlot", ax=None, **kwargs):
    r"""
    Plot correlation heat- and clustermaps.

    Parameters
    ----------
    df : pd.df
        Dataframe from MaxQuant file.
    columns : list of strings, optional
        The columns to be visualized. The default is None.
    cluster : bool, optional
        Whether to plot a clustermap.
        If True, only a clustermap will be returned.
        The default is False.
    annot : bool or rectangular dataset, optional
        If True, write the data value in each cell.
        If an array-like with the same shape as data, then use this to annotate
        the heatmap instead of the data. Note that DataFrames will match on
        position, not index. The default is None.
    cmap : matplotlib colormap name or object, or list of colors, optional
        The mapping from data values to color space.
        The default is "YlGn".
    figsize : tuple of int, optional
        Size of the figure. The default is (7,7).
    saveDir : str, optional
        Where the plots are saved. The default is None.
    saveType : str, optional
        What format the saved plots have (pdf, png). The default is "pdf".
    saveName : str, optional
        The name of the saved file. The default is "pairPlot".
    ax : plt.axis, optional
        The axis to plot. The default is None.
    **kwargs :
        passed to seaborn.heatmap and seaborn.clustermap.

    Returns
    -------
    None.

    Examples
    --------
    To plot a heatmap with annotated values call corrMap directly:

    >>> autoprot.visualization.corrMap(prot,mildLogInt, annot=True)

    .. plot::
        :context: close-figs

        import autoprot.preprocessing as pp
        import autoprot.visualization as vis

        prot = pd.read_csv("_static/testdata/proteinGroups.zip", sep='\t', low_memory=False)
        mildInt = ["Intensity M BC18_1","Intensity H BC18_2","Intensity M BC18_3",
                   "Intensity M BC36_1","Intensity M BC36_2","Intensity H BC36_2"]
        prot = pp.log(prot, mildInt, base=10)
        mildLogInt = [f"log10_{i}" for i in mildInt]
        vis.corrMap(prot,mildLogInt, annot=True)
        plt.show()

    If you want to plot the clustermap, set cluster to True.
    The correlation coefficients are colour-coded.

    >>>  autoprot.visualization.corrMap(prot, mildLogInt, cmap="autumn", annot=None, cluster=True)

    .. plot::
        :context: close-figs

        vis.corrMap(prot, mildLogInt, cmap="autumn", annot=None, cluster=True)
        plt.show()
    """
    corr = df[columns].corr()
    if cluster:
        sns.clustermap(corr, cmap=cmap, annot=annot, **kwargs)

    elif ax is None:
        plt.figure(figsize=figsize)
        sns.heatmap(corr, cmap=cmap, square=True, cbar=False, annot=annot, **kwargs)
    else:
        sns.heatmap(corr, cmap=cmap, square=True, cbar=False, annot=annot, ax=ax, **kwargs)
    if saveDir is not None:
        if saveType == "pdf":
            plt.savefig(f"{saveDir}/{saveName}.pdf")
        elif saveType == "png":
            plt.savefig(f"{saveDir}/{saveName}.png")


def probPlot(df, col, dist="norm", figsize=(6, 6)):
    r"""
    Plot a QQ_plot of the provided column.

    Data are compared against a theoretical distribution (default is normal)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    col : list of str
        Columns containing the data for analysis.
    dist : str or stats.distributions instance, optional, optional
        Distribution or distribution function name.
        The default is ‘norm’ for a normal probability plot.
        Objects that look enough like a stats.distributions instance
        (i.e. they have a ppf method) are also accepted.
    figsize : tuple of int, optional
        Size of the figure. The default is (6,6).

    Returns
    -------
    None.

    Examples
    --------
    Plot to check if the experimental data points follow the distribution function
    indicated by dist.

    >>> vis.probPlot(prot,'log10_Intensity H BC18_1')

    .. plot::
        :context: close-figs

        import autoprot.preprocessing as pp
        import autoprot.visualization as vis
        import autoprot.analysis as ana
        import pandas as pd

        prot = pd.read_csv("_static/testdata/proteinGroups.zip", sep='\t', low_memory=False)
        prot = pp.cleaning(prot, "proteinGroups")
        protInt = prot.filter(regex='Intensity').columns
        prot = pp.log(prot, protInt, base=10)

        vis.probPlot(prot,'log10_Intensity H BC18_1')
        plt.show()

    In contrast when the data does not follow the distribution, outliers from the
    linear plot will be visible.

    >>> vis.probPlot(prot,'log10_Intensity H BC18_1', dist=stats.uniform)

    .. plot::
        :context: close-figs

        import scipy.stats as stats
        vis.probPlot(prot,'log10_Intensity H BC18_1', dist=stats.uniform)

    """
    t = stats.probplot(df[col].replace([-np.inf, np.inf], [np.nan, np.nan]).dropna(), dist=dist)
    label = f"R²: {round(t[1][2], 4)}"
    y = []
    x = []
    for i in np.linspace(min(t[0][0]), max(t[0][0]), 100):
        y.append(t[1][0] * i + t[1][1])
        x.append(i)
    plt.figure(figsize=figsize)
    plt.scatter(t[0][0], t[0][1], alpha=.3, color="purple",
                label=label)
    plt.plot(x, y, color="teal")
    sns.despine()
    plt.title(f"Probability Plot\n{col}")
    plt.xlabel("Theorectical Quantiles")
    plt.ylabel("Sample Quantiles")
    plt.legend()


def boxplot(df, reps: list, title=None, labels=None, compare=False,
            ylabel="log_fc", file=None, ret_fig=False, figsize=(15, 5),
            **kwargs):
    r"""
    Plot intensity boxplots.

    Parameters
    ----------
    df : pd.Dataframe
        INput dataframe.
    reps : list
        Colnames of replicates.
    title : str, optional
        Title of the plot. The default is None.
    labels : list of str, optional
        List with labels for the axis.
        The default is [].
    compare : bool, optional
        If False reps is expected to be a single list,
        if True two list are expected (e.g. normalized and non-normalized Ratios).
        The default is False.
    ylabel : str, optional
        Either "log_fc" or "Intensity". The default is "log_fc".
    file : str, optional
        Path to a folder where the figure should be saved.
        The default is None.
    ret_fig : bool, optional
        Whether to return the figure object.
        The default is False.
    figsize : tuple of int, optional
        Figure size. The default is (15,5).
    **kwargs :
        Passed to pandas boxplot.

    Raises
    ------
    ValueError
        If the reps input does not match the compare setting.

    Returns
    -------
    fig : plt.figure
        Plot figure object.

    Examples
    --------
    To inspect unnormalised data, you can generate a boxplot comparing the
    fold-change differences between conditions or replicates

    >>> autoprot.visualization.boxplot(df=prot,reps=protRatio, compare=False,
    ...                                labels=labels, title="Unnormalized Ratios Boxplot",
    ...                                ylabel="log_fc")

    .. plot::
        :context: close-figs

        import pandas as pd
        import autoprot.visualization as vis
        import autoprot.preprocessing as pp

        prot = pd.read_csv("_static/testdata/proteinGroups.zip", sep='\t', low_memory=False)
        prot = pp.cleaning(prot, "proteinGroups")
        protRatio = prot.filter(regex="Ratio .\/. BC.*_1").columns
        prot = pp.log(prot, protRatio, base=2)
        protRatio = prot.filter(regex="log2_Ratio.*").columns
        prot = pp.vsn(prot, protRatio)
        protRatio = prot.filter(regex="log2_Ratio.*_1$").columns
        labels = [i.split(" ")[1]+"_"+i.split(" ")[-1] for i in protRatio]
        vis.boxplot(df=prot,reps=protRatio, compare=False, labels=labels, title="Unnormalized Ratios Boxplot",
                ylabel="log_fc")
        plt.show()

    If you have two datasets for comparison (e.g. normalised and non-normalised)
    fold-changes, you can use boxplot to plot them side-by-side.

    >>> vis.boxplot(prot,[protRatio, protRatioNorm], compare=True, labels=labels,
    ...             title=["unormalized", "normalized"], ylabel="log_fc")

    .. plot::
        :context: close-figs

        protRatioNorm = prot.filter(regex="log2_Ratio.*normalized").columns
        vis.boxplot(prot,[protRatio, protRatioNorm], compare=True, labels=labels, title=["unormalized", "normalized"],
                   ylabel="log_fc")
    """
    if labels is None:
        labels = []
    # check if inputs make sense
    if compare and len(reps) != 2:
        raise ValueError("You want to compare two sets, provide two sets.")

    if compare:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        ax[0].set_ylabel(ylabel)
        ax[1].set_ylabel(ylabel)
        if title:
            for idx, t in enumerate(title):
                ax[idx].set_title(f"{t}")

        for idx, rep in enumerate(reps):
            df[rep].boxplot(ax=ax[idx], **kwargs)
            ax[idx].grid(False)
            if ylabel == "log_fc":
                ax[idx].axhline(0, 0, 1, color="gray", ls="dashed")

        if len(labels) > 0:
            for idx in [0, 1]:
                temp = ax[idx].set_xticklabels(labels)
                tlabel = ax[idx].get_xticklabels()
                for i, label in enumerate(tlabel):
                    label.set_y(label.get_position()[1] - (i % 2) * .05)
        else:
            ax[0].set_xticklabels([str(i + 1) for i in range(len(reps[0]))])
            ax[1].set_xticklabels([str(i + 1) for i in range(len(reps[1]))])
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

        df[reps].boxplot(**kwargs)
        ax.grid(False)
        plt.title(title)
        plt.ylabel(ylabel)

        if len(labels) > 0:
            temp = ax.set_xticklabels(labels)
            ax.set_xticklabels(labels)
            for i, label in enumerate(temp):
                label.set_y(label.get_position()[1] - (i % 2) * .05)
        else:
            ax.set_xticklabels(str(i + 1) for i in range(len(reps)))
        if ylabel == "log_fc":
            ax.axhline(0, 0, 1, color="gray", ls="dashed")
    sns.despine()

    if file is not None:
        plt.savefig(fr"{file}/BoxPlot.pdf")
    if ret_fig:
        return fig


def intensityRank(data, rankCol="log10_Intensity", label=None, n=5,
                  title="Rank Plot", figsize=(15, 7), file=None, hline=None,
                  ax=None, **kwargs):
    """
    Draw a rank plot.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe.
    rankCol : str, optional
        the column with the values to be ranked (e.g. Intensity values).
        The default is "log10_Intensity".
    label : str, optional
        Colname of the column with the labels.
        The default is None.
    n : int, optional
        How many points to label on the top and bottom of the y-scale.
        The default is 5.
    title : str, optional
        The title of the plot.
        The default is "Rank Plot".
    figsize : tuple of int, optional
        The figure size. The default is (15,7).
    file : str, optional
        Path to a folder where the resulting sigure should be saved.
        The default is None.
    hline : numeric, optional
        y value to place a horizontal line.
        The default is None.
    ax : matplotlib.axis
        Axis to plot on
    **kwargs :
        Passed to seaborn.scatterplot.

    Returns
    -------
    None.

    Examples
    --------
    Annotate a protein groups datafile with the proteins of highest and lowest
    intensity. The 15 most and least intense proteins will be labelled.
    Note that marker is passed to seaborn and results in points marked as diamonds.

    >>> autoprot.visualization.intensityRank(data, rankCol="log10_Intensity",
    ...                                      label="Gene names", n=15,
    ...                                      title="Rank Plot",
    ...                                      hline=8, marker="d")

    .. plot::
        :context: close-figs

        data = pp.log(prot,["Intensity"], base=10)
        data = data[["log10_Intensity", "Gene names"]]
        data = data[data["log10_Intensity"]!=-np.inf]

        vis.intensityRank(data, rankCol="log10_Intensity", label="Gene names", n=15, title="Rank Plot",
                         hline=8, marker="d")

    """
    # ToDo: add option to highlight a set of datapoints could be alternative to topN labeling

    # remove NaNs
    data = data.copy().dropna(subset=[rankCol])

    # if data has mroe columns than 1
    if data.shape[1] > 1:
        data = data.sort_values(by=rankCol, ascending=True)
        y = data[rankCol]
    else:
        y = data.sort_values(ascending=True)

    x = range(data.shape[0])

    # new plot if no axis was given
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()

    # plot on the axis
    sns.scatterplot(x=x, y=y,
                    linewidth=0,
                    ax=ax,
                    **kwargs)

    if hline is not None:
        ax.axhline(hline, 0, 1, ls="dashed", color="lightgray")

    if label is not None:
        # high intensity labels labels
        top_y = y.iloc[-1]

        top_yy = np.linspace(top_y - n * 0.4, top_y, n)
        top_oy = y[-n:]
        top_xx = x[-n:]
        top_ss = data[label].iloc[-n:]

        for ys, xs, ss, oy in zip(top_yy, top_xx, top_ss, top_oy):
            ax.plot([xs, xs + len(x) * .1], [oy, ys], color="gray")
            ax.text(x=xs + len(x) * .1, y=ys, s=ss)

        # low intensity labels
        low_y = y.iloc[0]
        low_yy = np.linspace(low_y, low_y + n * 0.4, n)
        low_oy = y[:n]
        low_xx = x[:n]
        low_ss = data[label].iloc[:n]

        for ys, xs, ss, oy in zip(low_yy, low_xx, low_ss, low_oy):
            ax.plot([xs, xs + len(x) * .1], [oy, ys], color="gray")
            ax.text(x=xs + len(x) * .1, y=ys, s=ss)

    sns.despine()
    ax.set_xlabel("# rank")
    if ax is None:
        plt.title(title)

    if file is not None:
        plt.savefig(fr"{file}/RankPlot.pdf")


def vennDiagram(df, figsize=(10, 10), retFig=False, proportional=True):
    r"""
    Draw vennDiagrams.

    The .vennDiagram() function allows to draw venn diagrams for 2 to 6 replicates.
    Even though you can compare 6 replicates in a venn diagram does not mean
    that you should. It becomes extremly messy.

    The labels in the diagram can be read as follows:
    Comparing two conditions you will see the labels 10, 11 and 01. This can be read as:
    Only in replicate 1 (10), in both replicates (11) and only in replicate 2 (01).
    The same notation extends to all venn diagrams.

    Notes
    -----
    vennDiagram compares row containing not NaN between columns. Therefore,
    you have to pass columns containing NaN on rows where no common protein was
    found (e.g. after ratio calculation).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    figsize : tuple of int, optional
        Figure size. The default is (10,10).
    retFig : bool, optional
        Whether to return the figure.
        The default is False.
    proportional : bool, optional
        Whether to draw area-proportiona Venn diagrams.
        The default is True.

    Raises
    ------
    ValueError
        If the number of provided columns is below 2 or greater than 6.

    Returns
    -------
    fig : matplotlib.figure
        The figure object.
        Only returned if ret_fig is True, else None.

    Examples
    --------
    You can specify up to 6 columns containing values and NaNs. Only rows showing
    values in two columns will be grouped together in the Venn diagram.

    >>> data = prot[twitchVsmild[:3]]
    >>> autoprot.visualization.vennDiagram(data, figsize=(5,5))

    .. plot::
        :context: close-figs

        import autoprot.preprocessing as pp
        import autoprot.visualization as vis
        import pandas as pd

        prot = pd.read_csv("_static/testdata/proteinGroups.zip", sep='\t', low_memory=False)
        prot = pp.cleaning(prot, "proteinGroups")
        protRatio = prot.filter(regex="Ratio .\/. BC.*").columns
        prot = pp.log(prot, protRatio, base=2)

        twitchVsmild = ['Ratio H/M BC18_1','Ratio M/L BC18_2','Ratio H/M BC18_3',
                        'Ratio H/L BC36_1','Ratio H/M BC36_2','Ratio M/L BC36_2']


        data = prot[twitchVsmild[:3]]
        vis.vennDiagram(data, figsize=(5,5))
        plt.show()

    Only up to three conditions can be compared in non-proportional Venn
    diagrams

    >>> autoprot.visualization.vennDiagram(data, figsize=(5,5), proportional=False)

    .. plot::
        :context: close-figs

        vis.vennDiagram(data, figsize=(5,5), proportional=False)
        plt.show()

    Copmaring up to 6 conditions is possible but the resulting Venn diagrams
    get quite messy.

    >>> data = prot[twitchVsmild[:6]]
    >>> vis.vennDiagram(data, figsize=(20,20))

    .. plot::
        :context: close-figs

        data = prot[twitchVsmild[:6]]
        vis.vennDiagram(data, figsize=(20,20))
        plt.show()

    """
    data = df.copy(deep=True)
    n = data.shape[1]
    if n > 6:
        raise ValueError("You cannot analyse more than 6 conditions in a venn diagram!")
    elif n == 1:
        raise ValueError("You should at least provide 2 conditions to compare in a venn diagram!")
    reps = data.columns.to_list()
    data["UID"] = range(data.shape[0])
    if n == 2:
        g1 = data[[reps[0]] + ["UID"]]
        g2 = data[[reps[1]] + ["UID"]]
        g1 = set(g1["UID"][g1[reps[0]].notnull()].values)
        g2 = set(g2["UID"][g2[reps[1]].notnull()].values)
        if proportional:
            venn2([g1, g2], set_labels=reps)
        else:
            labels = venn.get_labels([g1, g2], fill=["number", "logic"])
            fig, ax = venn.venn2(labels, names=[reps[0], reps[1]], figsize=figsize)
            if retFig:
                return fig

    elif n == 3:
        g1 = data[[reps[0]] + ["UID"]]
        g2 = data[[reps[1]] + ["UID"]]
        g3 = data[[reps[2]] + ["UID"]]
        g1 = set(g1["UID"][g1[reps[0]].notnull()].values)
        g2 = set(g2["UID"][g2[reps[1]].notnull()].values)
        g3 = set(g3["UID"][g3[reps[2]].notnull()].values)
        if proportional:
            venn3([g1, g2, g3], set_labels=reps)
        else:
            labels = venn.get_labels([g1, g2, g3], fill=["number", "logic"])
            fig, ax = venn.venn3(labels, names=[reps[0], reps[1], reps[2]], figsize=figsize)
            if retFig:
                return fig

    elif n == 4:
        g1 = data[[reps[0]] + ["UID"]]
        g2 = data[[reps[1]] + ["UID"]]
        g3 = data[[reps[2]] + ["UID"]]
        g4 = data[[reps[3]] + ["UID"]]
        g1 = set(g1["UID"][g1[reps[0]].notnull()].values)
        g2 = set(g2["UID"][g2[reps[1]].notnull()].values)
        g3 = set(g3["UID"][g3[reps[2]].notnull()].values)
        g4 = set(g4["UID"][g4[reps[3]].notnull()].values)
        labels = venn.get_labels([g1, g2, g3, g4], fill=["number", "logic"])
        fig, ax = venn.venn4(labels, names=[reps[0], reps[1], reps[2], reps[3]], figsize=figsize)

        if retFig:
            return fig
    elif n == 5:
        g1 = data[[reps[0]] + ["UID"]]
        g2 = data[[reps[1]] + ["UID"]]
        g3 = data[[reps[2]] + ["UID"]]
        g4 = data[[reps[3]] + ["UID"]]
        g5 = data[[reps[4]] + ["UID"]]
        g1 = set(g1["UID"][g1[reps[0]].notnull()].values)
        g2 = set(g2["UID"][g2[reps[1]].notnull()].values)
        g3 = set(g3["UID"][g3[reps[2]].notnull()].values)
        g4 = set(g4["UID"][g4[reps[3]].notnull()].values)
        g5 = set(g5["UID"][g5[reps[4]].notnull()].values)
        labels = venn.get_labels([g1, g2, g3, g4, g5], fill=["number", "logic"])
        fig, ax = venn.venn5(labels, names=[reps[0], reps[1], reps[2], reps[3], reps[4]], figsize=figsize)

        if retFig:
            return fig
    elif n == 6:
        g1 = data[[reps[0]] + ["UID"]]
        g2 = data[[reps[1]] + ["UID"]]
        g3 = data[[reps[2]] + ["UID"]]
        g4 = data[[reps[3]] + ["UID"]]
        g5 = data[[reps[4]] + ["UID"]]
        g6 = data[[reps[5]] + ["UID"]]
        g1 = set(g1["UID"][g1[reps[0]].notnull()].values)
        g2 = set(g2["UID"][g2[reps[1]].notnull()].values)
        g3 = set(g3["UID"][g3[reps[2]].notnull()].values)
        g4 = set(g4["UID"][g4[reps[3]].notnull()].values)
        g5 = set(g5["UID"][g5[reps[4]].notnull()].values)
        g6 = set(g6["UID"][g6[reps[5]].notnull()].values)
        labels = venn.get_labels([g1, g2, g3, g4, g5, g6], fill=["number", "logic"])
        fig, ax = venn.venn6(labels, names=[reps[0], reps[1], reps[2], reps[3], reps[4], reps[5]], figsize=figsize)

        if retFig:
            return fig


def volcano(df, logFC, p=None, score=None, pt=0.05, fct=None, annot=None,
            interactive=False, sig_col="green", bg_col="lightgray",
            title="Volcano Plot", figsize=(6, 6), hover_name=None, highlight=None,
            pointsize_name=None,
            highlight_col="red", annot_highlight="all", custom_bg=None,
            custom_fg=None, custom_hl=None, ret_fig=False, ax=None, legend=True):
    r"""
    Draw Volcano plot.

    This function can either plot a static or an interactive version of the
    volcano. Further it allows the user to set the desired log_fc and p value
    threshold as well as toggle the annotation of the plot. If provided it is
    possible to highlight a selection of datapoints in the plot.
    Those will then be annotated instead of all significant entries.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe which contains the data.
    logFC : str
        column of the dataframe with the log fold change.
    p : str, optional
        column of the dataframe containing p values (provide score or p).
        The default is None.
    score : str, optional
        column of the dataframe containing -log10(p values) (provide score or p).
        The default is None.
    pt : float, optional
        p-value threshold under which a entry is deemed significantly regulated.
        The default is 0.05.
    fct : float, optional
        fold change threshold at which an entry is deemed significant regulated.
        The default is None.
    annot : str, optional
        Column name to annotate the plot. The default is None.
    interactive : bool, optional
         The default is False.
    sig_col : str, optional
        Colour for significant points. The default is "green".
    bg_col : str, optional
        Background colour. The default is "lightgray".
    title : str, optional
        Title for the plot. The default is "Volcano Plot".
    figsize : tuple of int, optional
        Size of the figure. The default is (6,6).
    hover_name : str, optional
        Colname to use for labels in interactive plot.
        The default is None.
    highlight : pd.index, optional
        Rows to highlight in the plot.
        The default is None.
    pointsize_name: str or float, optional
        Name of a column to use as emasure for point size.
        Alternatively the size of all points.
    highlight_col : str, optional
        Colour for the highlights. The default is "red".
    annot_highlight : str, optional
        'all' or 'sig'.
        Whether to highlight all rows in indicated by highlight or only
        the significant positions.
        The default is "all".
    custom_bg : dict, optional
        Key:value pairs that are passed as kwargs to plt.scatter to define the background.
        Ignored for the interactive plots.
        The default is None.
    custom_fg : dict, optional
        Key:value pairs that are passed as kwargs to plt.scatter to define the foreground.
        Ignored for the interactive plots.
        The default is None.
    custom_hl : dict, optional
        Key:value pairs that are passed as kwargs to plt.scatter to define the highlighted points.
        Ignored for the interactive plots.
        The default is None.
    ret_fig : bool, optional
        Whether to return the figure, can be used to further
        customize it afterwards. The default is False.
    ax : matplotlib.axis, optional
        Axis to print on. The default is None.
    legend : bool, optional
        Whether to plot a legend. The default is True.

    Raises
    ------
    ValueError
        If neither a p-score nor a p value is provided by the user.

    Notes
    -----
    Setting a strict log_fc threshold is arbitrary and should generally be avoided.
    Annotation of volcano plot can become cluttered quickly.
    You might want to prettify annotation in illustrator.
    Alternatively, use VolcanoDashBoard to interactively annotate your Volcano
    plot or an interactive version of volcano plot to investigate the results.

    Returns
    -------
    matplotlib.figure
        The figure object.

    Examples
    --------
    The function .volcano() draws a volcano plot.
    You can either provide precalculated scores or raw (adjusted) p values.
    You can also set a desired significance threshold (p value as well as log_fc).
    You can customize the volcano plot, for instance you can also choose between
    interactive and static plot. When you provide a set of indices in the
    highlight parameter those will be highlighted for you in the plot.

    >>> vis.volcano(df=prot_limma, log_fc="logFC_TvM", p="P.Value_TvM")

    .. plot::
        :context: close-figs

        import autoprot.preprocessing as pp
        import autoprot.visualization as vis
        import autoprot.analysis as ana
        import pandas as pd

        prot = pd.read_csv("_static/testdata/proteinGroups.zip", sep='\t', low_memory=False)
        prot = pp.cleaning(prot, "proteinGroups")
        protRatio = prot.filter(regex="^Ratio .\/.( | normalized )B").columns
        prot = pp.log(prot, protRatio, base=2)
        twitchVsmild = ['log2_Ratio H/M normalized BC18_1','log2_Ratio M/L normalized BC18_2','log2_Ratio H/M normalized BC18_3',
                         'log2_Ratio H/L normalized BC36_1','log2_Ratio H/M normalized BC36_2','log2_Ratio M/L normalized BC36_2']
        prot_limma = ana.limma(prot, twitchVsmild, cond="_TvM")
        vis.volcano(df=prot_limma, log_fc="logFC_TvM", p="P.Value_TvM")
        plt.show()

    The volcano function allows a certain amount of customisation. E.g. proteins
    exceeding a certain threshold can be annotated and titles and colours can
    be adapted to your needs.

    >>> vis.volcano(df=prot_limma, log_fc="logFC_TvM", p="P.Value_TvM", pt=0.01,
    ...             fct=2, annot="Gene names", sig_col="purple", bg_col="teal",
    ...             title="Custom Title", figsize=(15,5))

    .. plot::
        :context: close-figs

        vis.volcano(df=prot_limma, log_fc="logFC_TvM", p="P.Value_TvM", pt=0.01,
           fct=2, annot="Gene names", sig_col="purple", bg_col="teal",
           title="Custom Title", figsize=(15,5))
        plt.show()

    Moreover, custom entries can be highlghted such as target proteins of a study.

    >>> idx = prot_limma[prot_limma['logFC_TvM'] > 1].sample(10).index
    >>> vis.volcano(df=prot_limma, log_fc="logFC_TvM", p="P.Value_TvM", highlight=idx, annot="Gene names",
    ...             figsize=(15,5))

    .. plot::
        :context: close-figs

        idx = prot_limma[prot_limma['logFC_TvM'] > 1].sample(10).index
        vis.volcano(df=prot_limma, log_fc="logFC_TvM", p="P.Value_TvM", highlight=idx, annot="Gene names",
                    figsize=(15,5))
        plt.show()

    Using dictionaries of matplotlib keywords eventually allows a higher degree
    of customisation.

    >>> vis.volcano(df=prot_limma, log_fc="logFC_TvM", p="P.Value_TvM", highlight=idx, annot="Gene names",
    ...             figsize=(15,5), highlight_col = "teal", sig_col="lightgray",
    ...             custom_bg = {"s":1, "alpha":.1},
    ...             custom_fg = {"s":5, "alpha":.33},
    ...             custom_hl = {"s":40, "linewidth":1, "edgecolor":"purple"})

    .. plot::
        :context: close-figs

        vis.volcano(df=prot_limma, log_fc="logFC_TvM", p="P.Value_TvM", highlight=idx, annot="Gene names",
           figsize=(15,5), highlight_col = "teal", sig_col="lightgray",
           custom_bg = {"s":1, "alpha":.1},
           custom_fg = {"s":5, "alpha":.33},
           custom_hl = {"s":40, "linewidth":1, "edgecolor":"purple"})
        plt.show()

    You can also collect the volcano plot on an axis and plot multiple plots
    on a single figure.

    >>> fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15,10))
    >>> vis.volcano(df=prot_limma, log_fc="logFC_TvM", p="P.Value_TvM", highlight=idx, annot="Gene names",
    ...             figsize=(15,5), ax=ax[0])
    >>> vis.volcano(df=prot_limma, log_fc="logFC_TvM", p="P.Value_TvM", highlight=idx, annot="Gene names",
    ...             figsize=(15,5), ax=ax[1])
    >>> ax[1].set_ylim(2,4)
    >>> ax[1].set_xlim(0,4)

    .. plot::
        :context: close-figs

        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15,10))
        vis.volcano(df=prot_limma, log_fc="logFC_TvM", p="P.Value_TvM", highlight=idx, annot="Gene names",
                   figsize=(15,5), ax=ax[0])
        vis.volcano(df=prot_limma, log_fc="logFC_TvM", p="P.Value_TvM", highlight=idx, annot="Gene names",
                   figsize=(15,5), ax=ax[1])
        ax[1].set_ylim(2,4)
        ax[1].set_xlim(0,4)
        plt.show()

    If you set the interactive keyword arg to True, you can explore your volcano
    plots interactively using plotly.

    >>> vis.volcano(df=prot_limma, log_fc="logFC_TvM", p="P.Value_TvM", interactive=True, hover_name="Gene names",
                    fct=0.4)

    """

    if custom_hl is None:
        custom_hl = {}
    if custom_fg is None:
        custom_fg = {}
    if custom_bg is None:
        custom_bg = {}

    def setAesthetic(d, typ, interactive):
        """
        Set standard aesthetics of volcano and integrate with user defined settings.

        Parameters
        ----------
        d : dict
            User defined dictionary.
        typ : str
            Whether background 'bg', foreground 'fg' or highlight 'hl' points.
        interactive : bool
            Whether this is an interactive plot.

        Returns
        -------
        d : dict
            The input dict plus standard settings if not specified.

        """
        if typ == "bg":
            standard = {"alpha": 0.33,
                        "s": 2,
                        "label": "background",
                        "linewidth": 0}  # this hugely improves performance in illustrator

        elif typ == "fg":
            standard = {"alpha": 1,
                        "s": 6,
                        "label": "sig",
                        "linewidth": 0}

        elif typ == "hl":
            standard = {"alpha": 1,
                        "s": 20,
                        "label": "POI",
                        "linewidth": 0}

        if not interactive:
            for k in standard.keys():
                if k not in d:
                    d[k] = standard[k]

        return d

    def checkData(df, logFC, score, p, pt, fct):
        if score is None and p is None:
            raise ValueError("You have to provide either a score or a (adjusted) p value.")
        elif score is None:
            df["score"] = -np.log10(df[p])
            score = "score"
        else:
            df.rename(columns={score: "score"}, inplace=True)
            score = "score"
            p = "p"
            df["p"] = 10 ** (df["score"] * -1)

        # define the significant entries in dataframe
        df["SigCat"] = "-"
        if fct is not None:
            df.loc[(df[p] < pt) & (abs(df[logFC]) > fct), "SigCat"] = '*'
        else:
            df.loc[(df[p] < pt), "SigCat"] = '*'
        sig = df[df["SigCat"] == '*'].index
        unsig = df[df["SigCat"] == "-"].index

        return df, score, sig, unsig

    df = df.copy(deep=True)
    # set up standard aesthetics
    custom_bg = setAesthetic(custom_bg, typ="bg", interactive=interactive)
    custom_fg = setAesthetic(custom_fg, typ="fg", interactive=interactive)
    if highlight is not None:
        custom_hl = setAesthetic(custom_hl, typ="hl", interactive=interactive)

    # check for input correctness and make sure score is present in df for plot
    df, score, sig, unsig = checkData(df, logFC, score, p, pt, fct)

    if interactive:
        colors = [bg_col, sig_col]
        if highlight is not None:

            df["SigCat"] = "-"
            df.loc[highlight, "SigCat"] = "*"
            if hover_name is not None:
                fig = px.scatter(data_frame=df, x=logFC, y=score, hover_name=hover_name,
                                 size=pointsize_name,
                                 color="SigCat", color_discrete_sequence=colors,
                                 opacity=0.5, category_orders={"SigCat": ["-", "*"]}, title=title)
            else:
                fig = px.scatter(data_frame=df, x=logFC, y=score,
                                 size=pointsize_name,
                                 color="SigCat", color_discrete_sequence=colors,
                                 opacity=0.5, category_orders={"SigCat": ["-", "*"]}, title=title)

        else:
            if hover_name is not None:
                fig = px.scatter(data_frame=df, x=logFC, y=score, hover_name=hover_name,
                                 size=pointsize_name,
                                 color="SigCat", color_discrete_sequence=colors,
                                 opacity=0.5, category_orders={"SigCat": ["-", "*"]}, title=title)
            else:
                fig = px.scatter(data_frame=df, x=logFC, y=score,
                                 size=pointsize_name,
                                 color="SigCat", color_discrete_sequence=colors,
                                 opacity=0.5, category_orders={"SigCat": ["-", "*"]}, title=title)

        fig.update_yaxes(showgrid=False, zeroline=True)
        fig.update_xaxes(showgrid=False, zeroline=False)

        fig.add_trace(
            go.Scatter(
                x=[df[logFC].min(), df[logFC].max()],
                y=[-np.log10(pt), -np.log10(pt)],
                mode="lines",
                line=go.scatter.Line(color="teal", dash="longdash"),
                showlegend=False)
        )
        if fct is not None:
            # add fold change visualization
            fig.add_trace(
                go.Scatter(
                    x=[-fct, -fct],
                    y=[0, df[score].max()],
                    mode="lines",
                    line=go.scatter.Line(color="teal", dash="longdash"),
                    showlegend=False)
            )
            fig.add_trace(
                go.Scatter(
                    x=[fct, fct],
                    y=[0, df[score].max()],
                    mode="lines",
                    line=go.scatter.Line(color="teal", dash="longdash"),
                    showlegend=False)
            )

        fig.update_layout(template='simple_white',
                          showlegend=legend,
                          )

        if ret_fig:
            return fig

    else:
        # draw figure
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = plt.subplot()  # for a bare minimum plot you do not need this line
        # the following lines of code generate the scatter the rest is styling

        ax.scatter(df[logFC].loc[unsig], df["score"].loc[unsig], color=bg_col, **custom_bg)
        ax.scatter(df[logFC].loc[sig], df["score"].loc[sig], color=sig_col, **custom_fg)
        if highlight is not None:
            ax.scatter(df[logFC].loc[highlight], df["score"].loc[highlight], color=highlight_col, **custom_hl)

        # draw threshold lines
        if fct:
            ax.axvline(fct, 0, 1, ls="dashed", color="lightgray")
            ax.axvline(-fct, 0, 1, ls="dashed", color="lightgray")
        ax.axhline(-np.log10(pt), 0, 1, ls="dashed", color="lightgray")

        # remove of top and right plot boundary
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # setting x and y labels and title
        ax.set_ylabel("score")
        ax.set_xlabel("log_fc")
        ax.set_title(title, size=18)

        # add legend
        if legend:
            ax.legend()

            # Annotation
        if annot is not None:
            # get x and y coordinates as well as strings to plot
            if highlight is None:
                xs = df[logFC].loc[sig]
                ys = df["score"].loc[sig]
                ss = df[annot].loc[sig]
            else:
                if annot_highlight == "all":
                    xs = df[logFC].loc[highlight]
                    ys = df["score"].loc[highlight]
                    ss = df[annot].loc[highlight]
                elif annot_highlight == "sig":
                    xs = df[logFC].loc[set(highlight) & set(sig)]
                    ys = df["score"].loc[set(highlight) & set(sig)]
                    ss = df[annot].loc[set(highlight) & set(sig)]

            # annotation
            for idx, (x, y, s) in enumerate(zip(xs, ys, ss)):
                if idx % 2 != 0:
                    if x < 0:
                        ax.plot([x, x - .2], [y, y + .2], color="gray")
                        ax.text(x - .3, y + .25, s)
                    else:
                        ax.plot([x, x + .2], [y, y + .2], color="gray")
                        ax.text(x + .2, y + .2, s)
                else:
                    if x < 0:
                        ax.plot([x, x - .2], [y, y - .2], color="gray")
                        ax.text(x - .3, y - .25, s)
                    else:
                        ax.plot([x, x + .2], [y, y - .2], color="gray")
                        ax.text(x + .2, y - .2, s)

        if ret_fig:
            return fig


def logIntPlot(df, log_fc, Int, fct=None, annot=False, interactive=False,
               sig_col="green", bg_col="lightgray", title="LogFC Intensity Plot",
               figsize=(6, 6), ret_fig=False):
    r"""
    Draw a log-foldchange vs log-intensity plot.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    log_fc : str
        Colname containing log fold-changes.
    Int : str
        Colname containing the log intensities.
    fct : float, optional
        fold change threshold at which an entry is deemed significant regulated.
        The default is None.
    annot : str, optional
        Which column to use for plot annotation. The default is False.
    interactive : bool, optional
         The default is False.
    sig_col : str, optional
        Colour for significant points. The default is "green".
    bg_col : str, optional
        Background colour. The default is "lightgray".
    title : str, optional
        Title for the plot.
        The default is "Volcano Plot".
    figsize : tuple of int, optional
        Size of the figure. The default is (6,6).
    ret_fig : bool, optional
        Whether or not to return the figure, can be used to further
        customize it afterwards.. The default is False.
    Returns
    -------
    None.

    Examples
    --------
    The log_fc Intensity plot requires the log fold changes as calculated e.g.
    during t-test or LIMMA analysis and (log) intensities to separate points
    on the y axis.

    >>> autoprot.visualization.logIntPlot(prot_limma, "logFC_TvM",
    ...                                   "log10_Intensity BC4_3", fct=0.7, figsize=(15,5))

    .. plot::
        :context: close-figs

        import autoprot.preprocessing as pp
        import autoprot.visualization as vis
        import autoprot.analysis as ana
        import pandas as pd

        prot = pd.read_csv("_static/testdata/proteinGroups.zip", sep='\t', low_memory=False)
        prot = pp.cleaning(prot, "proteinGroups")
        protRatio = prot.filter(regex="^Ratio .\/.( | normalized )B").columns
        prot = pp.log(prot, protRatio, base=2)
        protInt = prot.filter(regex='Intensity').columns
        prot = pp.log(prot, protInt, base=10)
        twitchVsmild = ['log2_Ratio H/M normalized BC18_1','log2_Ratio M/L normalized BC18_2','log2_Ratio H/M normalized BC18_3',
                         'log2_Ratio H/L normalized BC36_1','log2_Ratio H/M normalized BC36_2','log2_Ratio M/L normalized BC36_2']
        prot_limma = ana.limma(prot, twitchVsmild, cond="_TvM")
        prot["log10_Intensity BC4_3"].replace(-np.inf, np.nan, inplace=True)

        vis.logIntPlot(prot_limma, "logFC_TvM", "log10_Intensity BC4_3", fct=0.7, figsize=(15,5))

    Similar to the visualization using a volcano plot, points of interest can be
    selected and labelled.

    >>> autoprot.visualization.logIntPlot(prot_limma, "logFC_TvM", "log10_Intensity BC4_3",
                   fct=2, annot=True, interactive=False, hover_name="Gene names")

    .. plot::
        :context: close-figs

        vis.logIntPlot(prot_limma, "logFC_TvM", "log10_Intensity BC4_3",
                       fct=2, annot=True, interactive=False, hover_name="Gene names")

    And the plots can also be investigated interactively

    >>> autoprot.visualization.logIntPlot(prot_limma, "logFC_TvM",
    ...                                   "log10_Intensity BC4_3", fct=0.7,
    ...                                   figsize=(15,5), interactive=True)
    """
    # TODO: Copy features from volcano function (highlight etc)
    # TODO also add option to not highlight anything
    df = df.copy(deep=True)

    df = df[~df[Int].isin([-np.inf, np.nan])]
    df["SigCat"] = "-"
    if fct is not None:
        df.loc[abs(df[log_fc]) > fct, "SigCat"] = "*"
    unsig = df[df["SigCat"] == "-"].index
    sig = df[df["SigCat"] == "*"].index

    if not interactive:
        # draw figure
        plt.figure(figsize=figsize)
        ax = plt.subplot()
        plt.scatter(df[log_fc].loc[unsig], df[Int].loc[unsig], color=bg_col, alpha=.75, s=5, label="background")
        plt.scatter(df[log_fc].loc[sig], df[Int].loc[sig], color=sig_col, label="POI")

        # draw threshold lines
        if fct:
            plt.axvline(fct, 0, 1, ls="dashed", color="lightgray")
            plt.axvline(-fct, 0, 1, ls="dashed", color="lightgray")
        plt.axvline(0, 0, 1, ls="dashed", color="gray")

        # remove of top and right plot boundary
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # seting x and y labels and title
        plt.ylabel("log Intensity")
        plt.xlabel("logFC")
        plt.title(title, size=18)

        # add legend
        plt.legend()

        if annot:
            # Annotation
            # get x and y coordinates as well as strings to plot
            xs = df[log_fc].loc[sig]
            ys = df[Int].loc[sig]
            ss = df[annot].loc[sig]

            # annotation
            for idx, (x, y, s) in enumerate(zip(xs, ys, ss)):
                if idx % 2 == 0:
                    if x < 0:
                        plt.plot([x, x - .2], [y, y - .2], color="gray")
                        plt.text(x - .3, y - .25, s)
                    else:
                        plt.plot([x, x + .2], [y, y - .2], color="gray")
                        plt.text(x + .2, y - .2, s)

                elif x < 0:
                    plt.plot([x, x - .2], [y, y + .2], color="gray")
                    plt.text(x - .3, y + .25, s)
                else:
                    plt.plot([x, x + .2], [y, y + .2], color="gray")
                    plt.text(x + .2, y + .2, s)
    if interactive:
        if annot:
            fig = px.scatter(data_frame=df, x=log_fc, y=Int, hover_name=annot,
                             color="SigCat", color_discrete_sequence=["cornflowerblue", "mistyrose"],
                             opacity=0.5, category_orders={"SigCat": ["*", "-"]}, title="Volcano plot")
        else:
            fig = px.scatter(data_frame=df, x=log_fc, y=Int,
                             color="SigCat", color_discrete_sequence=["cornflowerblue", "mistyrose"],
                             opacity=0.5, category_orders={"SigCat": ["*", "-"]}, title="Volcano plot")

        fig.update_yaxes(showgrid=False, zeroline=True)
        fig.update_xaxes(showgrid=False, zeroline=False)

        fig.add_trace(
            go.Scatter(
                x=[0, 0],
                y=[0, df[Int].max()],
                mode="lines",
                line=go.scatter.Line(color="purple", dash="longdash"),
                showlegend=False)
        )

        fig.add_trace(
            go.Scatter(
                x=[-fct, -fct],
                y=[0, df[Int].max()],
                mode="lines",
                line=go.scatter.Line(color="teal", dash="longdash"),
                showlegend=False)
        )

        fig.add_trace(
            go.Scatter(
                x=[fct, fct],
                y=[0, df[Int].max()],
                mode="lines",
                line=go.scatter.Line(color="teal", dash="longdash"),
                showlegend=False)
        )

        fig.update_layout({
            'plot_bgcolor': 'rgba(70,70,70,1)',
            'paper_bgcolor': 'rgba(128, 128, 128, 0.25)',
        })

        if ret_fig:
            return fig
        else:
            fig.show()


def MAPlot(df, x, y, interactive=False, fct=None,
           title="MA Plot", figsize=(6, 6), hover_name=None):
    r"""
    Plot log intensity ratios (M) vs. the average intensity (A).

    Notes
    -----
    The MA plot is useful to determine whether a data normalization is needed.
    The majority of proteins is considered to be unchanged between between
    treatments and thereofore should lie on the y=0 line.
    If this is not the case, a normalisation should be applied.

    Parameters
    ----------
    df : pd.dataFrame
        Input dataframe with log intensities.
    x : str
        Colname containing intensities of experiment1.
    y : str
        Colname containing intensities of experiment2.
    interactive : bool, optional
        Whether to return an interactive plotly plot.
        The default is False.
    fct : numeric, optional
        The value in M to draw a horizontal line.
        The default is None.
    title : str, optional
        Title of the figure. The default is "MA Plot".
    figsize : tuple of int, optional
        Size of the figure. The default is (6,6).
    hover_name : str, optional
        Colname to use for labels in interactive plot.
        The default is None.

    Returns
    -------
    None.

    Examples
    --------
    The MA plot allows to easily visualize difference in intensities between
    experiments or replicates and therefore to judge if data normalisation is
    required for further analysis.
    The majority of intensities should be unchanged between conditions and
    therefore most points should lie on the y=0 line.

    >>> autoprot.visualization.MAPlot(prot, twitch, ctrl, fct=2,interactive=False)

    .. plot::
        :context: close-figs

        import autoprot.preprocessing as pp
        import autoprot.visualization as vis
        import autoprot.analysis as ana
        import pandas as pd

        prot = pd.read_csv("_static/testdata/proteinGroups.zip", sep='\t', low_memory=False)
        prot = pp.cleaning(prot, "proteinGroups")
        protInt = prot.filter(regex='Intensity').columns
        prot = pp.log(prot, protInt, base=10)

        x = "log10_Intensity BC4_3"
        y = "log10_Intensity BC36_1"

        vis.MAPlot(prot, x, y, fct=2,interactive=False)
        plt.show()

    If this is not the case, a normalisation using e.g. LOESS should be applied

    >>> autoprot.visualization.MAPlot(prot, twitch, ctrl, fct=2,interactive=False)

    .. plot::
        :context: close-figs

        twitch = "log10_Intensity H BC18_1"
        ctrl = "log10_Intensity L BC18_1"

        vis.MAPlot(prot, twitch, ctrl, fct=2,interactive=False)
        plt.show()
    """
    df = df.copy(deep=True)
    df["M"] = df[x] - df[y]
    df["A"] = 1 / 2 * (df[x] + df[y])
    df["M"].replace(-np.inf, np.nan, inplace=True)
    df["A"].replace(-np.inf, np.nan, inplace=True)
    df["SigCat"] = False
    if fct is not None:
        df.loc[abs(df["M"]) > fct, "SigCat"] = True
    if not interactive:
        # draw figure
        plt.figure(figsize=figsize)
        sns.scatterplot(data=df, x='A', y='M', linewidth=0, hue="SigCat")
        plt.axhline(0, 0, 1, color="black", ls="dashed")
        plt.title(title)
        plt.ylabel("M")
        plt.xlabel("A")

        if fct is not None:
            plt.axhline(fct, 0, 1, color="gray", ls="dashed")
            plt.axhline(-fct, 0, 1, color="gray", ls="dashed")

    else:
        if hover_name is not None:
            fig = px.scatter(data_frame=df, x='A', y='M', hover_name=hover_name,
                             color="SigCat", color_discrete_sequence=["cornflowerblue", "mistyrose"],
                             opacity=0.5, category_orders={"SigCat": ["*", "-"]}, title=title)
        else:
            fig = px.scatter(data_frame=df, x='A', y='M',
                             color="SigCat", color_discrete_sequence=["cornflowerblue", "mistyrose"],
                             opacity=0.5, category_orders={"SigCat": ["*", "-"]}, title=title)

        fig.update_yaxes(showgrid=False, zeroline=True)
        fig.update_xaxes(showgrid=False, zeroline=False)

        if fct is not None:
            fig.add_trace(
                go.Scatter(
                    y=[fct, fct],
                    x=[df['A'].min(), df['A'].max()],
                    mode="lines",
                    line=go.scatter.Line(color="teal", dash="longdash"),
                    showlegend=False)
            )

            fig.add_trace(
                go.Scatter(
                    y=[-fct, -fct],
                    x=[df['A'].min(), df['A'].max()],
                    mode="lines",
                    line=go.scatter.Line(color="teal", dash="longdash"),
                    showlegend=False)
            )

        fig.update_layout({
            'plot_bgcolor': 'rgba(70,70,70,1)',
            'paper_bgcolor': 'rgba(128, 128, 128, 0.25)',
        })
        fig.show()


def meanSd(df, reps):
    r"""
    Rank vs. standard deviation plot.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    reps : list of str
        Column names over which to calculate standard deviations and rank.

    Returns
    -------
    None.

    Examples
    --------
    Visualise the intensity distirbutions of proteins depending on their
    total indensity.

    >>> autoprot.visualization.meanSd(prot, twitchInt)

    .. plot::
        :context: close-figs

        import autoprot.preprocessing as pp
        import autoprot.visualization as vis
        import autoprot.analysis as ana
        import pandas as pd

        prot = pd.read_csv("_static/testdata/proteinGroups.zip", sep='\t', low_memory=False)
        prot = pp.cleaning(prot, "proteinGroups")
        protInt = prot.filter(regex='Intensity').columns
        prot = pp.log(prot, protInt, base=10)

        twitchInt = ['log10_Intensity H BC18_1','log10_Intensity M BC18_2','log10_Intensity H BC18_3',
                 'log10_Intensity BC36_1','log10_Intensity H BC36_2','log10_Intensity M BC36_2']

        vis.meanSd(prot, twitchInt)
    """

    def hexa(x, y):
        plt.hexbin(x, y, cmap="BuPu",
                   gridsize=40)
        plt.plot(x, y.rolling(window=200, min_periods=10).mean(), color="teal")
        plt.xlabel("rank (mean)")

    df = df.copy(deep=True)
    df["mean"] = abs(df[reps].mean(1))
    df["sd"] = df[reps].std(1)
    df = df.sort_values(by="mean")

    p = sns.JointGrid(
        x=range(df.shape[0]),
        y=df['sd']
    )

    p = p.plot_joint(
        hexa
    )

    p.ax_marg_y.hist(
        df['sd'],
        orientation='horizontal',
        alpha=0.5,
        bins=50
    )

    p.ax_marg_x.get_xaxis().set_visible(False)
    p.ax_marg_x.set_title("Mean SD plot", fontsize=18)


def plotTraces(df, cols, labels=None, colors=None, z_score=None,
               xlabel="", ylabel="log_fc", title="", ax=None,
               plot_summary=False, plot_summary_only=False, summary_color="red",
               summary_type="Mean", summary_style="solid", **kwargs):
    r"""
    Plot numerical data such as fold changes vs. columns (e.g. conditions).

    Parameters
    ----------
    df : pd.DataFame
        Input dataframe.
    cols : list of str
        The colnames from which the values are plotted.
    labels : list of str, optional
        Corresponds to data, used to label traces.
        The default is None.
    colors : list of colours, optional
        Colours to labels the traces.
        Must be the same length as the values in cols.
        The default is None.
    z_score : int, optional
        Whether to apply zscore transformation.
        Must be between 0 and 1 for True.
        The default is None.
    xlabel : str, optional
        Label for the x axis. The default is "".
    ylabel : str, optional
        Label for the y axis.
        The default is "log_fc".
    title : str, optional
        Title of the plot.
        The default is "".
    ax : matplotlib axis, optional
        Axis to plot on.
        The default is None.
    plot_summary : bool, optional
        Whether to plot a line corresponding to a summary of the traces as defined
        by summary_type.
        The default is False.
    plot_summary_only : bool, optional
        Whether to plot only the summary.
        The default is False.
    summary_color : Colour-like, optional
        The colour for the summary.
        The default is "red".
    summary_type : str, optional
        "Mean" or "Median". The default is "Mean".
    summary_style : matplotlib.linestyle, optional
        Style for the summary trace as defined in
        https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html.
        The default is "solid".
    **kwargs :
        passed to matplotlib.pyplot.plot.

    Returns
    -------
    None.

    Examples
    --------
    Plot the log fold-changes of 10 phosphosites during three comparisons.

    >>> idx = phos.sample(10).index
    >>> test = phos.filter(regex="logFC_").loc[idx]
    >>> label = phos.loc[idx, "Gene names"]
    >>> vis.plotTraces(test, test.columns, labels=label, colors=["red", "green"]*5,
    ...                xlabel='Column', z_score=None)

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
        phos = ana.ttest(df=phos_expanded, reps=twitchVsmild, cond="MvC", mean=True)

        idx = phos.sample(10).index
        test = phos.filter(regex="logFC_").loc[idx]
        label = phos.loc[idx, "Gene names"]
        vis.plotTraces(test, test.columns, labels=label, colors=["red", "green"]*5,
                       xlabel='Column')
        plt.show()

    """
    # TODO Add parameter to plot yerr
    # TODO xlabels from colnames
    x = range(len(cols))
    y = df[cols].T.values
    if z_score is not None and z_score in [0, 1]:
        y = zscore(y, axis=z_score)

    if ax is None:
        plt.figure()
        ax = plt.subplot()
    ax.set_title(title)
    if not plot_summary_only:
        if colors is None:
            f = ax.plot(x, y, **kwargs)
        else:
            f = []
            for i, yi in enumerate(y.T):
                f += ax.plot(x, yi, color=colors[i], **kwargs)
    if plot_summary or plot_summary_only:
        if summary_type == "Mean":
            f = ax.plot(x, np.mean(y, 1), color=summary_color,
                        lw=3, linestyle=summary_style, **kwargs)
        elif summary_type == "Median":
            f = ax.plot(x, np.median(y, 1), color=summary_color,
                        lw=3, linestyle=summary_style, **kwargs)

    if labels is not None:
        for s, line in zip(labels, f):
            # get last point for annotation
            ly = line.get_data()[1][-1]
            lx = line.get_data()[0][-1] + 0.1
            plt.text(lx, ly, s)

    sns.despine(ax=ax)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)


def sequenceLogo(df, motif, file=None, rename_to_st=False):
    r"""
    Generate sequence logo plot based on experimentally observed phosphosites.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe from which experimentally determined sequences are extracted.
    motif : tuple of str
        A tuple of the sequence_motif and its name.
        The phosphosite residue in the sequence_motif should be indicated by a
        lowercase character.
        Example ("..R.R..s.......", "MK_down").
    file : str
        Path to write the figure to outfile_path.
        Default is None.
    rename_to_st : bool, optional
        If true, the phoshoresidue will be considered to be
        either S or T. The default is False.

    Raises
    ------
    ValueError
        If the phosphoresidue was not indicated by lowercase character.

    Returns
    -------
    None.

    Examples
    --------
    First define the sequence_motif of interest. Note that the phosphorylated residue
    should be marked by a lowercase character.

    >>> sequence_motif = ("..R.R..s.......", "MK_down")
    >>> autoprot.visualization.sequenceLogo(phos, sequence_motif)

    allow s and t as central residue

    >>> autoprot.visualization.sequenceLogo(phos, sequence_motif, path, rename_to_st=True)

    """

    # TODO: sequence_motif and name should be provided in 2 parameter

    def generate_sequence_logo(seq: list, outfile_path: str = None, sequence_motif: str = ""):
        """
        Draw a sequence logo plot for a sequence_motif.

        Parameters
        ----------
        seq : list of str
            List of experimentally determined sequences matching the sequence_motif.
        outfile_path : str
            path to folder where the output file will be written.
            Default is None.
        sequence_motif : str, optional
            The sequence_motif used to find the sequences.
            The default is "".

        Returns
        -------
        None.
        """
        aa_dic = dict(G=0, P=0, A=0, V=0, L=0, I=0, M=0, C=0, F=0, Y=0, W=0, H=0, K=0, R=0, Q=0, N=0, E=0, D=0, S=0,
                      T=0)

        seq = [i for i in seq if len(i) == 15]
        seqT = [''.join(s) for s in zip(*seq)]
        scoreMatrix = []
        for pos in seqT:
            d = aa_dic.copy()
            for aa in pos:
                aa = aa.upper()
                if aa not in ['.', '-', '_', "X"]:
                    d[aa] += 1
            scoreMatrix.append(d)

        for pos in scoreMatrix:
            for k in pos.keys():
                pos[k] /= len(seq)

        # empty array -> (sequenceWindow, aa)
        m = np.empty((15, 20))
        for i in range(m.shape[0]):
            x = list(scoreMatrix[i].values())
            m[i] = x

        # create Logo object
        kinase_motif_df = pd.DataFrame(m).fillna(0)
        kinase_motif_df.columns = aa_dic.keys()
        k_logo = logomaker.Logo(kinase_motif_df,
                                font_name="Arial",
                                color_scheme="dmslogo_funcgroup",
                                vpad=0,
                                width=.8)

        k_logo.highlight_position(p=7, color='purple', alpha=.5)
        plt.title(f"{sequence_motif} SequenceLogo")
        k_logo.ax.set_xticklabels(labels=[-7, -7, -5, -3, -1, 1, 3, 5, 7])
        sns.despine()
        if outfile_path is not None:
            plt.savefig(outfile_path)

    def find_motif(x: pd.DataFrame, sequence_motif: str, typ: str, rename_to_st=False):
        """
        Return the input sequence_motif if it fits to the value provided in "Sequence window" of a dataframe row.

        Parameters
        ----------
        x : pd.DataFrame
            Dataframe containing the identified sequence windows.
        sequence_motif : str
            The kinase sequence_motif.
        typ : str
            The kinase sequence_motif.
        rename_to_st : bool, optional
            Look for S and T at the phosphorylation position.
            The phoshorylated residue should be S or T, otherwise it is transformed
            to S/T.
            The default is False.

        Raises
        ------
        ValueError
            If not lowercase phospho residue is given.

        Returns
        -------
        typ : str
            The kinase sequence_motif.

        """
        global pos1
        import re
        # identified sequence window
        d = x["Sequence window"]
        # In Sequence window the aa of interest is always at pos 15
        # This loop will check if the sequence_motif we are interested in is
        # centered with its phospho residue at pos 15 of the sequence window
        check_lower = False
        for j, i in enumerate(sequence_motif):
            # the phospho residue in the sequence_motif is indicated by lowercase character
            if i.islower():
                # pos1 is position of the phosphosite in the sequence_motif
                pos1 = len(sequence_motif) - j
                check_lower = True
        if not check_lower:
            raise ValueError("Phosphoresidue has to be lower case!")
        if rename_to_st:
            # insert the expression (S/T) on the position of the phosphosite
            exp = sequence_motif[:pos1 - 1] + "(S|T)" + sequence_motif[pos1:]
        else:
            # for finding pos2, the whole sequence_motif is uppercase
            exp = sequence_motif.upper()

        # pos2 is the last position of the matched sequence
        # the MQ Sequence window is always 30 AAs long and centred on the modified
        # amino acid. Hence, for a true hit, pos2-pos1 should be 15
        if pos2 := re.search(exp.upper(), d):
            pos2 = pos2.end()
            pos = pos2 - pos1
            if pos == 15:
                return typ

    # init empty col corresponding to sequence sequence_motif
    df[motif[0]] = np.nan
    # returns the input sequence sequence_motif for rows where the sequence_motif fits the sequence
    # window
    df[motif[0]] = df.apply(lambda x: find_motif(x, motif[0], motif[0], rename_to_st), 1)

    if file is not None:
        # consider only the +- 7 amino acids around the modified residue (x[8:23])
        generate_sequence_logo(df["Sequence window"][df[motif[0]].notnull()].apply(lambda x: x[8:23]),
                               outfile_path=file + "/{}_{}.svg".format(motif[0], motif[1]),
                               sequence_motif="{} - {}".format(motif[0], motif[1]))
    else:
        generate_sequence_logo(df["Sequence window"][df[motif[0]].notnull()].apply(lambda x: x[8:23]),
                               sequence_motif="{} - {}".format(motif[0], motif[1]))


def visPs(name, length, domain_position=None, ps=None, pl=None, plc=None, pls=4, ax=None, domain_color='tab10'):
    """
    Visualize domains and phosphosites on a protein of interest.

    Parameters
    ----------
    name : str
        Name of the protein.
        Used for plot title.
    length : int
        Length of the protein.
    domain_position : list of tuples of int
        Each element is a tuple of domain start and end postiions.
    ps : list of int
        position of phosphosites.
    pl : list of str
        label for ps (has to be in same order as ps).
    plc : list of colours
        optionally one can provide a list of colors for the phosphosite labels.
    pls : int, optional
        Fontsize for the phosphosite labels. The default is 4.
    ax: matplotlib axis, optional
        To draw on an existing axis
    domain_color: str
        Either a matplotlib colormap (see https://predictablynoisy.com/matplotlib/gallery/color/colormap_reference.html)
        or a single color

    Returns
    -------
    matplotlib.figure
        The figure object.

    Examples
    --------
    Draw an overview on the phosphorylation of AKT1S1.

    >>> name = "AKT1S1"
    >>> length = 256
    >>> domain_position = [35,43,
    ...                    77,96]
    >>> ps = [88, 92, 116, 183, 202, 203, 211, 212, 246]
    >>> pl = ["pS88", "pS92", "pS116", "pS183", "pS202", "pS203", "pS211", "pS212", "pS246"]

    colors (A,B,C,D (gray -> purple), Ad, Bd, Cd, Dd (gray -> teal) can be used to indicate regulation)

    >>> plc = ['C', 'A', 'A', 'C', 'Cd', 'D', 'D', 'B', 'D']
    >>> autoprot.visualization.visPs(name, length, domain_position, ps, pl, plc, pls=12)

    .. plot::
        :context: close-figs

        import autoprot.visualization as vis

        name = "AKT1S1"
        length = 256
        domain_position = [(35,43),
                           (77,96)]
        ps = [88, 92, 116, 183, 202, 203, 211, 212, 246]
        pl = ["pS88", "pS92", "pS116", "pS183", "pS202", "pS203", "pS211", "pS212", "pS246"]
        plc = ['C', 'A', 'A', 'C', 'Cd', 'D', 'D', 'B', 'D']
        vis.visPs(name, length, domain_position, ps, pl, plc, pls=12)
        plt.show()

    """
    if domain_position is None:
        domain_position = []
    # check if domain_color is a cmap name
    try:
        cm = plt.get_cmap(domain_color)
        color = cm(np.linspace(0, 1, len(domain_position)))
    except ValueError as e:
        # it is not, so is it a single colour?
        if isinstance(domain_color, str):
            color = [domain_color, ] * len(domain_position)
        elif isinstance(domain_color, list):
            if len(domain_color) != len(domain_position):
                raise Exception("Please provide one domain colour per domain") from e
            else:
                color = domain_color
        else:
            raise Exception("You must provide a colormap name, a colour name or a list of colour names") from e

    lims = (1, length)
    height = lims[1] / 25

    if ax is None:
        fig1 = plt.figure(figsize=(15, 2))
        ax1 = fig1.add_subplot(111, aspect='equal')
    else:
        ax1 = ax

    # background of the whole protein in grey
    ax1.add_patch(
        patches.Rectangle((0, 0), length, height, color='lightgrey'))

    for idx, (start, end) in enumerate(domain_position):
        width = end - start
        ax1.add_patch(
            patches.Rectangle((start, 0), width, height, color=color[idx]))

    # only plot phosphosite if there are any
    if ps is not None:
        textColor = {"A": "gray",
                     "Ad": "gray",
                     "B": "#dc86fa",
                     "Bd": "#6AC9BE",
                     "C": "#aa00d7",
                     "Cd": "#239895",
                     "D": "#770087",
                     "Dd": "#008080"}

        for idx, site in enumerate(ps):
            plt.axvline(site, 0, 1, color="red")
            plt.text(site - 1,
                     height - (height + height * 0.15),
                     pl[idx] if pl != None else '',
                     fontsize=pls,
                     rotation=90,
                     color=textColor[plc[idx]] if plc != None else 'black')

    plt.subplots_adjust(left=0.25)
    plt.ylim(height)
    plt.xlim(lims)
    ax1.axes.get_yaxis().set_visible(False)
    plt.title(name + '\n', size=18)
    plt.tight_layout()


def styCountPlot(df, figsize=(12, 8), typ="bar", ret_fig=False):
    r"""
    Draw an overview of Number of Phospho (STY) of a Phospho(STY) file.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
        Must contain a column "Number of Phospho (STY)".
    figsize : tuple of int, optional
        Figure size. The default is (12,8).
    typ : str, optional
        'bar' or 'pie'. The default is "bar".
    ret_fig : bool, optional
        Whether to return the figure. The default is False.

    Returns
    -------
    fig : matplotlib.figure
        The figure object.

    Examples
    --------
    Plot a bar chart of the distribution of the number of phosphosites on the peptides.

    >>> autoprot.visualization.styCountPlot(phos, typ="bar")
    Number of phospho (STY) [total] - (count / # Phospho)
    [(29, 0), (37276, 1), (16460, 2), (4276, 3), (530, 4), (52, 5)]
    Percentage of phospho (STY) [total] - (% / # Phospho)
    [(0.05, 0), (63.59, 1), (28.08, 2), (7.29, 3), (0.9, 4), (0.09, 5)]

    .. plot::
        :context: close-figs

        import autoprot.preprocessing as pp
        import autoprot.visualization as vis
        import pandas as pd

        phos = pd.read_csv("_static/testdata/Phospho (STY)Sites_mod.zip", sep="\t", low_memory=False)
        phos = pp.cleaning(phos, file = "Phospho (STY)")
        vis.styCountPlot(phos, typ="bar")
        plt.show()

    """
    noOfPhos = [int(i) for i in list(pl.flatten([str(i).split(';') for i in df["Number of Phospho (STY)"].fillna(0)]))]
    count = [(noOfPhos.count(i), i) for i in set(noOfPhos)]
    counts_perc = [(round(noOfPhos.count(i) / len(noOfPhos) * 100, 2), i) for i in set(noOfPhos)]

    print("Number of phospho (STY) [total] - (count / # Phospho)")
    print(count)
    print("Percentage of phospho (STY) [total] - (% / # Phospho)")
    print(counts_perc)
    df = pd.DataFrame(noOfPhos, columns=["Number of Phospho (STY)"])

    if typ == "bar":
        fig = plt.figure(figsize=figsize)
        ax = sns.countplot(x="Number of Phospho (STY)", data=df)
        plt.title('Number of Phospho (STY)')
        plt.xlabel('Number of Phospho (STY)')
        ncount = df.shape[0]

        # Make twin axis
        ax2 = ax.twinx()

        ax2.yaxis.tick_left()
        ax.yaxis.tick_right()

        ax.yaxis.set_label_position('right')
        ax2.yaxis.set_label_position('left')

        ax2.set_ylabel('Frequency [%]')

        for p in ax.patches:
            x = p.get_bbox().get_points()[:, 0]
            y = p.get_bbox().get_points()[1, 1]
            ax.annotate('{:.1f}%'.format(100. * y / ncount), (x.mean(), y),
                        ha='center', va='bottom')  # set the alignment of the text

        ax.yaxis.set_major_locator(ticker.LinearLocator(11))
        ax2.set_ylim(0, 100)
        ax.set_ylim(0, ncount)
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))
    elif typ == "pie":
        fig = plt.figure(figsize=figsize)
        plt.pie([i[0] for i in count], labels=[i[1] for i in count])
        plt.title("Number of Phosphosites")
    if ret_fig is True:
        return fig


# noinspection PyUnboundLocalVariable
def chargePlot(df, figsize=(12, 8), typ="bar", ret_fig=False, ax=None):
    r"""
    Plot a pie chart of the peptide charges of a phospho(STY) dataframe.

    Parameters
    ----------
    df : pd.Dataframe
        Input dataframe.
        Must contain a column named "Charge".
    figsize : tuple of int, optional
        The size of the figure. The default is (12,8).
    typ : str, optional
        "pie" or "bar".
        The default is "bar".
    ret_fig : bool, optional
        Whether to return the figure.
        The default is False.
    ax : matplotlib axis
        Axis to plot on

    Returns
    -------
    fig : matplotlib.figure
        The figure object.

    Examples
    --------
    Plot the charge states of a dataframe.

    >>> autoprot.visualization.chargePlot(phos, typ="pie")
    charge [total] - (count / # charge)
    [(44, 1), (20583, 2), (17212, 3), (2170, 4), (61, 5), (4, 6)]
    Percentage of charge [total] - (% / # charge)
    [(0.11, 1), (51.36, 2), (42.95, 3), (5.41, 4), (0.15, 5), (0.01, 6)]
    charge [total] - (count / # charge)
    [(44, 1), (20583, 2), (17212, 3), (2170, 4), (61, 5), (4, 6)]
    Percentage of charge [total] - (% / # charge)
    [(0.11, 1), (51.36, 2), (42.95, 3), (5.41, 4), (0.15, 5), (0.01, 6)]

    .. plot::
        :context: close-figs

        import autoprot.preprocessing as pp
        import autoprot.visualization as vis
        import pandas as pd

        phos = pd.read_csv("_static/testdata/Phospho (STY)Sites_mod.zip", sep="\t", low_memory=False)
        phos = pp.cleaning(phos, file = "Phospho (STY)")
        vis.chargePlot(phos, typ="pie")
        plt.show()
    """

    df = df.copy(deep=True)
    noOfPhos = [int(i) for i in list(pl.flatten([str(i).split(';') for i in df["Charge"].fillna(0)]))]
    count = [(noOfPhos.count(i), i) for i in set(noOfPhos)]
    counts_perc = [(round(noOfPhos.count(i) / len(noOfPhos) * 100, 2), i) for i in set(noOfPhos)]

    print("charge [total] - (count / # charge)")
    print(count)
    print("Percentage of charge [total] - (% / # charge)")
    print(counts_perc)
    df = pd.DataFrame(noOfPhos, columns=["charge"])

    if typ == "bar":
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.gca()

        sns.countplot(x="charge", data=df, ax=ax)
        plt.title('charge')
        plt.xlabel('charge')
        ncount = df.shape[0]

        # Make twin axis
        ax2 = ax.twinx()

        ax2.yaxis.tick_left()
        ax.yaxis.tick_right()

        ax.yaxis.set_label_position('right')
        ax2.yaxis.set_label_position('left')

        ax2.set_ylabel('Frequency [%]')

        for p in ax.patches:
            x = p.get_bbox().get_points()[:, 0]
            y = p.get_bbox().get_points()[1, 1]
            ax.annotate('{:.1f}%'.format(100. * y / ncount), (x.mean(), y),
                        ha='center', va='bottom')  # set the alignment of the text

        ax.yaxis.set_major_locator(ticker.LinearLocator(11))
        ax2.set_ylim(0, 100)
        ax.set_ylim(0, ncount)
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))
    elif typ == "pie":
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.gca()
        ax.pie([i[0] for i in count], labels=[i[1] for i in count])
        ax.set_title("charge")
    if not ret_fig:
        return
    return fig


def modAa(df, figsize=(6, 6), ret_fig=False):
    r"""
    Count the number of modifications per amino acid.

    Parameters
    ----------
    df : pd.Dataframe
        The input dataframe.
        Must contain a column "Amino acid".
    figsize : tuple of int, optional
        The size of the figure. The default is (6,6).
    ret_fig : bool, optional
        Whether to return the figure object. The default is False.

    Returns
    -------
    fig : matplotlib.figure
        The figure object.

    Examples
    --------
    Plot pie chart of modified amino acids.

    >>> autoprot.visualization.modAa(phos)

    .. plot::
        :context: close-figs

        import autoprot.preprocessing as pp
        import autoprot.visualization as vis
        import pandas as pd

        phos = pd.read_csv("_static/testdata/Phospho (STY)Sites_mod.zip", sep="\t", low_memory=False)
        phos = pp.cleaning(phos, file = "Phospho (STY)")
        vis.modAa(phos)
        plt.show()

    """
    labels = [str(i) + '\n' + str(round(j / df.shape[0] * 100, 2)) + '%'
              for i, j in zip(df["Amino acid"].value_counts().index,
                              df["Amino acid"].value_counts().values)]

    fig = plt.figure(figsize=figsize)
    plt.pie(df["Amino acid"].value_counts().values,
            labels=(labels))
    plt.title("Modified AAs")
    if ret_fig == True:
        return fig


def wordcloud(text, pdffile=None, exlusionwords=None, background_color="white", mask=None, file="",
              contour_width=0, **kwargs):
    """
    Generate Wordcloud from string.

    Parameters
    ----------
    text : str
        text input as a string.
    exlusionwords : list of str, optional
        list of words to exclude from wordcloud. The default is None.
    background_color : colour, optional
        The background colour of the plot. The default is "white".
    mask : 'round' or path to png file, optional
        Used to mask the wordcloud.
        set it either to round or true and add a .png file.
        The default is None.
    file : str, optional
        file is given as path with path/to/file.filetype.
        The default is "".
    contour_width : int, optional
        If mask is not None and contour_width > 0, draw the mask contour.
        The default is 0.
    **kwargs :
        passed to wordcloud.WordCloud.

    Returns
    -------
    None.

    Examples
    --------
    Plot Hello World

    >>> autoprot.visualization.wordcloud(text="hello world!", contour_width=5, mask='round')

    .. plot::
        :context: close-figs

        import autoprot.visualization as vis
        vis.wordcloud(text="hello world!", contour_width=5, mask='round')
        plt.show()

    You can also use the extractPDF method to input a pdf text instead of a text
    string

    >>> text = autoprot.visualization.wordcloud.extractPDF('/path/to/pdf')
    >>> autoprot.visualization.wordcloud(text="hello world!", contour_width=5, mask='round')
    """

    def extractPDF(file):
        """
        Extract text from PDF file.

        Parameters
        ----------
        file : str
            Path to pdf file.
        ----------
        """
        resource_manager = PDFResourceManager()
        fake_file_handle = io.StringIO()
        converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
        page_interpreter = PDFPageInterpreter(resource_manager, converter)

        with open(file, 'rb') as fh:
            # 'rb' is opening the pdf file in binary mode

            for page in PDFPage.get_pages(fh,
                                          caching=True,
                                          check_extractable=True):
                page_interpreter.process_page(page)

            text = fake_file_handle.getvalue()
        # close open handles
        converter.close()
        fake_file_handle.close()
        return text

    if exlusionwords is not None:
        exlusionwords = exlusionwords + list(STOPWORDS)

    if mask is not None:
        if mask.split('.')[-1] == "png":
            mask = np.array(Image.open(mask))
        elif mask == "round":
            x, y = np.ogrid[:1000, :1000]
            mask = (x - 500) ** 2 + (y - 500) ** 2 > 400 ** 2
            mask = 255 * mask.astype(int)
        wc = WordCloud(background_color=background_color, mask=mask, contour_width=contour_width,
                       stopwords=exlusionwords, **kwargs).generate(text)
    else:
        wc = WordCloud(background_color="white", stopwords=exlusionwords, width=1800, height=500).generate(text)

    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()

    return wc


def BHplot(df, ps, adj_ps, title=None, alpha=0.05, zoom=20):
    r"""
    Visualize Benjamini Hochberg p-value correction.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with p values.
    ps : str
        Colname of column with p-values.
    adj_ps : str
        column with adj_p values.
    title : str, optional
        Plot title. The default is None.
    alpha : flaot, optional
        The significance level drawn in the plot. The default is 0.05.
    zoom : int, optional
        Zoom on the first n points. The default is 20.

    Returns
    -------
    None.

    Examples
    --------
    The function generates two plots, left with all datapoints sorted by p-value
    and right with a zoom on the first 6 values (zoom=7).
    The grey line indicates the provided alpha level. Values below it are considered
    significantly different.

    >>> autoprot.visualization.BHplot(phos,'pValue_TvC', 'adj.pValue_TvC', alpha=0.05, zoom=7)

    .. plot::
        :context: close-figs

        import autoprot.preprocessing as pp
        import autoprot.analysis as ana
        import autoprot.visualization as vis
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

        mildVsctrl = ["log2_Ratio M/L normalized R1","log2_Ratio H/L normalized R2","log2_Ratio M/L normalized R3",
                      "log2_Ratio H/M normalized R4","log2_Ratio M/L normalized R5","log2_Ratio H/L normalized R6"]

        phos = ana.ttest(df=phos_expanded, reps=mildVsctrl, cond="MvC", mean=True)

        vis.BHplot(phos,'pValue_MvC', 'adj.pValue_MvC', alpha=0.05, zoom=7)
    """
    n = len(df[ps][df[ps].notnull()])
    x = range(n)
    y = [((i + 1) * alpha) / n for i in x]

    idx = df[ps][df[ps].notnull()].sort_values().index

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ax[0].set_title(title)
    ax[0].plot(x, y, color='gray', label=r'$\frac{i * \alpha}{n}$')
    ax[0].scatter(x, df[ps].loc[idx].sort_values(), label="p_values", color="teal", alpha=0.5)
    ax[0].scatter(x, df[adj_ps].loc[idx], label="adj. p_values", color="purple", alpha=0.5)
    ax[0].legend(fontsize=12)

    ax[1].plot(x[:zoom], y[:zoom], color='gray')
    ax[1].scatter(x[:zoom], df[ps].loc[idx].sort_values().iloc[:zoom], label="p_values", color="teal")
    ax[1].scatter(x[:zoom], df[adj_ps].loc[idx][:zoom], label="adj. p_values", color="purple")

    sns.despine(ax=ax[0])
    sns.despine(ax=ax[1])
