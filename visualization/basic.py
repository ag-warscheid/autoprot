# -*- coding: utf-8 -*-
"""
Autoprot Basic Plotting Functions.

@author: Wignand, Julian, Johannes

@documentation: Julian
"""
from functools import reduce

from scipy import stats
from scipy.stats import zscore, gaussian_kde
from scipy.linalg import LinAlgError
import pandas as pd
# noinspection PyProtectedMember
from pandas.api.types import is_numeric_dtype
import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import matplotlib

from matplotlib_venn import venn2
from matplotlib_venn import venn3
from adjustText import adjust_text
import matplotlib.patches as patches
from itertools import combinations

from ..dependencies.venn import venn
from .. import visualization as vis
from .. import common as com

import plotly.express as px
import plotly.graph_objects as go

import upsetplot

from typing import Literal, Union, List

from matplotlib import pyplot as plt
import seaborn as sns


def correlogram(df, columns=None, file="proteinGroups", log=True, save_dir=None,
                save_type="pdf", save_name="pairPlot", lower_triang="scatter",
                sample_frac=None, bins=100, ret_fig=False, correlation_colorrange: tuple[float] = (0.8, 1),
                figsize: Union[bool, tuple] = None):
    # noinspection PyUnresolvedReferences
    r"""Plot a pair plot of the dataframe intensity columns in order to assess the reproducibility.

    Notes
    -----
    The lower half of the correlogram shows a scatter plot comparing pairs of
    conditions while the upper part shows you the color coded correlation
    coefficients as well as the intersection of hits between both conditions.
    In tiles corresponding to self-comparison (the same value on y and x-axis)
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
    correlation_colorrange : tuple, optional
        Sets the colormap range for the upper-right correlation tiles. Default is (0.8, 1)
    figsize : tuple, optional
        The figure size in x and y direction.

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

    # TODO: Add functionality of embedding all the plots as subplots in figures by providing ax parameter

    if columns is None:
        columns = []
    if columns is None:
        columns = []

    def calculate_correlation(a: np.array, b: np.array) -> tuple[float, pd.DataFrame]:
        """
        Calculate the correlation between two lists of values.

        Returns
        -------
        r (float): Pearson correlation coefficient
        d (pd.DataFrame): The dataframe with the matched values
        """
        d = pd.DataFrame({"x": a, "y": b})
        d = d.dropna(how='any')
        a = d["x"].values
        b = d["y"].values
        r, _ = stats.pearsonr(a, b)
        return r, d

    # noinspection PyShadowingNames
    def corrfunc(x, y, **kwargs):
        """Function for seaborn PairGrid to calculate correlation coefficient and add text to axis."""
        _ = kwargs  # kwargs required to catch seaborn calling kwargs color and label
        r, _ = calculate_correlation(x, y)
        ax = plt.gca()
        ax.annotate("r = {:.2f}".format(r),
                    xy=(.1, .9), xycoords=ax.transAxes)

    # noinspection PyShadowingNames
    def heatmap(x, y, color=None, label=None):
        """Calculate correlation coefficient and add coloured tile to axis."""
        df = pd.DataFrame({"x": x, "y": y})
        df = df.replace(-np.inf, np.nan).dropna()
        x = df["x"].values
        y = df["y"].values
        r, _ = stats.pearsonr(x, y)
        ax = plt.gca()

        # normalize the values so that the lowest value of the cmap is reach at R=0.8
        norm = matplotlib.colors.Normalize(vmin=correlation_colorrange[0], vmax=correlation_colorrange[1])
        if (color is None) or (color not in plt.colormaps()):
            cmap = matplotlib.cm.get_cmap('Blues')
        else:
            cmap = matplotlib.cm.get_cmap(color)
        ax.add_patch(mpl.patches.Rectangle((0, 0), 5, 5,
                                           color=cmap(norm(r)),
                                           transform=ax.transAxes,
                                           label=label))
        ax.tick_params(axis="both", which="both", length=0)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    # noinspection PyShadowingNames
    def lower_scatter(x, y, color=None, label=None):
        """Plot data points as scatter plot to axis."""
        data = pd.DataFrame({"x": x, "y": y})
        if sample_frac is not None:
            data = data.sample(int(data.shape[0] * sample_frac))
        ax = plt.gca()
        ax.scatter(data['x'], data['y'], linewidth=0,
                   color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0] if color is None else color,
                   label=label)

    # noinspection PyShadowingNames
    def lower_hex_bin(x, y, color=None, label=None):
        """Plot data points as hexBin plot to axis."""
        # only fall back to default if no colormap is given or the given string is no colormap
        if (color is None) or (color not in plt.colormaps()):
            plt.hexbin(x, y, cmap="Blues", bins=bins, gridsize=50, label=label)
        else:
            plt.hexbin(x, y, cmap=color, bins=bins, gridsize=50, label=label)

    # noinspection PyShadowingNames
    def lower_hist_2d(x, y, color=None, label=None):
        """Plot data points as hist2d plot to axis."""
        df = pd.DataFrame({"x": x, "y": y})
        df = df.dropna()
        x = df["x"].values
        y = df["y"].values

        # only fall back to default if no colormap is given or the given string is no colormap
        if (color is None) or (color not in plt.colormaps()):
            plt.hist2d(x, y, bins=bins, cmap="Blues", vmin=0, vmax=1, label=label)
        else:
            plt.hist2d(x, y, bins=bins, cmap=color, vmin=0, vmax=1, label=label)

    # noinspection PyShadowingNames
    def proteins_found(x, y, **kwargs):
        _ = kwargs  # kwargs required to catch seaborn calling kwargs color and label
        r, d = calculate_correlation(x, y)
        ax = plt.gca()

        if file == "Phospho (STY)":
            to_count = 'peptides'
        else:
            to_count = 'proteins'

        ax.annotate(f"{len(d)} {to_count}", xy=(0.1, 0.9), xycoords=ax.transAxes)
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
    if figsize is not None:
        g.fig.set_size_inches(*figsize)
    # accesses the lower triangle
    g.map_lower(corrfunc)
    # plot the data points on the lower triangle
    if lower_triang == "scatter":
        g.map_lower(lower_scatter)
    elif lower_triang == "hexBin":
        g.map_lower(lower_hex_bin)
    elif lower_triang == "hist2d":
        g.map_lower(lower_hist_2d)
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


def corr_map(df, columns, cluster=False, annot=None, cmap="YlGn", figsize=(7, 7),
             save_dir=None, save_type="pdf", save_name="pairPlot", ax=None, **kwargs):
    # noinspection PyUnresolvedReferences
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
    save_dir : str, optional
        Where the plots are saved. The default is None.
    save_type : str, optional
        What format the saved plots have (pdf, png). The default is "pdf".
    save_name : str, optional
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

    >>> autoprot.visualization.corr_map(prot,mildLogInt, annot=True)

    .. plot::
        :context: close-figs

        import autoprot.preprocessing as pp
        import autoprot.visualization as vis

        prot = pd.read_csv("_static/testdata/proteinGroups.zip", sep='\t', low_memory=False)
        mildInt = ["Intensity M BC18_1","Intensity H BC18_2","Intensity M BC18_3",
                   "Intensity M BC36_1","Intensity M BC36_2","Intensity H BC36_2"]
        prot = pp.log(prot, mildInt, base=10)
        mildLogInt = [f"log10_{i}" for i in mildInt]
        vis.corr_map(prot,mildLogInt, annot=True)
        plt.show()

    If you want to plot the clustermap, set cluster to True.
    The correlation coefficients are colour-coded.

    >>>  autoprot.visualization.corr_map(prot, mildLogInt, cmap="autumn", annot=None, cluster=True)

    .. plot::
        :context: close-figs

        vis.corr_map(prot, mildLogInt, cmap="autumn", annot=None, cluster=True)
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
    if save_dir is not None:
        if save_type == "pdf":
            plt.savefig(f"{save_dir}/{save_name}.pdf")
        elif save_type == "png":
            plt.savefig(f"{save_dir}/{save_name}.png")


def prob_plot(df, col, dist="norm", figsize=(6, 6), ax=None):
    # noinspection PyUnresolvedReferences
    r"""
    Plot a QQ_plot of the provided column.

    Data are compared against a theoretical distribution (default is normal)

    Parameters
    ----------
    ax : plt.axis
        Axis to plot on, optional.
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

    >>> vis.prob_plot(prot,'log10_Intensity H BC18_1')

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

        vis.prob_plot(prot,'log10_Intensity H BC18_1')
        plt.show()

    In contrast when the data does not follow the distribution, outliers from the
    linear plot will be visible.

    >>> vis.prob_plot(prot,'log10_Intensity H BC18_1', dist=stats.uniform)

    .. plot::
        :context: close-figs

        import scipy.stats as stats
        vis.prob_plot(prot,'log10_Intensity H BC18_1', dist=stats.uniform)

    """
    t = stats.probplot(df[col].replace([-np.inf, np.inf], [np.nan, np.nan]).dropna(), dist=dist)
    label = f"R²: {round(t[1][2], 4)}"
    y = []
    x = []
    for i in np.linspace(min(t[0][0]), max(t[0][0]), 100):
        y.append(t[1][0] * i + t[1][1])
        x.append(i)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(t[0][0], t[0][1], alpha=.3, color="purple",
               label=label)
    ax.plot(x, y, color="teal")
    sns.despine()
    ax.set_title(f"Probability Plot\n{col}")
    ax.set_xlabel("Theorectical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    plt.legend()


def boxplot(df: pd.DataFrame, reps: list, title: Union[str, list[str], None] = None, labels: Union[list, None] = None,
            compare: bool = False, ylabel: str = "log_fc", file: Union[None, str] = None, ret_fig: bool = False,
            figsize: tuple = (15, 5), ax: Union[plt.axis, None] = None, **kwargs: object) -> plt.figure:
    # noinspection PyUnresolvedReferences
    r"""
    Plot intensity boxplots.

    Parameters
    ----------
    ax : plt.axis
        Axis to plot on, optional.
    df : pd.Dataframe
        INput dataframe.
    reps : list
        Colnames of replicates.
    title : str or list of str, optional
        Title(s) of the plot. List of two if compare is True. The default is None.
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

    if compare:
        # check if inputs make sense
        if len(reps) != 2:
            raise ValueError("You want to compare two replicates, please provide two lists of column names.")
        if (labels is not None) and (len(labels) != 2):
            raise ValueError("You want to compare two replicates, please provide two lists of row labels.")
        if (title is not None) and (len(title) != 2):
            raise ValueError("You want to compare two replicates, please provide two titles.")

        if ax is not None:
            raise ValueError('You cannot use compare and specify an axis. Do either.')

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

        if labels is not None:
            for idx, l in zip([0, 1], labels):
                temp = ax[idx].set_xticklabels(l)
                for i, label in enumerate(temp):
                    label.set_y(label.get_position()[1] - (i % 2) * .05)
        else:
            ax[0].set_xticklabels([str(i + 1) for i in range(len(reps[0]))])
            ax[1].set_xticklabels([str(i + 1) for i in range(len(reps[1]))])
    else:
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        else:
            fig = ax.get_figure()

        df[reps].boxplot(ax=ax, **kwargs)
        ax.grid(False)
        plt.title(title)
        plt.ylabel(ylabel)

        ticks_loc = ax.get_xticks().tolist()
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))

        if labels is not None:
            temp = ax.set_xticklabels(labels)
            for i, label in enumerate(temp):
                label.set_y(label.get_position()[1] - (i % 2) * .05)
        else:
            ax.set_ticklabels(str(i + 1) for i in range(len(reps)))
        if ylabel == "log_fc":
            ax.axhline(0, 0, 1, color="gray", ls="dashed")
    sns.despine()

    if file is not None:
        plt.savefig(fr"{file}/BoxPlot.pdf")
    if ret_fig:
        return fig


def intensity_rank(data, rank_col="log10_Intensity", annotate_colname=None, n: Union[int, None] = 5,
                   title="Rank Plot", figsize=(15, 7), file=None, hline=None,
                   ax=None, highlight=None, kwargs_highlight=None, ascending=True, **kwargs):
    # noinspection PyUnresolvedReferences
    """
    Draw a rank plot.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe.
    rank_col : str, optional
        the column with the values to be ranked (e.g. Intensity values).
        The default is "log10_Intensity".
    annotate_colname : str, optional
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
    highlight : pd.Index, optional
        Index of the data to highlight.
        The default is None.
    kwargs_highlight: dict, optional
        Keyword arguments to be passed to the highlight plot.
    ascending : bool, optional
        Whether to sort the data in ascending order.
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

    >>> autoprot.visualization.intensity_rank(data, rank_col="log10_Intensity",
    ...                                      annotate_colname="Gene names", n=15,
    ...                                      title="Rank Plot",
    ...                                      hline=8, marker="d")

    .. plot::
        :context: close-figs

        data = pp.log(prot,["Intensity"], base=10)
        data = data[["log10_Intensity", "Gene names"]]
        data = data[data["log10_Intensity"]!=-np.inf]

        vis.intensity_rank(data, rank_col="log10_Intensity", label="Gene names", n=15, title="Rank Plot",
                         hline=8, marker="d")

    """
    # remove NaNs
    data = data.copy().dropna(subset=[rank_col])
    # sort by rank
    data = data.sort_values(by=rank_col, ascending=ascending)
    # generate ranked index
    data['# rank'] = np.arange(1, len(data) + 1)

    # new plot if no axis was given
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # plot on the axis
    sns.scatterplot(data=data,
                    x='# rank',
                    y=rank_col,
                    linewidth=0,
                    ax=ax,
                    legend=False,
                    **kwargs)

    if hline is not None:
        ax.axhline(hline, 0, 1, ls="dashed", color="lightgray")

    if annotate_colname is not None:

        if highlight is None:
            if n is not None:
                highlight = []
                kwargs_highlight = []
        else:
            highlight = [highlight, ]
            kwargs_highlight = [kwargs_highlight, ]

        # add the n largest and small ranks to the highlight
        if n is not None:
            highlight.append(data.nlargest(n, '# rank').index.union(data.nsmallest(n, '# rank').index))
            kwargs_highlight.append({'color': 'salmon'})

        if highlight is not None:
            _plot_highlights_scatter(highlight=highlight,
                                     kwargs_highlight=kwargs_highlight,
                                     df=data,
                                     ax=ax,
                                     x_colname='# rank',
                                     y_colname=rank_col,
                                     pointsize_colname=None)

            _label_scatter(df=data,
                           ax=ax,
                           x_colname="# rank",
                           y_colname=rank_col,
                           annotate="highlight",
                           highlight=highlight,
                           annotate_colname=annotate_colname,
                           annotate_density=100)
        else:
            print("annotate colname provided but neither n nor highlight given. Skipping annotation.")

        plt.title(title)

    if file is not None:
        plt.savefig(fr"{file}/RankPlot.pdf")


def venn_diagram(df: pd.DataFrame, figsize: tuple = (10, 10), ret_fig: bool = False, proportional: bool = True):
    # noinspection PyUnresolvedReferences
    # noinspection PyShadowingNames
    r"""
    Draw vennDiagrams.

    The .venn_diagram() function allows to draw venn diagrams for 2 to 6 replicates.
    Even though you can compare 6 replicates in a venn diagram does not mean
    that you should. It becomes extremly messy.

    The labels in the diagram can be read as follows:
    Comparing two conditions you will see the labels 10, 11 and 01. This can be read as:
    Only in replicate 1 (10), in both replicates (11) and only in replicate 2 (01).
    The same notation extends to all venn diagrams.

    Notes
    -----
    venn_diagram compares row containing not NaN between columns. Therefore,
    you have to pass columns containing NaN on rows where no common protein was
    found (e.g. after ratio calculation).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    figsize : tuple of int, optional
        Figure size. The default is (10,10).
    ret_fig : bool, optional
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
    >>> autoprot.visualization.venn_diagram(data, figsize=(5,5))

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
        vis.venn_diagram(data, figsize=(5,5))
        plt.show()

    Only up to three conditions can be compared in non-proportional Venn
    diagrams

    >>> autoprot.visualization.venn_diagram(data, figsize=(5,5), proportional=False)

    .. plot::
        :context: close-figs

        vis.venn_diagram(data, figsize=(5,5), proportional=False)
        plt.show()

    Copmaring up to 6 conditions is possible but the resulting Venn diagrams
    get quite messy.

    >>> data = prot[twitchVsmild[:6]]
    >>> vis.venn_diagram(data, figsize=(20,20))

    .. plot::
        :context: close-figs

        data = prot[twitchVsmild[:6]]
        vis.venn_diagram(data, figsize=(20,20))
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

    g1 = data[[reps[0]] + ["UID"]]
    g2 = data[[reps[1]] + ["UID"]]
    g1 = set(g1["UID"][g1[reps[0]].notnull()].values)
    g2 = set(g2["UID"][g2[reps[1]].notnull()].values)

    if n == 2:
        if proportional:
            venn2([g1, g2], set_labels=reps)
        else:
            labels = venn.get_labels([g1, g2], fill=["number", "logic"])
            fig, ax = venn.venn2(labels, names=[reps[0], reps[1]], figsize=figsize)
            if ret_fig:
                return fig

    elif n == 3:
        g3 = data[[reps[2]] + ["UID"]]
        g3 = set(g3["UID"][g3[reps[2]].notnull()].values)
        if proportional:
            venn3([g1, g2, g3], set_labels=reps)
        else:
            labels = venn.get_labels([g1, g2, g3], fill=["number", "logic"])
            fig, ax = venn.venn3(labels, names=[reps[0], reps[1], reps[2]], figsize=figsize)
            if ret_fig:
                return fig

    elif n == 4:
        g3 = data[[reps[2]] + ["UID"]]
        g4 = data[[reps[3]] + ["UID"]]
        g3 = set(g3["UID"][g3[reps[2]].notnull()].values)
        g4 = set(g4["UID"][g4[reps[3]].notnull()].values)
        labels = venn.get_labels([g1, g2, g3, g4], fill=["number", "logic"])
        fig, ax = venn.venn4(labels, names=[reps[0], reps[1], reps[2], reps[3]], figsize=figsize)

        if ret_fig:
            return fig
    elif n == 5:
        g3 = data[[reps[2]] + ["UID"]]
        g4 = data[[reps[3]] + ["UID"]]
        g5 = data[[reps[4]] + ["UID"]]
        g3 = set(g3["UID"][g3[reps[2]].notnull()].values)
        g4 = set(g4["UID"][g4[reps[3]].notnull()].values)
        g5 = set(g5["UID"][g5[reps[4]].notnull()].values)
        labels = venn.get_labels([g1, g2, g3, g4, g5], fill=["number", "logic"])
        fig, ax = venn.venn5(labels, names=[reps[0], reps[1], reps[2], reps[3], reps[4]], figsize=figsize)

        if ret_fig:
            return fig
    elif n == 6:
        g3 = data[[reps[2]] + ["UID"]]
        g4 = data[[reps[3]] + ["UID"]]
        g5 = data[[reps[4]] + ["UID"]]
        g6 = data[[reps[5]] + ["UID"]]
        g3 = set(g3["UID"][g3[reps[2]].notnull()].values)
        g4 = set(g4["UID"][g4[reps[3]].notnull()].values)
        g5 = set(g5["UID"][g5[reps[4]].notnull()].values)
        g6 = set(g6["UID"][g6[reps[5]].notnull()].values)
        labels = venn.get_labels([g1, g2, g3, g4, g5, g6], fill=["number", "logic"])
        fig, ax = venn.venn6(labels, names=[reps[0], reps[1], reps[2], reps[3], reps[4], reps[5]], figsize=figsize)

        if ret_fig:
            return fig


# COMMON FOR ALL SCATTER PLOTS
def _limit_density(xs, ys, ss, threshold):
    """
    Reduce the points for annotation through a point density threshold.

    Parameters
    ----------
    xs: numpy.ndarray
        x values
    ys: numpy.ndarray
        y values
    ss: numpy.ndarray
        labels
    threshold: float
        Probability threshold. Only points with 1/density above the value will be retained.
    """
    # Remove NaNs
    if np.isnan(np.array(xs)).any() or np.isnan(np.array(ys)).any():
        nan_idx = np.isnan(np.array(xs)) | np.isnan(np.array(ys))
        xs = xs[~nan_idx]
        ys = ys[~nan_idx]
        ss = ss[~nan_idx]

    # if there is only one datapoint kernel density estimation with fail
    if len(xs) < 3 or len(ys) < 3:
        print("Not enough data points for KDE. Return original point cloud.")
        return xs, ys, ss

    # Make some random Gaussian data
    data = np.array(list(zip(xs, ys)))
    try:
        # Compute KDE
        kde = gaussian_kde(data.T)
    except LinAlgError:
        print("Couldn't compute KDE. Return original point cloud.")
        return xs, ys, ss
    # Choice probabilities are computed from inverse probability density in KDE
    p = 1 / kde.pdf(data.T)
    # Normalize choice probabilities
    p /= np.sum(p)
    # Make subsample using choice probabilities
    idx = np.asarray(p > threshold).nonzero()

    return xs[idx], ys[idx], ss[idx]


def _init_scatter(ax, df, figsize, pointsize_colname, pointsize_scaler):
    # draw figure
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot()  # for a bare minimum plot you do not need this line
    else:
        fig = ax.get_figure()

    # PLOTTING
    if pointsize_colname is not None:
        if not is_numeric_dtype(df[pointsize_colname]):
            raise ValueError(
                "The column provided for point sizing should only contain numeric values"
            )
        # normalize the point sizes
        df["s"] = (
                pointsize_scaler
                * 100
                * (df[pointsize_colname] - df[pointsize_colname].min())
                / df[pointsize_colname].max()
        )

    return fig, ax, df


def _stylize_scatter(df, ax, show_legend, show_caption, title, pointsize_colname, pointsize_scaler):
    if show_legend:
        _stylize_scatter_legend(ax, pointsize_colname, df, pointsize_scaler)
    if show_caption:
        plt.figtext(
            1,  # x position
            -0.1,  # y position
            f"total = {len(df)} entries",  # text
            transform=plt.gca().transAxes,
            wrap=True,
            horizontalalignment="right"
        )

    if title is not None:
        if show_legend:
            ax.set_title(title, y=1.1, loc='left')
        else:
            ax.set_title(title, loc='left')


def _stylize_scatter_legend(ax, pointsize_colname, df, pointsize_scaler):
    legend = ax.legend(loc='upper left', bbox_to_anchor=(0, 0.9, 1, 0.1), mode="expand", ncol=3,
                       bbox_transform=ax.transAxes)

    # this fixes the legend points having the same size as the points in the scatter plot
    for handle in legend.legendHandles:
        handle._sizes = [30]
    ax.add_artist(legend)

    # shrink the plot so that the legend does not cover any points
    lim = ax.get_ylim()
    ax.set_ylim(lim[0], 1.25 * lim[1])

    if pointsize_colname is not None:
        mlabels = np.linspace(
            start=df[pointsize_colname].max() / 5,
            stop=df[pointsize_colname].max(),
            num=4,
        )

        msizes = pointsize_scaler * 100 * np.linspace(start=0.2, stop=1, num=4)

        markers = [
            plt.scatter([], [], c="grey", s=size, label=int(label))
            for label, size in zip(mlabels, msizes)
        ]
        legend2 = ax.legend(handles=markers, loc="lower left")
        ax.add_artist(legend2)


def _label_scatter(df: pd.DataFrame, ax: matplotlib.axes.Axes, x_colname: str, y_colname: str,
                   annotate: Union[str, None], highlight: Union[pd.Index, List[pd.Index], None],
                   annotate_colname: str, annotate_density: float) -> None:
    """
    Add labels to a scatter plot based on certain conditions.

    Parameters
    ----------
    df : pd.DataFrame
       The dataframe containing the data to be plotted.
    ax : matplotlib.axes.Axes
       The axes object on which to add the labels.
    x_colname : str
       The name of the column in df to use for the x-values of the scatter plot.
    y_colname : str
       The name of the column in df to use for the y-values of the scatter plot.
    annotate : str or None
       The condition to determine which points to label. Can be "highlight", "p-value and log2FC", "p-value",
       "log2FC" or None.
    highlight : pd.Index, list of pd.Index or None
       A list of indices to highlight if annotate is "highlight".
    annotate_colname : str
       The name of the column in df to use for the labels of the scatter plot.
    annotate_density : float
       The density of the annotations. Used to limit the number of annotations if there are too many points.

    Returns
    -------
    None

    Raises
    ------
    ValueError
       If annotate is "highlight" but highlight is None, or if annotate is not one of the allowed values.
    """
    # If no annotation is required, return without doing anything
    if annotate is None:
        return

    # If annotation is set to "highlight", check if highlight indices are provided
    if annotate == "highlight":
        if highlight is None:
            raise ValueError("Highlight is None, but 'highlight' was passed to 'annotate'.")
        elif isinstance(highlight, list):
            to_label = reduce(lambda x, y: x.union(y), highlight)
        elif isinstance(highlight, pd.Index):
            to_label = highlight
        else:
            raise ValueError("'highlight' must be a list of pd.Index or a pd.Index or None.")
    # If annotation is set to one of the other allowed values, determine the points to label based on the condition
    elif annotate in ["p-value and log2FC", "p-value", "log2FC", "ratio_thresh"]:
        to_label = df[df["SigCat"] == annotate].index
    # If annotation is not one of the allowed values, raise an error
    else:
        raise ValueError('Annotate must be "highlight", "p-value and log2FC", "p-value", "log2FC" or None')

    # Limit the number of annotations if there are too many points
    xs, ys, ss = _limit_density(df.loc[to_label, x_colname].to_numpy(),
                                df.loc[to_label, y_colname].to_numpy(),
                                df.loc[to_label, annotate_colname].to_numpy(),
                                1 / annotate_density)

    # Add the labels to the scatter plot
    texts = [
        ax.text(x, y, s, ha="center", va="center") for (x, y, s) in zip(xs, ys, ss)
    ]
    # adjust the label positions so that they do not overlap
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color="black"), ax=ax)


def _plot_highlights_scatter(highlight: Union[pd.Index, list[pd.Index], None],
                             kwargs_highlight: Union[dict, list[dict], None],
                             df: pd.DataFrame,
                             ax: plt.axis,
                             x_colname: str,
                             y_colname: str,
                             pointsize_colname: Union[str, None],
                             ):
    if isinstance(highlight, list):  # highlight is a list
        if kwargs_highlight is None:  # if no kwargs are given, generate a matching length kwarg list with Nones
            print("No kwargs provided for highlights. Using default values.")
            kwargs_highlight = [None, ] * len(highlight)
        elif isinstance(kwargs_highlight, list) and len(kwargs_highlight) == 1 and \
                isinstance(kwargs_highlight[0], dict):  # one-element list
            kwargs_highlight = [kwargs_highlight[0], ] * len(highlight)
        elif isinstance(kwargs_highlight, dict):  # single dictionary to propagate to all highlights
            kwargs_highlight = [kwargs_highlight, ] * len(highlight)
        elif len(highlight) != len(kwargs_highlight):
            raise ValueError("'highlight' and 'kwargs_highlight' must be lists of the same length.")

        for h, k in zip(highlight, kwargs_highlight):
            k = com.set_default_kwargs(
                k,
                dict(
                    color="orange",
                    alpha=0.8,
                    s=df.loc[h, "s"] if pointsize_colname is not None else None,
                ),
            )
            ax.scatter(
                df.loc[h, x_colname],
                df.loc[h, y_colname],
                **k,
            )
    elif isinstance(highlight, pd.Index):  # highlight is an index
        if kwargs_highlight is None:  # if no kwargs are given, generate a matching length kwarg list with Nones
            print("No kwargs provided for highlights. Using default values.")
        elif isinstance(kwargs_highlight, list) and len(kwargs_highlight) == 1 and \
                isinstance(kwargs_highlight[0], dict):  # kwargs is one-element list
            print("Only one kwargs provided for highlights. Using this for all highlights.")
            kwargs_highlight = kwargs_highlight[0]
        elif isinstance(kwargs_highlight, dict):  # kwargs is single dict
            pass
        else:
            raise ValueError("'highlight' and 'kwargs_highlight' must be lists of the same length or pd.Index and "
                             "dict.")

        kwargs_highlight = com.set_default_kwargs(
            kwargs_highlight,
            dict(
                color="orange",
                alpha=0.8,
                s=df.loc[highlight, "s"] if pointsize_colname is not None else None,
            ),
        )

        ax.scatter(
            df.loc[highlight, x_colname],
            df.loc[highlight, y_colname],
            **kwargs_highlight,
        )
    else:
        raise ValueError("'highlight' and 'kwargs_highlight' must be lists of the same length or pd.Index and "
                         f"dict. However 'highlight' was {type(highlight)} and 'kwargs_highlight' "
                         f"was {type(kwargs_highlight)}")


# VOLCANO PLOTS #
def _prep_volcano_data(
        df, log_fc_colname, score_colname, p_colname, p_thresh, log_fc_thresh
):
    """
    Input check for volcano functions.

    Raises
    ------
    ValueError
        If neither a p-score nor a p value is provided by the user.

    """
    # Work with a copy of the dataframe
    df = df.copy()

    if score_colname is None and p_colname is None:
        raise ValueError("You have to provide either a score or a (adjusted) p value.")
    elif score_colname is None:
        df["score"] = -np.log10(df[p_colname])
        score_colname = "score"
    else:
        df.rename(columns={score_colname: "score"}, inplace=True)
        score_colname = "score"
        p_colname = "p"
        df["p"] = 10 ** (df["score"] * -1)

    # four groups of points are present in a volcano plot:
    # (1) non-significant
    df["SigCat"] = "NS"
    # (2) significant by score
    df.loc[df[p_colname] < p_thresh, "SigCat"] = "p-value"

    if log_fc_thresh is not None:
        # (3) significant above or below fc-thresh
        df.loc[
            (df["SigCat"] == "NS") & (abs(df[log_fc_colname]) > log_fc_thresh), "SigCat"
        ] = "log2FC"
        # (4) significant by both
        df.loc[
            (df["SigCat"] == "p-value") & (abs(df[log_fc_colname]) > log_fc_thresh),
            "SigCat",
        ] = "p-value and log2FC"

    unsig = df[df["SigCat"] == "NS"].index
    sig_fc = df[df["SigCat"] == "log2FC"].index
    sig_p = df[df["SigCat"] == "p-value"].index
    sig_both = df[df["SigCat"] == "p-value and log2FC"].index

    return df, score_colname, unsig, sig_fc, sig_p, sig_both


def volcano(
        df: pd.DataFrame,
        log_fc_colname: str,
        p_colname: str = None,
        score_colname: str = None,
        p_thresh: float or None = 0.05,
        log_fc_thresh: float or None = np.log2(2),
        pointsize_colname: str or float = None,
        pointsize_scaler: float = 1,
        highlight: Union[pd.Index, list[pd.Index], None] = None,
        title: str = None,
        show_legend: bool = True,
        show_caption: bool = True,
        show_thresh: bool = True,
        ax: plt.axis = None,
        ret_fig: bool = True,
        figsize: tuple = (8, 8),
        annotate: Union[Literal["highlight", "p-value and log2FC", "p-value", "log2FC"], None] = "p-value "
                                                                                                 "and log2FC",
        annotate_colname: str = "Gene names",
        kwargs_ns: dict = None,
        kwargs_p_sig: dict = None,
        kwargs_log_fc_sig: dict = None,
        kwargs_both_sig: dict = None,
        kwargs_highlight: Union[dict, list[dict], None] = None,
        annotate_density: int = 100,
):
    # noinspection PyUnresolvedReferences
    # noinspection PyShadowingNames
    # noinspection PyPep8
    """
    Return static volcano plot.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the data to plot.
    log_fc_colname : str
        column of the dataframe with the log fold change.
    p_colname : str, optional
        column of the dataframe containing p values (provide score_colname or p_colname).
        The default is None.
    score_colname : str, optional
        column of the dataframe containing -log10(p values) (provide score or p).
        The default is None.
    p_thresh : float or None, optional
        p-value threshold under which a entry is deemed significantly regulated.
        The default is 0.05.
    log_fc_thresh : float or None, optional
        fold change threshold at which an entry is deemed significant regulated.
        The default is log2(2).
    pointsize_colname: str or float, optional
        Name of a column to use as measure for point size.
        Alternatively the size of all points.
    pointsize_scaler: float, optional
        Value to scale all point sizes.
        Default is 1.
    highlight : pd.Index or list of pd.Index, optional
        Rows to highlight in the plot.
        The default is None.
    title : str, optional
        Title for the plot. The default is None.
    show_legend : bool, optional
        Whether to plot a legend. The default is True.
    show_caption: bool, optional
        Whether to show the caption below the plot. The default is True.
    show_thresh: bool, optional
        Whether to show the thresholds as dashed lines. The default is True.
    ax : matplotlib.pyplot.axis, optional
        Axis to plot on
    ret_fig : bool, optional
        Whether to return the figure, can be used to further
        customize it later. The default is False.
    figsize: tuple, optional
        The size of the figure.
        Default is (8,8)
    annotate: "highlight", "p-value and log2FC", "p-value", "log2FC", None or pd.Index, optional
        Whether to generate labels for the significant or highlighted points.
        Default is "p-value and log2FC".
    annotate_colname: str, optional
        The column name to use for the annotation.
        Default is "Gene names".
    annotate_density : int, optional
        The density (normalised to 1) below which points are ignored from labelling.
        Default is 100.
    kwargs_ns : dict, optional
        Custom kwargs to pass to matplotlib.pyplot.scatter when generating the non-significant points.
        The default is None.
    kwargs_p_sig : dict, optional
        Custom kwargs to pass to matplotlib.pyplot.scatter when generating the p-value significant points.
        The default is None.
    kwargs_log_fc_sig : dict, optional
        Custom kwargs to pass to matplotlib.pyplot.scatter when generating the log2 fold-change significant points.
        The default is None.
    kwargs_both_sig : dict, optional
        Custom kwargs to pass to matplotlib.pyplot.scatter when generating the overall significant points.
        The default is None.
    kwargs_highlight : dict or list of dict, optional
        Custom kwargs to pass to plt.scatter when generating the highlighted points.
        Only relevant if highlight is not None.
        The default is None.

    Returns
    -------
    matplotlib.figure
        The figure object if ret_fig kwarg is True.

    Examples
    --------
    The standard setting of volcano should be sufficient for getting a first glimpse on the data. Note that the
    point labels are automatically adjusted to prevent overlapping text.

    >>> prot_limma['Gene names 1st'] = prot_limma['Gene names'].str.split(';').str[0]
    >>> fig = vis.volcano(
    >>>     df=prot_limma,
    >>>     log_fc_colname="logFC_TvM",
    >>>     p_colname="P.Value_TvM",
    >>>     title="Volcano Plot",
    >>>     annotate_colname="Gene names 1st",
    >>> )
    >>>
    >>> fig.show()

    .. plot::
        :context: close-figs

         prot = pd.read_csv("_static/testdata/proteinGroups.zip", sep='\t', low_memory=False)
         prot = pp.cleaning(prot, "proteinGroups")
         protRatio = prot.filter(regex="^Ratio .\/.( | normalized )B").columns
         prot = pp.log(prot, protRatio, base=2)
         twitchVsmild = ['log2_Ratio H/M normalized BC18_1','log2_Ratio M/L normalized BC18_2',
                         'log2_Ratio H/M normalized BC18_3',
                         'log2_Ratio H/L normalized BC36_1','log2_Ratio H/M normalized BC36_2',
                         'log2_Ratio M/L normalized BC36_2']
         prot_limma = ana.limma(prot, twitchVsmild, cond="_TvM")
         prot_limma['Gene names 1st'] = prot_limma['Gene names'].str.split(';').str[0]

         fig = vis.volcano(
             df=prot_limma,
             log_fc_colname="logFC_TvM",
             p_colname="P.Value_TvM",
             title="Volcano Plot",
             annotate_colname="Gene names 1st",
         )

         fig.show()

    Thresholds can easily be modified using the log_fc_thresh and p_thresh kwargs:

    >>> fig = vis.volcano(
    >>>     df=prot_limma,
    >>>     log_fc_colname="logFC_TvM",
    >>>     p_colname="P.Value_TvM",
    >>>     p_thresh=0.01,
    >>>     title="Volcano Plot",
    >>>     annotate_colname="Gene names 1st",
    >>> )
    >>>
    >>> fig.show()

    .. plot::
        :context: close-figs

        fig = vis.volcano(
            df=prot_limma,
            log_fc_colname="logFC_TvM",
            p_colname="P.Value_TvM",
            p_thresh=0.01,
            title="Volcano Plot",
            annotate_colname="Gene names 1st",
        )

        fig.show()

    All points in the plot can be customized by supplying kwargs to the volcano function. These can be any arguments
    accepted by matplotlib.pyplot.scatter.

    >>> non_sig_kwargs = dict(color="black", marker="x")
    >>> sig_kwargs = dict(color="red", marker=7, s=100)
    >>>
    >>> fig = vis.volcano(
    >>>     df=prot_limma,
    >>>     log_fc_colname="logFC_TvM",
    >>>     p_colname="P.Value_TvM",
    >>>     p_thresh=0.01,
    >>>     title="Customised Volcano Plot",
    >>>     annotate_colname="Gene names 1st",
    >>>     kwargs_ns=non_sig_kwargs,
    >>>     kwargs_p_sig=non_sig_kwargs,
    >>>     kwargs_log_fc_sig=non_sig_kwargs,
    >>>     kwargs_both_sig=sig_kwargs,
    >>> )
    >>>
    >>> fig.show()

    .. plot::
        :context: close-figs

        non_sig_kwargs = dict(color="black", marker="x")
        sig_kwargs = dict(color="red", marker=7, s=100)

        fig = vis.volcano(
            df=prot_limma,
            log_fc_colname="logFC_TvM",
            p_colname="P.Value_TvM",
            p_thresh=0.01,
            title="Customised Volcano Plot",
            annotate_colname="Gene names 1st",
            kwargs_ns=non_sig_kwargs,
            kwargs_p_sig=non_sig_kwargs,
            kwargs_log_fc_sig=non_sig_kwargs,
            kwargs_both_sig=sig_kwargs,
        )

        fig.show()

    All other elements of the plot can be customised by accessing the figure and axis objects. The axis can be extracted
    from the figure returned by volano.

    >>> fig = vis.volcano(
    >>>     df=prot_limma,
    >>>     log_fc_colname="logFC_TvM",
    >>>     p_colname="P.Value_TvM",
    >>>     title="Volcano Plot",
    >>>     annotate_colname="Gene names 1st",
    >>> )
    >>>
    >>> ax = fig.gca()
    >>> ax.axhline(y=3, color="red", linestyle=":")
    >>> ax.axhline(y=4, color="blue", linestyle=":")
    >>>
    >>> fig.show()

    .. plot::
        :context: close-figs

        fig = vis.volcano(
           df=prot_limma,
           log_fc_colname="logFC_TvM",
           p_colname="P.Value_TvM",
           title="Volcano Plot",
           annotate_colname="Gene names 1st",
        )

        ax = fig.gca()
        ax.axhline(y=3, color='red', linestyle=':')
        ax.axhline(y=4, color='blue', linestyle=':')

        fig.show()

    Volcano also allows you to supply a numeric column of your dataframe as agument to pointsize_colname. The
    numeric values will be noramlized between min and max and used for sizing the points. If the standard size is
    inconvenient, the point_scaler kwarg enables manual adjustemnt of the point sizes.

    >>> fig = vis.volcano(
    >>>     df=prot_limma,
    >>>     log_fc_colname="logFC_TvM",
    >>>     p_colname="P.Value_TvM",
    >>>     pointsize_colname='iBAQ',
    >>>     pointsize_scaler=5,
    >>>     title="Volcano Plot",
    >>>     annotate_colname="Gene names 1st",
    >>> )
    >>>
    >>> fig.show()

    .. plot::
        :context: close-figs

        fig = vis.volcano(
            df=prot_limma,
            log_fc_colname="logFC_TvM",
            p_colname="P.Value_TvM",
            pointsize_colname='iBAQ',
            pointsize_scaler=5,
            title="Volcano Plot",
            annotate_colname="Gene names 1st",
        )

        fig.show()

    Custom points can also be highlighted by providing a pandas Index object of the corresponding rows as input to
    the highlight kwarg. Note that the annotated kwarg must be updated if you want to also label your highlighted
    points.

    >>> to_highlight = prot_limma[prot_limma['iBAQ'] > 10e8].index
    >>>
    >>> fig = vis.volcano(
    >>>     df=prot_limma,
    >>>     log_fc_colname="logFC_TvM",
    >>>     p_colname="P.Value_TvM",
    >>>     highlight=to_highlight,
    >>>     annotate='highlight',
    >>>     title="Volcano Plot",
    >>>     annotate_colname="Gene names 1st",
    >>> )
    >>>
    >>> fig.show()

    .. plot::
        :context: close-figs

        to_highlight = prot_limma[prot_limma['iBAQ'] > 10e8].index

        fig = vis.volcano(
            df=prot_limma,
            log_fc_colname="logFC_TvM",
            p_colname="P.Value_TvM",
            highlight=to_highlight,
            annotate='highlight',
            title="Volcano Plot",
            annotate_colname="Gene names 1st",
        )

        fig.show()
    """

    # check for input correctness and make sure score is present in df for plot
    df, score_colname, unsig, sig_fc, sig_p, sig_both = _prep_volcano_data(
        df, log_fc_colname, score_colname, p_colname, p_thresh, log_fc_thresh
    )

    fig, ax, df = _init_scatter(ax, df, figsize, pointsize_colname, pointsize_scaler)

    # Non-Significant
    kwargs_ns = com.set_default_kwargs(
        kwargs_ns,
        dict(
            color="lightgrey",
            alpha=0.5,
            s=df.loc[df["SigCat"] == "NS", "s"]
            if pointsize_colname is not None
            else None,
            label="NS",
        )
    )
    ax.scatter(
        df.loc[df["SigCat"] == "NS", log_fc_colname],
        df.loc[df["SigCat"] == "NS", "score"],
        **kwargs_ns,
    )

    # Significant by p-value
    kwargs_p_sig = com.set_default_kwargs(
        kwargs_p_sig,
        dict(
            color="lightblue",
            alpha=0.5,
            s=df.loc[df["SigCat"] == "p-value", "s"]
            if pointsize_colname is not None
            else None,
            label="p-value",
        ),
    )
    ax.scatter(
        df.loc[df["SigCat"] == "p-value", log_fc_colname],
        df.loc[df["SigCat"] == "p-value", "score"],
        **kwargs_p_sig,
    )

    # significant by log fold-change
    kwargs_log_fc_sig = com.set_default_kwargs(
        kwargs_log_fc_sig,
        dict(
            color="lightgreen",
            alpha=0.5,
            s=df.loc[df["SigCat"] == "log2FC", "s"]
            if pointsize_colname is not None
            else None,
            label=r"$\mathregular{log_2 FC}$",
        ),
    )
    ax.scatter(
        df.loc[df["SigCat"] == "log2FC", log_fc_colname],
        df.loc[df["SigCat"] == "log2FC", "score"],
        **kwargs_log_fc_sig,
    )

    # significant by both
    kwargs_both_sig = com.set_default_kwargs(
        kwargs_both_sig,
        dict(
            color="tomato",
            alpha=0.5,
            s=df.loc[df["SigCat"] == "p-value and log2FC", "s"]
            if pointsize_colname is not None
            else None,
            label=r"$\mathregular{log_2 FC}$ & p-value",
        ),
    )
    ax.scatter(
        df.loc[df["SigCat"] == "p-value and log2FC", log_fc_colname],
        df.loc[df["SigCat"] == "p-value and log2FC", "score"],
        **kwargs_both_sig,
    )

    if highlight is not None:
        _plot_highlights_scatter(highlight=highlight, kwargs_highlight=kwargs_highlight, df=df,
                                 x_colname=log_fc_colname, y_colname="score", ax=ax,
                                 pointsize_colname=pointsize_colname)

    ax.set_xlabel(r"$\mathregular{log_2 fold-change}$")
    ax.set_ylabel(r"$\mathregular{-log_{10} P}$")

    # ANNOTATION AND LABELING
    _label_scatter(df=df, ax=ax, x_colname=log_fc_colname, y_colname='score', annotate=annotate, highlight=highlight,
                   annotate_colname=annotate_colname, annotate_density=annotate_density)

    # STYLING
    _stylize_scatter(df, ax, show_legend, show_caption, title, pointsize_colname, pointsize_scaler)

    if show_thresh:
        if log_fc_thresh is not None:
            ax.axvline(x=log_fc_thresh, color="black", linestyle="--")
            ax.axvline(x=-log_fc_thresh, color="black", linestyle="--")
        if p_thresh is not None:
            ax.axhline(y=-np.log10(p_thresh), color="black", linestyle="--")

    if ret_fig:
        return fig


def ivolcano(
        df: pd.DataFrame,
        log_fc_colname: str,
        p_colname: str = None,
        score_colname: str = None,
        p_thresh: float or None = 0.05,
        log_fc_thresh: float or None = None,
        annotate_colname: str = None,
        pointsize_colname: str or float = None,
        highlight: pd.Index = None,
        title: str = "Volcano Plot",
        show_legend: bool = True,
        ret_fig: bool = True,
):
    """
    Return interactive volcano plot.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the data to plot.
    log_fc_colname : str
        column of the dataframe with the log fold change.
    p_colname : str, optional
        column of the dataframe containing p values (provide score_colname or p_colname).
        The default is None.
    score_colname : str, optional
        column of the dataframe containing -log10(p values) (provide score or p).
        The default is None.
    p_thresh: float or None, optional
        p-value threshold under which an entry is deemed significantly regulated.
        The default is 0.05.
    log_fc_thresh : float or None, optional
        fold change threshold at which an entry is deemed significant regulated.
        The default is None
    annotate_colname : str, optional
        Colname to use for labels in interactive plot.
        The default is None.
    pointsize_colname: str or float, optional
        Name of a column to use as measure for point size.
        Alternatively the size of all points.
    highlight : pd.Index, optional
        Rows to highlight in the plot.
        The default is None.
    title : str, optional
        Title for the plot. The default is "Volcano Plot".
    show_legend : bool, optional
        Whether to plot a legend. The default is True.
    ret_fig : bool, optional
        Whether to return the figure, can be used to further
        customize it afterward. The default is False.

    Returns
    -------
    plotly.figure
        The figure object.
    """

    # check for input correctness and make sure score is present in df for plot
    df, score_colname, unsig, sig_fc, sig_p, sig_both = _prep_volcano_data(
        df, log_fc_colname, score_colname, p_colname, p_thresh, log_fc_thresh
    )

    categories = ["NS", "log2FC", "p-value", "p-value and log2FC"]

    if highlight is not None:
        df["SigCat"] = "-"
        df.loc[highlight, "SigCat"] = "*"
        fig = (
            px.scatter(
                data_frame=df,
                x=log_fc_colname,
                y=score_colname,
                hover_name=annotate_colname,
                size=pointsize_colname,
                color="SigCat",
                opacity=0.5,
                category_orders={"SigCat": ["-", "*"]},
                title=title,
            )
            if annotate_colname is not None
            else px.scatter(
                data_frame=df,
                x=log_fc_colname,
                y=score_colname,
                size=pointsize_colname,
                color="SigCat",
                opacity=0.5,
                category_orders={"SigCat": ["-", "*"]},
                title=title,
            )
        )
    elif annotate_colname is not None:
        fig = px.scatter(
            data_frame=df,
            x=log_fc_colname,
            y=score_colname,
            hover_name=annotate_colname,
            size=pointsize_colname,
            color="SigCat",
            opacity=0.5,
            category_orders={"SigCat": categories},
            title=title,
        )
    else:
        fig = px.scatter(
            data_frame=df,
            x=log_fc_colname,
            y=score_colname,
            size=pointsize_colname,
            color="SigCat",
            opacity=0.5,
            category_orders={"SigCat": categories},
            title=title,
        )

    fig.update_yaxes(showgrid=False, zeroline=True)
    fig.update_xaxes(showgrid=False, zeroline=False)

    if p_thresh is not None:
        fig.add_trace(
            go.Scatter(
                x=[df[log_fc_colname].min(), df[log_fc_colname].max()],
                y=[-np.log10(p_thresh), -np.log10(p_thresh)],
                mode="lines",
                line=go.scatter.Line(color="teal", dash="longdash"),
                showlegend=False,
            )
        )
    if log_fc_thresh is not None:
        # add fold change visualization
        fig.add_trace(
            go.Scatter(
                x=[-log_fc_thresh, -log_fc_thresh],
                y=[0, df[score_colname].max()],
                mode="lines",
                line=go.scatter.Line(color="teal", dash="longdash"),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[log_fc_thresh, log_fc_thresh],
                y=[0, df[score_colname].max()],
                mode="lines",
                line=go.scatter.Line(color="teal", dash="longdash"),
                showlegend=False,
            )
        )

    fig.update_layout(
        template="simple_white",
        showlegend=show_legend,
    )

    if ret_fig:
        return fig
    else:
        fig.show()


# RATIO-RATIO PLOTS #
# Preparing the dataset
def _prep_ratio_data(
        df, col_name1, col_name2, ratio_thresh):
    # Work with a copy of the dataframe
    df = df.copy()

    if col_name1 is None or col_name2 is None:
        raise ValueError("You have to provide column names for both ratios.")

    # two groups of points are present in a ratio-ratio plot:
    # (1) non-significant
    df["SigCat"] = "NS"
    # (2) significant by score
    if ratio_thresh is not None:
        df.loc[(df[col_name1] > ratio_thresh) & (df[col_name2] > ratio_thresh), "SigCat"] = "ratio_thresh"
        df.loc[(df[col_name1] < ratio_thresh * -1) & (df[col_name2] < ratio_thresh * -1), "SigCat"] = "ratio_thresh"

    unsig = df[df["SigCat"] == "NS"].index
    sig_ratio = df[df["SigCat"] == "ratio_thresh"].index

    return df, col_name1, col_name2, unsig, sig_ratio


def ratio_plot(
        df: pd.DataFrame,
        col_name1: str,
        col_name2: str = None,
        ratio_thresh: float = None,
        xlabel: str = "Ratio col1",
        ylabel: str = "Ratio col2",
        pointsize_colname: str or float = None,
        pointsize_scaler: float = 1,
        highlight: Union[list, pd.Index, None] = None,
        title: str = None,
        show_legend: bool = True,
        show_caption: bool = True,
        show_thresh: bool = True,
        ax: plt.axis = None,
        ret_fig: bool = True,
        figsize: tuple = (8, 8),
        annotate: Union[Literal["highlight", "ratio_thresh"], None] = "ratio_thresh",
        annotate_colname: str = "Gene names",
        kwargs_ns: dict = None,
        kwargs_r_sig: dict = None,
        kwargs_highlight: Union[list, dict, None] = None,
        annotate_density: int = 100):
    # noinspection PyUnresolvedReferences
    """
    Plot a ratio vs. ratio plot based on a pandas dataframe.

    Parameters
    ----------
    df: pd.Dataframe
    col_name1: str
    col_name2: str, optional
    ratio_thresh: float, optional
    xlabel: str, optional
    ylabel: str, optional
    pointsize_colname: str or float, optional
    pointsize_scaler: float, optional
    highlight: pd.Index or list of pd.Index, optional
    title: str, optional
    show_legend: bool, optional
    show_caption: bool, optional
    show_thresh: bool, optional
    ax: plt.axis, optional
    ret_fig: bool, optional
    figsize: tuple of int, optional
    annotate: "highlight" or "ratio_thresh" or None or pd.Index, optional
    annotate_colname: str, optional
    kwargs_ns: dict, optional
    kwargs_r_sig: dict, optional
    kwargs_highlight: dict or list of dict, optional
    annotate_density: int, optional

    Returns
    -------
    plotly.figure
        The figure object.

    Examples
    --------
    Similar to the volcano plot function, the ratio plot takes a dataframe as input together with the two
    column names to plot.

    >>> fig = vis.ratio_plot(
    >>>     prot,
    >>>     col_name1='Ratio M/L BC18_1',
    >>>     col_name2='Ratio M/L BC18_2',
    >>>     ratio_thresh= 3,
    >>>     annotate_colname='Gene names 1st',
    >>>     xlabel= 'Ratio Rep1',
    >>>     ylabel = 'Ratio Rep2',
    >>>     annotate_density=20,
    >>> )

    fig.show()

    .. plot::
    :context: close-figs

    prot = pd.read_csv("../docsrc/_static/testdata/proteinGroups.zip", sep='\t', low_memory=False)
    prot = pp.cleaning(prot, "proteinGroups")
    protRatio = prot.filter(regex="^Ratio .\/.( | normalized )B").columns
    prot = pp.log(prot, protRatio, base=2)
    prot['Gene names 1st'] = prot['Gene names'].str.split(';').str[0]

    fig = vis.ratio_plot(
        prot,
        col_name1='Ratio M/L BC18_1',
        col_name2='Ratio M/L BC18_2',
        ratio_thresh= 3,
        annotate_colname='Gene names 1st',
        xlabel= 'Ratio Rep1',
        ylabel = 'Ratio Rep2',
        annotate_density=20,
    )

    fig.show()
    """
    # check for input correctness and make sure score is present in df for plot

    df, col_name1, col_name2, unsig, sig_ratio = _prep_ratio_data(df, col_name1, col_name2, ratio_thresh)

    fig, ax, df = _init_scatter(ax, df, figsize, pointsize_colname, pointsize_scaler)

    # Non-Significant
    kwargs_ns = com.set_default_kwargs(kwargs_ns,
                                       dict(color="lightgrey",
                                            alpha=0.5,
                                            s=df.loc[
                                                df["SigCat"] == "NS", "s"] if pointsize_colname is not None else None,
                                            label="NS",
                                            ))
    ax.scatter(
        df.loc[df["SigCat"] == "NS", col_name1],
        df.loc[df["SigCat"] == "NS", col_name2],
        **kwargs_ns,
    )

    # Significant by ratio_thresh
    kwargs_r_sig = com.set_default_kwargs(
        kwargs_r_sig,
        dict(
            color="#FF886D",
            alpha=0.5,
            s=df.loc[df["SigCat"] == "ratio_thresh", "s"]
            if pointsize_colname is not None
            else None,
            label="Significant based on ratio threshold",
        ),
    )
    ax.scatter(
        df.loc[df["SigCat"] == "ratio_thresh", col_name1],
        df.loc[df["SigCat"] == "ratio_thresh", col_name2],
        **kwargs_r_sig,
    )

    if highlight is not None:
        _plot_highlights_scatter(highlight=highlight, kwargs_highlight=kwargs_highlight, df=df,
                                 x_colname=col_name1, y_colname=col_name2, ax=ax,
                                 pointsize_colname=pointsize_colname)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # ANNOTATION AND LABELING
    _label_scatter(df=df, ax=ax, x_colname=col_name1, y_colname=col_name2, annotate=annotate, highlight=highlight,
                   annotate_colname=annotate_colname, annotate_density=annotate_density)

    # STYLING
    _stylize_scatter(df, ax, show_legend, show_caption, title, pointsize_colname, pointsize_scaler)

    if show_thresh:
        _ratio_plot_style_axes(ax, ratio_thresh)
    if ret_fig:
        return fig


def _ratio_plot_style_axes(ax, ratio_thresh):
    ax.axvline(x=ratio_thresh, color="grey", linestyle="--", alpha=0.8)
    ax.axvline(x=-ratio_thresh, color="grey", linestyle="--", alpha=0.8)
    ax.axhline(y=ratio_thresh, color="grey", linestyle="--", alpha=0.8)
    ax.axhline(y=-ratio_thresh, color="grey", linestyle="--", alpha=0.8)
    ax.axhline(y=0, color="black", linestyle="-")
    ax.axvline(x=0, color="black", linestyle="-")


def iratio_plot(df: pd.DataFrame,
                col_name1: str,
                col_name2: str = None,
                ratio_thresh: float = None,
                xlabel: str = "Ratio col1",
                ylabel: str = "Ratio col2",
                pointsize_colname: str or float = None,
                highlight: pd.Index = None,
                title: str = None,
                show_legend: bool = True,
                ret_fig: bool = True,
                annotate_colname: str = "Gene names"):
    df, col_name1, col_name2, unsig, sig_ratio = _prep_ratio_data(df, col_name1, col_name2, ratio_thresh)

    categories = ["NS", "ratio_thresh"]

    if highlight is not None:
        if not isinstance(highlight, pd.Index):
            raise ValueError("You must provide a pd.Index object for highlighting")
        df["SigCat"] = "-"
        df.loc[highlight, "SigCat"] = "*"
        fig = (
            px.scatter(
                data_frame=df,
                x=col_name1,
                y=col_name2,
                hover_name=annotate_colname,
                size=pointsize_colname,
                color="SigCat",
                opacity=0.5,
                category_orders={"SigCat": ["-", "*"]},
                title=title,
            )
            if annotate_colname is not None
            else px.scatter(
                data_frame=df,
                x=col_name1,
                y=col_name2,
                size=pointsize_colname,
                color="SigCat",
                opacity=0.5,
                category_orders={"SigCat": ["-", "*"]},
                title=title,
            )
        )

    elif annotate_colname is not None:
        fig = px.scatter(
            data_frame=df,
            x=col_name1,
            y=col_name2,
            hover_name=annotate_colname,
            size=pointsize_colname,
            color="SigCat",
            opacity=0.5,
            category_orders={"SigCat": categories},
            title=title,
        )
    else:
        fig = px.scatter(
            data_frame=df,
            x=col_name1,
            y=col_name2,
            size=pointsize_colname,
            color="SigCat",
            opacity=0.5,
            category_orders={"SigCat": categories},
            title=title,
        )

    fig.update_yaxes(showgrid=False, zeroline=True)
    fig.update_xaxes(showgrid=False, zeroline=False)

    # horizontal threshold
    fig.add_trace(
        go.Scatter(
            x=[df[col_name1].min(), df[col_name1].max()],
            y=[ratio_thresh, ratio_thresh],
            mode="lines",
            line=go.scatter.Line(color="grey", dash="longdash"),
            showlegend=False,
        )
    )

    # vertical threshold
    fig.add_trace(
        go.Scatter(
            y=[df[col_name2].min(), df[col_name2].max()],
            x=[ratio_thresh, ratio_thresh],
            mode="lines",
            line=go.scatter.Line(color="grey", dash="longdash"),
            showlegend=False,
        )
    )

    fig.update_layout(
        template="simple_white",
        showlegend=show_legend,
        xaxis_title=xlabel,
        yaxis_title=ylabel
    )

    if ret_fig:
        return fig
    else:
        fig.show()


# Log Intensity Plots #
def log_int_plot(df, log_fc, log_intens_col, fct=None, annot=False,
                 sig_col="green", bg_col="lightgray", title="LogFC Intensity Plot",
                 figsize=(6, 6), ax: plt.axis = None, ret_fig: bool = False, legend: bool = True):
    # noinspection PyUnresolvedReferences
    r"""
    Draw a log-foldchange vs log-intensity plot.

    Parameters
    ----------
    ret_fig : bool
        Whether to return the figrue object, optional.
    ret_fig :  bool
        Whether to return the figure object.
    ax : plt.Axes
        The axis to plot on, optional.
    df : pd.DataFrame
        Input dataframe.
    log_fc : str
        Colname containing log fold-changes.
    log_intens_col : str
        Colname containing the log intensities.
    fct : float, optional
        fold change threshold at which an entry is deemed significant regulated.
        The default is None.
    annot : str, optional
        Which column to use for plot annotation. The default is False.
    sig_col : str, optional
        Colour for significant points. The default is "green".
    bg_col : str, optional
        Background colour. The default is "lightgray".
    title : str, optional
        Title for the plot.
        The default is "Volcano Plot".
    figsize : tuple of int, optional
        Size of the figure. The default is (6,6).
    legend: bool, optional
        Whether to add a legend. Default is True.
    Returns
    -------
    None.

    Examples
    --------
    The log_fc Intensity plot requires the log fold changes as calculated e.g.
    during t-test or LIMMA analysis and (log) intensities to separate points
    on the y axis.

    >>> autoprot.visualization.log_int_plot(prot_limma, "logFC_TvM",
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
        twitchVsmild = ['log2_Ratio H/M normalized BC18_1','log2_Ratio M/L normalized BC18_2',
                        'log2_Ratio H/M normalized BC18_3',
                        'log2_Ratio H/L normalized BC36_1','log2_Ratio H/M normalized BC36_2',
                        'log2_Ratio M/L normalized BC36_2']
        prot_limma = ana.limma(prot, twitchVsmild, cond="_TvM")
        prot["log10_Intensity BC4_3"].replace(-np.inf, np.nan, inplace=True)

        vis.log_int_plot(prot_limma, "logFC_TvM", "log10_Intensity BC4_3", fct=0.7, figsize=(15,5))

    Similar to the visualization using a volcano plot, points of interest can be
    selected and labelled.

    >>> autoprot.visualization.log_int_plot(prot_limma, "logFC_TvM", "log10_Intensity BC4_3",
                   fct=2, annot=True,  annot="Gene names")

    .. plot::
        :context: close-figs

        vis.log_int_plot(prot_limma, "logFC_TvM", "log10_Intensity BC4_3",
                       fct=2, annot="Gene names")
    """
    # TODO: Copy features from volcano function (highlight etc)
    # TODO also add option to not highlight anything
    df = df.copy(deep=True)

    df = df[~df[log_intens_col].isin([-np.inf, np.nan])]
    df["SigCat"] = "-"
    if fct is not None:
        df.loc[abs(df[log_fc]) > fct, "SigCat"] = "*"
    unsig = df[df["SigCat"] == "-"].index
    sig = df[df["SigCat"] == "*"].index

    # draw figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.scatter(df[log_fc].loc[unsig], df[log_intens_col].loc[unsig], color=bg_col, alpha=.75, s=5,
               label="background")
    ax.scatter(df[log_fc].loc[sig], df[log_intens_col].loc[sig], color=sig_col, label="POI")

    # draw threshold lines
    if fct:
        ax.axvline(fct, 0, 1, ls="dashed", color="lightgray")
        ax.axvline(-fct, 0, 1, ls="dashed", color="lightgray")
    ax.axvline(0, 0, 1, ls="dashed", color="gray")

    # remove of top and right plot boundary
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # setting x and y labels and title
    ax.set_ylabel("log Intensity")
    ax.set_xlabel("log_fc")
    ax.set_title(title, size=18)

    if legend:
        # add legend
        ax.legend()

    if annot:
        # Annotation
        # get x and y coordinates as well as strings to plot
        xs = df[log_fc].loc[sig]
        ys = df[log_intens_col].loc[sig]
        ss = df[annot].loc[sig]

        # annotation
        for idx, (x, y, s) in enumerate(zip(xs, ys, ss)):
            if idx % 2 == 0:
                if x < 0:
                    ax.plot([x, x - .2], [y, y - .2], color="gray")
                    ax.text(x - .3, y - .25, s)
                else:
                    ax.plot([x, x + .2], [y, y - .2], color="gray")
                    ax.text(x + .2, y - .2, s)

            elif x < 0:
                ax.plot([x, x - .2], [y, y + .2], color="gray")
                ax.text(x - .3, y + .25, s)
            else:
                ax.plot([x, x + .2], [y, y + .2], color="gray")
                ax.text(x + .2, y + .2, s)

    if ret_fig:
        return fig


def ilog_int_plot(df, log_fc, log_intens_col, fct=None, annot=False, ret_fig=False):
    # noinspection PyUnresolvedReferences
    # TODO: Copy features from volcano function (highlight etc)
    # TODO also add option to not highlight anything
    df = df.copy(deep=True)

    df = df[~df[log_intens_col].isin([-np.inf, np.nan])]
    df["SigCat"] = "-"
    if fct is not None:
        df.loc[abs(df[log_fc]) > fct, "SigCat"] = "*"

    if annot:
        fig = px.scatter(data_frame=df, x=log_fc, y=log_intens_col, hover_name=annot,
                         color="SigCat", color_discrete_sequence=["cornflowerblue", "mistyrose"],
                         opacity=0.5, category_orders={"SigCat": ["*", "-"]}, title="Volcano plot")
    else:
        fig = px.scatter(data_frame=df, x=log_fc, y=log_intens_col,
                         color="SigCat", color_discrete_sequence=["cornflowerblue", "mistyrose"],
                         opacity=0.5, category_orders={"SigCat": ["*", "-"]}, title="Volcano plot")

    fig.update_yaxes(showgrid=False, zeroline=True)
    fig.update_xaxes(showgrid=False, zeroline=False)

    fig.add_trace(
        go.Scatter(
            x=[0, 0],
            y=[0, df[log_intens_col].max()],
            mode="lines",
            line=go.scatter.Line(color="purple", dash="longdash"),
            showlegend=False)
    )

    fig.add_trace(
        go.Scatter(
            x=[-fct, -fct],
            y=[0, df[log_intens_col].max()],
            mode="lines",
            line=go.scatter.Line(color="teal", dash="longdash"),
            showlegend=False)
    )

    fig.add_trace(
        go.Scatter(
            x=[fct, fct],
            y=[0, df[log_intens_col].max()],
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


# MA Plots #

def _init_ma_plot(df: pd.DataFrame, x: str, y: str, fct: Union[float, int]):
    df = df.copy(deep=True)
    df["M"] = df[x] - df[y]
    df["A"] = 1 / 2 * (df[x] + df[y])
    df["M"].replace(-np.inf, np.nan, inplace=True)
    df["A"].replace(-np.inf, np.nan, inplace=True)
    df["SigCat"] = False
    if fct is not None:
        df.loc[abs(df["M"]) > fct, "SigCat"] = True

    return df


def ma_plot(df: pd.DataFrame, x: str, y: str, fct: Union[float, int] = None,
            title: str = "MA Plot", ax: plt.axis = None, ret_fig: bool = False, figsize: tuple = (6, 6)):
    # noinspection PyUnresolvedReferences
    r"""
    Plot log intensity ratios (M) vs. the average intensity (A).

    Notes
    -----
    The MA plot is useful to determine whether a data normalization is needed.
    The majority of proteins is considered to be unchanged between
    treatments and thereofore should lie on the y=0 line.
    If this is not the case, a normalization should be applied.

    Parameters
    ----------
    ret_fig: bool
        Whether to return the figure or show it directly
    ax : plt.axis
        The axis to plot on
    df : pd.dataFrame
        Input dataframe with log intensities.
    x : str
        Colname containing intensities of experiment1.
    y : str
        Colname containing intensities of experiment2.
    fct : numeric, optional
        The value in M to draw a horizontal line.
        The default is None.
    title : str, optional
        Title of the figure. The default is "MA Plot".
    figsize : tuple of int, optional
        Size of the figure. The default is (6,6).

    Returns
    -------
    None.

    Examples
    --------
    The MA plot allows to easily visualize difference in intensities between
    experiments or replicates and therefore to judge if data normalization is
    required for further analysis.
    The majority of intensities should be unchanged between conditions and
    therefore most points should lie on the y=0 line.

    >>> autoprot.visualization.ma_plot(prot, twitch, ctrl, fct=2)

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

        vis.ma_plot(prot, x, y, fct=2)
        plt.show()

    If this is not the case, a normalization using e.g. LOESS should be applied

    >>> autoprot.visualization.ma_plot(prot, twitch, ctrl, fct=2)

    .. plot::
        :context: close-figs

        twitch = "log10_Intensity H BC18_1"
        ctrl = "log10_Intensity L BC18_1"

        vis.ma_plot(prot, twitch, ctrl, fct=2)
        plt.show()
    """
    df = _init_ma_plot(df, x, y, fct)

    # draw figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    sns.scatterplot(data=df, x='A', y='M', linewidth=0, hue="SigCat")
    plt.axhline(0, 0, 1, color="black", ls="dashed")
    plt.title(title)
    plt.ylabel("M")
    plt.xlabel("A")

    if fct is not None:
        plt.axhline(fct, 0, 1, color="gray", ls="dashed")
        plt.axhline(-fct, 0, 1, color="gray", ls="dashed")

    if ret_fig:
        return fig


def ima_plot(df, x, y, fct=None, title="MA Plot", annot=None):
    # sourcery skip: assign-if-exp, extract-method
    # noinspection PyUnresolvedReferences
    r"""
    Plot log intensity ratios (M) vs. the average intensity (A).

    Notes
    -----
    The MA plot is useful to determine whether a data normalization is needed.
    The majority of proteins is considered to be unchanged between
    treatments and therefore should lie on the y=0 line.
    If this is not the case, a normalization should be applied.

    Parameters
    ----------
    df : pd.dataFrame
        Input dataframe with log intensities.
    x : str
        Column name containing intensities of experiment1.
    y : str
        Column name containing intensities of experiment2.
    fct : numeric, optional
        The value in M to draw a horizontal line.
        The default is None.
    title : str, optional
        Title of the figure. The default is "MA Plot".
    annot : str, optional
        Column name to use for labels in interactive plot.
        The default is None.

    Returns
    -------
    None.
    """
    df = _init_ma_plot(df, x, y, fct)

    if annot:
        fig = px.scatter(data_frame=df, x='A', y='M', hover_name=annot,
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


# Mean SD plot #
def mean_sd_plot(df, reps):
    # noinspection PyUnresolvedReferences
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

    >>> autoprot.visualization.mean_sd_plot(prot, twitchInt)

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

        vis.mean_sd_plot(prot, twitchInt)
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
    p.ax_marg_x.set_title("Mean SD plot")


# Traces #
def plot_traces(df, cols: list, labels: list[str] = None, colors: list[str] = None, z_score: int = None,
                xlabel: str = "", ylabel: str = "log_fc", title: str = "", ax: plt.axis = None,
                plot_summary: bool = False, plot_summary_only: bool = False, summary_color: str = "red",
                summary_type: Literal["Mean", "Median"] = "Mean", summary_style: str = "solid", **kwargs):
    # noinspection PyUnresolvedReferences
    r"""
    Plot numerical data such as fold changes vs. columns (e.g. conditions).

    Parameters
    ----------
    df : pd.DataFame
        Input dataframe.
    cols : list
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
        Label for the x-axis. The default is "".
    ylabel : str, optional
        Label for the y-axis.
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
    >>> vis.plot_traces(test, test.columns, labels=label, colors=["red", "green"]*5,
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
        phos = ana.ttest(df=phos_expanded, reps=twitchVsmild, cond="_TvM", return_fc=True)
        phos = ana.ttest(df=phos_expanded, reps=twitchVsctrl, cond="_TvC", return_fc=True)
        phos = ana.ttest(df=phos_expanded, reps=twitchVsmild, cond="_MvC", return_fc=True)

        idx = phos.sample(10).index
        test = phos.filter(regex="logFC_").loc[idx]
        label = phos.loc[idx, "Gene names"]
        vis.plot_traces(test, test.columns, labels=label, colors=["red", "green"]*5,
                       xlabel='Column')
        plt.show()

    """
    # TODO Add parameter to plot yerr
    # TODO xlabels from colnames
    x = range(len(cols))
    y = df[cols].T.values
    if z_score is not None and z_score in {0, 1}:
        y = zscore(y, axis=z_score)

    if ax is None:
        plt.figure()
        ax = plt.subplot()
    ax.set_title(title)
    f = []
    if not plot_summary_only:
        if colors is None:
            f = ax.plot(x, y, **kwargs)
        else:
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


# p Value Histograms #
def pval_hist(df, ps, adj_ps, title=None, alpha=0.05, zoom=20):
    # noinspection PyUnresolvedReferences
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

    >>> autoprot.visualization.bh_plot(phos,'pValue_TvC', 'adj.pValue_TvC', alpha=0.05, zoom=7)

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
        phos = pp.filter_loc_prob(phos, thresh=.75)
        phosRatio = phos.filter(regex="log2_Ratio .\/.( | normalized )R.___").columns
        phos = pp.remove_non_quant(phos, phosRatio)

        phosRatio = phos.filter(regex="log2_Ratio .\/. normalized R.___").columns
        phos_expanded = pp.expand_site_table(phos, phosRatio)

        mildVsctrl = ["log2_Ratio M/L normalized R1","log2_Ratio H/L normalized R2","log2_Ratio M/L normalized R3",
                      "log2_Ratio H/M normalized R4","log2_Ratio M/L normalized R5","log2_Ratio H/L normalized R6"]

        phos = ana.ttest(df=phos_expanded, reps=mildVsctrl, cond="_MvC", return_fc=True)

        vis.bh_plot(phos,'pValue_MvC', 'adj.pValue_MvC', alpha=0.05, zoom=7)
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

    ax[1].plot(x[:zoom], y[:zoom], color='gray')
    ax[1].scatter(x[:zoom], df[ps].loc[idx].sort_values().iloc[:zoom], label="p_values", color="teal")
    ax[1].scatter(x[:zoom], df[adj_ps].loc[idx][:zoom], label="adj. p_values", color="purple")

    sns.despine(ax=ax[0])
    sns.despine(ax=ax[1])


# Upset PLot #
class UpSetGrouped(upsetplot.UpSet):
    # noinspection PyUnresolvedReferences
    """
    Generate upset plot as described in Lex2014 [1] and implemented in Python by jnothman.
    This function extends its use by the ability to colour and group bars in the bar plot.

    Notes
    -----
    This function uses the upset plot implementation of jnothman (https://github.com/jnothman/UpSetPlot) and extends it
    by grouping and colouring of the bar charts.
    Most troubleshooting can be accomplished looking at their documentation at
    https://upsetplot.readthedocs.io/en/stable/index.html. Especially their documentation of how to integrate
    commonly used data formats is very helpful: https://upsetplot.readthedocs.io/en/stable/formats.html.

    References
    ----------
    [1] Alexander Lex, Nils Gehlenborg, Hendrik Strobelt, Romain Vuillemot, Hanspeter Pfister,
    UpSet: Visualization of Intersecting Sets, IEEE Transactions on Visualization and Computer Graphics (InfoVis
    '14), vol. 20, no. 12, pp. 1983–1992, 2014. doi: doi.org/10.1109/TVCG.2014.2346248

    Examples
    --------

    Here we first generate a dummy data set akin to phosphoproteomics results of up- and downregulated sites that
    is then used to showcase the usage of the UpSetGrouped class.

    >>> example
    120_down  60_down  30_down  10_down  120_up  60_up  30_up  10_up
    False     False    False    False    False   False  True   False    106
                                                        False  True      85
                                                 True   False  False     50
                       True     False    False   False  False  False     30
                       False    False    True    False  False  False     29
                                True     False   False  False  False     26
    True      False    False    False    False   False  False  False     17
    False     True     False    False    False   False  False  False     14
              False    False    False    False   False  True   True      94
                                         True    True   False  False     33
                                         False   True   True   False     30
    True      True     False    False    False   False  False  False     19
    False     True     True     False    False   False  False  False     13
    True      False    True     False    False   False  False  False      9
    False     False    True     True     False   False  False  False      7
                       False    False    False   True   False  True       5
                                         True    False  True   False      4
                                                        False  True       2
              True     False    True     False   False  False  False      1
              False    True     False    True    False  False  False      1
                       False    True     True    False  False  False      1
    True      False    False    False    False   False  False  True       1
    False     False    False    False    True    True   True   False    102
                                         False   True   True   True      29
                                         True    False  True   True      14
    True      True     True     False    False   False  False  False     11
    False     True     True     True     False   False  False  False      1
              False    False    False    True    True   True   True      60
    True      True     True     True     False   False  False  False      3
    dtype: int64

    The example dataset is loaded into the UpSetGrouped class and the styling_helper is used to colour the up- and down-
    regulated groups. Note that the styling_helper operates on the basis of string comparison with the titles of the
    data frame categories.
    Also note that the replot_totals function can only be used after calling the plot-function as is requires the
    axis on which the original bar plot was drawn.

    >>> upset = UpSetGrouped(example,
    ...                      show_counts=True,
    ...                      #show_percentages=True,
    ...                      sort_by=None,
    ...                      sort_categories_by='cardinality',
    ...                      facecolor="gray")
    >>> upset.styling_helper('up', facecolor='darkgreen', label='up regulated')
    >>> upset.styling_helper('down', facecolor='darkblue', label='down regulated')
    >>> upset.styling_helper(['up', 'down'], facecolor='darkred', label='reversibly regulated')
    >>> specs = upset.plot()
    >>> upset.replot_totals(specs=specs, color=['darkgreen',
    ...                                         'darkgreen',
    ...                                         'darkgreen',
    ...                                         'darkgreen',
    ...                                         'darkblue',
    ...                                         'darkblue',
    ...                                         'darkblue',
    ...                                         'darkblue',])
    >>> plt.show()

    .. plot::
        :context: close-figs

        import pandas as pd
        import autoprot.visualization as vis

        arrays = [(False,False,False,False,False,False,True,False),
                  (False,False,False,False,False,False,False,True),
                  (False,False,False,False,False,True,False,False),
                  (False,False,True,False,False,False,False,False),
                  (False,False,False,False,True,False,False,False),
                  (False,False,False,True,False,False,False,False),
                  (True,False,False,False,False,False,False,False),
                  (False,True,False,False,False,False,False,False),
                  (False,False,False,False,False,False,True,True),
                  (False,False,False,False,True,True,False,False),
                  (False,False,False,False,False,True,True,False),
                  (True,True,False,False,False,False,False,False),
                  (False,True,True,False,False,False,False,False),
                  (True,False,True,False,False,False,False,False),
                  (False,False,True,True,False,False,False,False),
                  (False,False,False,False,False,True,False,True),
                  (False,False,False,False,True,False,True,False),
                  (False,False,False,False,True,False,False,True),
                  (False,True,False,True,False,False,False,False),
                  (False,False,True,False,True,False,False,False),
                  (False,False,False,True,True,False,False,False),
                  (True,False,False,False,False,False,False,True),
                  (False,False,False,False,True,True,True,False),
                  (False,False,False,False,False,True,True,True),
                  (False,False,False,False,True,False,True,True),
                  (True,True,True,False,False,False,False,False),
                  (False,True,True,True,False,False,False,False),
                  (False,False,False,False,True,True,True,True),
                  (True,True,True,True,False,False,False,False)]
        arrays = np.array(arrays).T
        values = (106,85,50,30,29,26,17,14,94,33,30,19,13,9,7,5,4,2,1,1,1,1,102,29,14,11,1,60,3)
        example = pd.Series(values,
                            index=pd.MultiIndex.from_arrays(arrays,
                                                            names=('120_down',
                                                                   '60_down',
                                                                   '30_down',
                                                                   '10_down',
                                                                   '120_up',
                                                                   '60_up',
                                                                   '30_up',
                                                                   '10_up')
                                                           )
                           )

        upset = vis.UpSetGrouped(example,
                                 show_counts=True,
                                 #show_percentages=True,
                                 sort_by=None,
                                 sort_categories_by='cardinality',
                                 facecolor="gray")
        upset.styling_helper('up', facecolor='darkgreen', label='up regulated')
        upset.styling_helper('down', facecolor='darkblue', label='down regulated')
        upset.styling_helper(['up', 'down'], facecolor='darkred', label='reversibly regulated')
        specs = upset.plot()
        upset.replot_totals(specs=specs, color=['darkgreen',
                                                'darkgreen',
                                                'darkgreen',
                                                'darkgreen',
                                                'darkblue',
                                                'darkblue',
                                                'darkblue',
                                                'darkblue',])

        plt.show()

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def styling_helper(self, label_substrings, mode='intersection', **kwargs):
        """
        Helper function for styling upsetplot category plots.

        Parameters
        --------
        label_substrings : str or list
            Substrings that are contained in the names of the row
            indices of the UpSet data (e.g. '_up')
        mode : "union" or "intersection"
            Whether bars containing with index
            names corresponding to either of the labels (union) or only combinations that contain both labels (
            intersection) should be treated
        kwargs : As passed to UpSet.style_subsets
        """
        labels = self.intersections.index.names

        if isinstance(label_substrings, str):
            label_substrings = [label_substrings, ]

        if mode not in ['union', 'intersection']:
            raise AttributeError('Please provide either "union" or "intersection" as argument to mode')

        for L in range(len(labels) + 1):
            for subset in combinations(labels, L):

                label_found = []
                for label in label_substrings:
                    tests = [label in s for s in subset]
                    label_found.append(any(tests))

                if (
                        mode == 'intersection'
                        and all(label_found)
                        or mode != 'intersection'
                        and any(label_found)
                ):
                    self.style_subsets(present=subset, **kwargs)

    def replot_totals(self, specs, color):
        """
        Plot bars indicating total set size.

        Parameters
        ----------
        specs : dict
            Dict of object axes as returned by upset.plot()
        color : str or list of str
            Color(s) of the bars.
        """

        for artist in specs['totals'].lines + specs['totals'].collections:
            artist.remove()

        orig_ax = specs['totals']
        ax = self._reorient(specs['totals'])
        rects = ax.barh(np.arange(len(self.totals.index.values)), self.totals,
                        .5, color=color, align='center')

        ax.set_yticklabels(specs['matrix'].get_yticklabels())

        self._label_sizes(ax, rects, 'left' if self._horizontal else 'top')

        max_total = self.totals.max()
        if self._horizontal:
            orig_ax.set_xlim(max_total, 0)
        for x in ['top', 'left', 'right']:
            ax.spines[self._reorient(x)].set_visible(False)
        ax.yaxis.set_visible(False)
        ax.xaxis.grid(True)
        ax.yaxis.grid(False)
        ax.patch.set_visible(False)
