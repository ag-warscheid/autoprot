# -*- coding: utf-8 -*-
"""
Autoprot Quality Control Plotting Functions.

@author: Wignand, Julian, Johannes

@documentation: Julian
"""

import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
import pylab as pl

import plotly.express as px

# noinspection PyUnresolvedReferences
from autoprot.dependencies.venn import venn
# noinspection PyUnresolvedReferences
from autoprot import visualization as vis
from typing import Literal, Union


def _bar_plot_style(df, ax):
    """
    Style the secondary y-axis for bar plots
    """
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


# STY COUNT PLOT ##
def sty_count_plot(df: pd.DataFrame, figsize: tuple = (12, 8), chart_type: Literal['bar', 'pie'] = "bar",
                   ret_fig: bool = False, ax: Union[plt.axis, None] = None, **kwargs):
    # sourcery skip: extract-method
    # noinspection PyUnresolvedReferences
    r"""
    Draw an overview of Number of Phospho (STY) of a Phospho(STY) file.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
        Must contain a column "Number of Phospho (STY)".
    figsize : tuple of int, optional
        Figure size. The default is (12,8).
    chart_type : str, optional
        'bar' or 'pie'. The default is "bar".
    ret_fig : bool, optional
        Whether to return the figure. The default is False.
    ax : matplotlib axis
        Axis to plot on
    **kwargs:
        Keyword arguments passed to sns.countplot or plt.pie

    Returns
    -------
    fig : matplotlib.figure
        The figure object.

    Examples
    --------
    Plot a bar chart of the distribution of the number of phosphosites on the peptides.

    >>> autoprot.visualization.sty_count_plot(phos, chart_type="bar")
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
        vis.sty_count_plot(phos, chart_type="bar")
        plt.show()

    """
    no_of_phos = [int(i) for i in
                  list(pl.flatten([str(i).split(';') for i in df["Number of Phospho (STY)"].fillna(0)]))]
    count = [(no_of_phos.count(i), i) for i in set(no_of_phos)]
    counts_perc = [(round(no_of_phos.count(i) / len(no_of_phos) * 100, 2), i) for i in set(no_of_phos)]

    print("Number of phospho (STY) [total] - (count / # Phospho)")
    print(count)
    print("Percentage of phospho (STY) [total] - (% / # Phospho)")
    print(counts_perc)
    df = pd.DataFrame(no_of_phos, columns=["Number of Phospho (STY)"])

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
    else:
        fig = ax.get_figure()

    if chart_type == "bar":
        sns.countplot(x="Number of Phospho (STY)", data=df, ax=ax, **kwargs)
        plt.title('Number of Phospho (STY)')
        plt.xlabel('Number of Phospho (STY)')
        _bar_plot_style(df, ax)

    elif chart_type == "pie":
        ax.pie([i[0] for i in count], labels=[i[1] for i in count], **kwargs)
        ax.set_title("Number of Phosphosites")
    else:
        raise TypeError("typ must be either 'bar' or 'pie")

    if ret_fig is True:
        return fig


def isty_count_plot(df: pd.DataFrame, chart_type: Literal['bar', 'pie'] = "bar", ret_fig: bool = False, **kwargs):
    # sourcery skip: extract-method
    # noinspection PyUnresolvedReferences
    r"""
    Draw an interactive overview of Number of Phospho (STY) of a Phospho(STY) file.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
        Must contain a column "Number of Phospho (STY)".
    chart_type : str, optional
        'bar' or 'pie'. The default is "bar".
    ret_fig : bool, optional
        Whether to return the figure. The default is False.
    **kwargs:
        Keyword arguments passed to plotly

    Returns
    -------
    fig : matplotlib.figure
        The figure object.

    """
    no_of_phos = [int(i) for i in
                  list(pl.flatten([str(i).split(';') for i in df["Number of Phospho (STY)"].fillna(0)]))]
    count = [(no_of_phos.count(i), i) for i in set(no_of_phos)]
    counts_perc = [(round(no_of_phos.count(i) / len(no_of_phos) * 100, 2), i) for i in set(no_of_phos)]

    print("Number of phospho (STY) [total] - (count / # Phospho)")
    print(count)
    print("Percentage of phospho (STY) [total] - (% / # Phospho)")
    print(counts_perc)
    df = pd.DataFrame(no_of_phos, columns=["Count"]).value_counts().reset_index(name='Number of Phospho (STY)')
    df = df.sort_index()

    if chart_type == "bar":
        fig = px.bar(df, x='Count', y="Number of Phospho (STY)", **kwargs)
    elif chart_type == "pie":
        fig = px.pie(df, names='Count', values='Number of Phospho (STY)', **kwargs)
    else:
        raise TypeError("typ must be either 'bar' or 'pie")

    if ret_fig is True:
        return fig

    fig.show()


# CHARGE PLOT #
# noinspection PyUnboundLocalVariable
def charge_plot(df: pd.DataFrame, figsize: tuple = (12, 8), chart_type: Literal['bar', 'pie'] = "bar",
                ret_fig: bool = False, ax: Union[plt.axis, None] = None, **kwargs):
    # noinspection PyUnresolvedReferences
    r"""
    Plot a pie chart of the peptide charges of a phospho(STY) dataframe.

    Parameters
    ----------
    df : pd.Dataframe
        Input dataframe.
        Must contain a column named "Charge".
    figsize : tuple of int, optional
        The size of the figure. The default is (12,8).
    chart_type : str, optional
        "pie" or "bar".
        The default is "bar".
    ret_fig : bool, optional
        Whether to return the figure.
        The default is False.
    ax : matplotlib axis
        Axis to plot on
    **kwargs:
        Keyword arguments passed to sns.countplot or plt.pie

    Returns
    -------
    fig : matplotlib.figure
        The figure object.

    Examples
    --------
    Plot the charge states of a dataframe.

    >>> autoprot.visualization.charge_plot(phos, chart_type="pie")
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
        vis.charge_plot(phos, chart_type="pie")
        plt.show()
    """

    df = df.copy(deep=True)
    no_of_phos = [int(i) for i in list(pl.flatten([str(i).split(';') for i in df["Charge"].fillna(0)]))]
    count = [(no_of_phos.count(i), i) for i in set(no_of_phos)]
    counts_perc = [(round(no_of_phos.count(i) / len(no_of_phos) * 100, 2), i) for i in set(no_of_phos)]

    print("charge [total] - (count / # charge)")
    print(count)
    print("Percentage of charge [total] - (% / # charge)")
    print(counts_perc)
    df = pd.DataFrame(no_of_phos, columns=["charge"])

    if chart_type == "bar":
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.gca()

        sns.countplot(x="charge", data=df, ax=ax, **kwargs)
        plt.title('charge')
        plt.xlabel('charge')
        _bar_plot_style(df, ax)
    elif chart_type == "pie":
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.gca()
        ax.pie([i[0] for i in count], labels=[i[1] for i in count], **kwargs)
        ax.set_title("charge")
    if not ret_fig:
        return
    return fig


def icharge_plot(df: pd.DataFrame, chart_type: Literal['bar', 'pie'] = "bar", ret_fig: bool = False, **kwargs):
    # noinspection PyUnresolvedReferences
    r"""
    Plot a pie chart of the peptide charges of a phospho(STY) dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
        Must contain a column "charge".
    chart_type : str, optional
        'bar' or 'pie'. The default is "bar".
    ret_fig : bool, optional
        Whether to return the figure object. The default is False.
    **kwargs:
        Keyword arguments passed to plotly

    Returns
    -------
    fig : plotly.figure
        The figure object.
    """

    df = df.copy(deep=True)
    no_of_phos = [int(i) for i in list(pl.flatten([str(i).split(';') for i in df["Charge"].fillna(0)]))]
    count = [(no_of_phos.count(i), i) for i in set(no_of_phos)]
    counts_perc = [(round(no_of_phos.count(i) / len(no_of_phos) * 100, 2), i) for i in set(no_of_phos)]

    print("charge [total] - (count / # charge)")
    print(count)
    print("Percentage of charge [total] - (% / # charge)")
    print(counts_perc)

    df = pd.DataFrame(no_of_phos, columns=["charge"]).value_counts().reset_index(name='charge')
    df = df.sort_index()

    if chart_type == "bar":
        fig = px.bar(df, x='Count', y="charge", **kwargs)

    elif chart_type == "pie":
        fig = px.pie(df, names='Count', values='charge', **kwargs)
    else:
        raise ValueError("typ must be either 'bar' or 'pie")

    if ret_fig is True:
        return fig

    fig.show()


# COUNT MODIFIED AMINO ACIDS #
def count_mod_aa(df: pd.DataFrame, figsize: tuple = (6, 6), ret_fig: bool = False, ax: Union[plt.axis, None] = None,
                 **kwargs):
    # noinspection PyUnresolvedReferences
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
    ax : matplotlib axis
        Axis to plot on
    **kwargs:
        Keyword arguments passed to plt.pie

    Returns
    -------
    fig : matplotlib.figure
        The figure object.

    Examples
    --------
    Plot pie chart of modified amino acids.

    >>> autoprot.visualization.count_mod_aa(phos)

    .. plot::
        :context: close-figs

        import autoprot.preprocessing as pp
        import autoprot.visualization as vis
        import pandas as pd

        phos = pd.read_csv("_static/testdata/Phospho (STY)Sites_mod.zip", sep="\t", low_memory=False)
        phos = pp.cleaning(phos, file = "Phospho (STY)")
        vis.count_mod_aa(phos)
        plt.show()

    """
    srs = df["Amino acid"].value_counts()
    labels = [str(i) + '\n' + str(round(j / srs.sum() * 100, 2)) + '%' for i, j in zip(srs.index, srs.values)]

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
    else:
        fig = ax.get_figure()

    ax.pie(srs.values,
           labels=labels,
           **kwargs)
    ax.set_title("Modified AAs")

    if ret_fig:
        return fig


def icount_mod_aa(df: pd.DataFrame, ret_fig: bool = False, **kwargs):
    # noinspection PyUnresolvedReferences
    r"""
    Count the number of modifications per amino acid.

    Parameters
    ----------
    df : pd.Dataframe
        The input dataframe.
        Must contain a column "Amino acid".
    ret_fig : bool, optional
        Whether to return the figure object. The default is False.
    **kwargs:
        Keyword arguments passed to plotly

    Returns
    -------
    fig : plotly.figure
        The figure object.
    """
    srs = df["Amino acid"].value_counts()
    labels = [str(i) + '\n' + str(round(j / srs.shape[0] * 100, 2)) + '%' for i, j in zip(srs.index, srs.values)]
    srs.index.name = 'Amino Acid'
    srs.name = 'Count'

    fig = px.pie(srs, names=srs.index, values='Count', title="Modified AAs", custom_data=[labels], **kwargs)

    fig.update_traces(hovertemplate="%{customdata[0]}")

    if ret_fig:
        return fig

    fig.show()
