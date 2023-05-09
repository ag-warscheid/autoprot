# -*- coding: utf-8 -*-
"""
Autoprot Annotation Functions.

@author: Wignand, Julian, Johannes

@documentation: Julian
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

import logomaker
import matplotlib.patches as patches

# noinspection PyUnresolvedReferences
from autoprot.dependencies.venn import venn
# noinspection PyUnresolvedReferences
from autoprot import visualization as vis
from autoprot.dependencies.plotlylogo.PlotlyLogo import logo as plogo

import plotly.graph_objects as go


# SEQUENCE LOGO

def _find_sequence_motif(row: pd.Series, sequence_motif: str, rename_to_st=False):
    """
    Return the input sequence_motif if it fits to the value provided in "Sequence window" of a dataframe row.

    Parameters
    ----------
    row : pd.Series
        Pandas dataframe row containing the identified sequence windows.
    sequence_motif : str
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
    import re
    # identified sequence window
    d = row["Sequence window"]
    # In Sequence window the aa of interest is always at pos 15
    # This loop will check if the sequence_motif we are interested in is
    # centered with its phospho residue at pos 15 of the sequence window
    pos1 = None
    for idx, i in enumerate(sequence_motif):
        # the phospho residue in the sequence_motif is indicated by lowercase character
        if i.islower():
            # pos1 is position of the phospho site in the sequence_motif
            pos1 = len(sequence_motif) - idx
    if pos1 is None:
        raise ValueError("Phospho residue has to be lower case!")
    # pos2 is the last position of the matched sequence
    # the MQ Sequence window is always 30 AAs long and centred on the modified
    # amino acid. Hence, for a true hit, pos2-pos1 should be 15
    if isinstance(d, str):  # only consider searchable strings, not NaN
        exp = (
            sequence_motif[: pos1 - 1] + "(S|T)" + sequence_motif[pos1:]
            if rename_to_st  # if phospho site is to be renamed
            else sequence_motif.upper()  # else keep the original sequence
        )
        if pos2 := re.search(exp.upper(), d):
            pos2 = pos2.end()
            pos = pos2 - pos1
            if pos == 15:
                return sequence_motif


def _generate_kinase_motif_df(seq: list):
    """
    Generate a dataframe with relative frequencies for amino acids.

    Parameters
    ----------
    seq : list of str
        List of experimentally determined sequences matching the sequence_motif.

    Returns
    -------
    pd.Dataframe : kinase_motif_df
    """
    aa_dic = dict(G=0, P=0, A=0, V=0, L=0, I=0, M=0, C=0, F=0, Y=0, W=0, H=0, K=0, R=0, Q=0, N=0, E=0, D=0, S=0,
                  T=0)

    seq = [i for i in seq if len(i) == 15]
    seq_t = [''.join(s) for s in zip(*seq)]
    score_matrix = []
    for pos in seq_t:
        d = aa_dic.copy()
        for aa in pos:
            aa = aa.upper()
            if aa not in ['.', '-', '_', "X"]:
                d[aa] += 1
        score_matrix.append(d)

    for pos in score_matrix:
        for k in pos.keys():
            pos[k] /= len(seq)

    # empty array -> (sequenceWindow, aa)
    m = np.empty((15, 20))
    for i in range(m.shape[0]):
        x = list(score_matrix[i].values())
        m[i] = x

    # create Logo object
    kinase_motif_df = pd.DataFrame(m).fillna(0)
    kinase_motif_df.columns = aa_dic.keys()

    return kinase_motif_df


def sequence_logo(df, motif, file=None, rename_to_st=False):
    # noinspection PyUnresolvedReferences
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
    >>> autoprot.visualization.sequence_logo(phos, sequence_motif)

    allow s and t as central residue

    >>> autoprot.visualization.sequence_logo(phos, sequence_motif, path, rename_to_st=True)

    """

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

        kinase_motif_df = _generate_kinase_motif_df(seq)

        k_logo = logomaker.Logo(kinase_motif_df,
                                font_name="Arial",
                                color_scheme="dmslogo_funcgroup",
                                vpad=0,
                                width=.8)

        k_logo.highlight_position(p=7, color='purple', alpha=.5)
        plt.title(f"{sequence_motif} SequenceLogo")

        # generate x labels corresponding to sequence indices
        k_logo.ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15])
        k_logo.ax.set_xticklabels(labels=[-7, -5, -3, -1, 1, 3, 5, 7])
        sns.despine()
        if outfile_path is not None:
            plt.savefig(outfile_path)

    # init empty col corresponding to sequence sequence_motif
    df[motif[0]] = np.nan
    # returns the input sequence sequence_motif for rows where the sequence_motif fits the sequence
    # window
    df[motif[0]] = df.apply(lambda row: _find_sequence_motif(row, motif[0], rename_to_st), axis=1)

    if file is not None:
        # consider only the +- 7 amino acids around the modified residue (x[8:23])
        generate_sequence_logo(df["Sequence window"][df[motif[0]].notnull()].apply(lambda x: x[8:23]),
                               outfile_path=file + "/{}_{}.svg".format(motif[0], motif[1]),
                               sequence_motif="{} - {}".format(motif[0], motif[1]))
    else:
        generate_sequence_logo(df["Sequence window"][df[motif[0]].notnull()].apply(lambda x: x[8:23]),
                               sequence_motif="{} - {}".format(motif[0], motif[1]))


def isequence_logo(df, motif, rename_to_st=False, ret_fig=False):
    """
    Plot interactive sequence logo

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe from which experimentally determined sequences are extracted.
    motif : tuple of str
        A tuple of the sequence_motif and its name.
        The phospho site residue in the sequence_motif should be indicated by a
        lowercase character.
        Example ("..R.R..s.......", "MK_down").
    rename_to_st : bool, optional
        If true, the phospho residue will be considered to be
        either S or T. The default is False.
    ret_fig : bool, optional
        Whether to return the figure object.

    Returns
    -------
    plotly.figure: The interactive figure object
    """

    def interactive_sequence_logo(seq: list, sequence_motif: str = ""):
        """
        Draw a sequence logo plot for a sequence_motif.

        Parameters
        ----------
        seq : list of str
            List of experimentally determined sequences matching the sequence_motif.
        sequence_motif : str, optional
            The sequence_motif used to find the sequences.
            The default is "".

        Returns
        -------
        None.
        """
        kinase_motif_df = _generate_kinase_motif_df(seq)
        interactive_logo = plogo.logo(kinase_motif_df, return_fig=True)
        interactive_logo.update_layout(title=f"{sequence_motif} SequenceLogo",
                                       margin=dict(t=50)  # add margin to accommodate title
                                       )

        return interactive_logo

    # init empty col corresponding to sequence sequence_motif
    df[motif[0]] = np.nan
    # returns the input sequence sequence_motif for rows where the sequence_motif fits the sequence
    # window
    df[motif[0]] = df.apply(lambda row: _find_sequence_motif(row, motif[0], rename_to_st), axis=1)

    fig = interactive_sequence_logo(df["Sequence window"][df[motif[0]].notnull()].apply(lambda x: x[8:23]),
                                    sequence_motif="{} - {}".format(motif[0], motif[1]))

    if ret_fig:
        return fig
    fig.show()


# VISUALIZE PHOSPHO SITES

def _vis_psites_init(domain_position, domain_color, length):
    if domain_position is None:
        domain_position = []
    # check if domain_color is a cmap name
    try:
        cm = plt.get_cmap(domain_color)
        color = cm(np.linspace(0, 1, len(domain_position)))
    except ValueError as e:
        if isinstance(domain_color, str):
            color = [domain_color, ] * len(domain_position)
        elif isinstance(domain_color, list):
            if len(domain_color) != len(domain_position):
                raise TypeError("Please provide one domain colour per domain") from e
            else:
                color = domain_color
        else:
            raise TypeError("You must provide a colormap name, a colour name or a list of colour names") from e

    lims = (1, length)
    height = lims[1] / 25

    return lims, height, color, domain_position


def vis_psites(name, length, domain_position=None, ps=None, pl=None, plc=None, pls=4, ax=None, domain_color='tab10'):
    # noinspection PyUnresolvedReferences
    # noinspection PyShadowingNames
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
    >>> domain_position = [(35,43),
    ...                    (77,96)]
    >>> ps = [88, 92, 116, 183, 202, 203, 211, 212, 246]
    >>> pl = ["pS88", "pS92", "pS116", "pS183", "pS202", "pS203", "pS211", "pS212", "pS246"]

    colors (A,B,C,D (gray -> purple), Ad, Bd, Cd, Dd (gray -> teal) can be used to indicate regulation)

    >>> plc = ['C', 'A', 'A', 'C', 'Cd', 'D', 'D', 'B', 'D']
    >>> autoprot.visualization.vis_psites(name, length, domain_position, ps, pl, plc, pls=12)

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
        vis.vis_psites(name, length, domain_position, ps, pl, plc, pls=12)
        plt.show()

    """

    lims, height, color, domain_position = _vis_psites_init(domain_position, domain_color, length)

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

    # only plot phospho site if there are any
    if ps is not None:
        text_color = {"A": "gray",
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
                     pl[idx] if pl is not None else '',
                     fontsize=pls,
                     rotation=90,
                     color=text_color[plc[idx]] if plc is not None else 'black')

    plt.subplots_adjust(left=0.25)
    plt.ylim(height)
    plt.xlim(lims)
    ax1.axes.get_yaxis().set_visible(False)
    plt.title(name + '\n', size=18)
    plt.tight_layout()


def ivis_psites(name, length, domain_position=None, ps=None, pl=None, plc=None, domain_color='tab10',
                ret_fig=False):
    lims, height, color, domain_position = _vis_psites_init(domain_position, domain_color, length)

    def to_rgba(ndarray):
        return f"rgba({ndarray[0] * 256}, {ndarray[1] * 256}, {ndarray[2] * 256}, {ndarray[3]})"

    fig = go.Figure()

    # the protein background
    shapes = [
        dict(
            type='rect',
            xref='x',
            yref='y',
            x0=0,
            y0=0,
            x1=length,
            y1=height,
            fillcolor='lightgray',
            layer='below'
        )
    ]

    # the annotated domains
    for idx, (start, end) in enumerate(domain_position):
        width = end - start
        shapes.append(
            dict(
                type='rect',
                xref='x',
                yref='y',
                x0=start,
                y0=0,
                x1=start+width,
                y1=height,
                fillcolor=to_rgba(color[idx]),
                layer='below'

            )
        )

    fig.update_layout(shapes=shapes,
                      xaxis_range=[0, length],  # length
                      yaxis_range=[0, 2*height],  # height
                      xaxis=dict(showgrid=False,
                                 visible=True,
                                 zeroline=False,
                                 showticklabels=True),
                      yaxis=dict(showgrid=False,
                                 visible=False,
                                 showticklabels=False),
                      plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)",
                      title=name
                      )

    # only plot phospho site if there are any
    if ps is not None:
        text_color = {"A": "gray",
                      "Ad": "gray",
                      "B": "#dc86fa",
                      "Bd": "#6AC9BE",
                      "C": "#aa00d7",
                      "Cd": "#239895",
                      "D": "#770087",
                      "Dd": "#008080"}

        for idx, site in enumerate(ps):
            fig.add_trace(go.Scatter(x=[site, ] * int(height), y=np.linspace(0, height, int(height)), mode='lines',
                                     hovertemplate=f'{pl[idx]}<br>Pos: {site}<extra></extra>',
                                     line=dict(color=text_color[plc[idx]] if plc is not None else 'black')))

    if ret_fig:
        return fig
    fig.show()
