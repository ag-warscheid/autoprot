# -*- coding: utf-8 -*-
"""
Autoprot Analysis Functions.

@author: Wignand, Julian, Johannes

@documentation: Julian
"""
from typing import Literal
from datetime import date

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from operator import itemgetter
import missingno as msn

from .. import r_helper

from gprofiler import GProfiler

gp = GProfiler(user_agent="autoprot", return_dataframe=True)
RFUNCTIONS, R = r_helper.return_r_path()

# check where this is actually used and make it local
cmap = sns.diverging_palette(150, 275, s=80, l=55, n=9)


def miss_analysis(
    df,
    cols,
    n=None,
    sort="ascending",
    text=True,
    vis=True,
    extra_vis=False,
    save_dir=None,
):
    # noinspection PyUnresolvedReferences
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

    .. plot::
        :context: close-figs

        phos = pd.read_csv("../data/Phospho (STY)Sites_minimal.zip", sep="\t", low_memory=False)
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
    if sort == "descending":
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
        allines = ""
        for i in range(n):
            allines += f"{data[i][0]} has {data[i][2]} of {data[i][1]} entries missing ({round(data[i][3], 2)}%)."
            allines += "\n"
            # line separator
            allines += "-" * 80

        if save_dir:
            with open(f"{save_dir}/missAnalysis_text.txt", "w") as f:
                for _ in range(n):
                    f.write(allines)

        # write all lines at once
        print(allines)

    if (
        vis
    ):  # Visualize the % missingness of first n entries of dataframe as a bar plot.
        data = pd.DataFrame(
            data=data, columns=["Name", "tot_values", "tot_miss", "perc_miss", "rank"]
        )

        plt.figure(figsize=(7, 7))
        ax = plt.subplot()
        # plot colname against total missing values
        splot = sns.barplot(x=data["tot_miss"].iloc[:n], y=data["Name"].iloc[:n])

        # add the percentage of missingness to every bar of the plot
        for idx, p in enumerate(splot.patches):
            s = f"{str(round(data.iloc[idx, 3], 2))}%"
            x = p.get_width() + p.get_width() * 0.01
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
        msn.matrix(df, sort="ascending", ax=ax, sparkline=False)
        if save_dir:
            plt.savefig(save_dir + "/missAnalysis_vis2.pdf")
        return True


def missed_cleavages(df_evidence, enzyme="Trypsin/P", save=True, ax=None, title=None):
    """
    Parameters
    ----------
    df_evidence : cleaned pandas DataFrame from Maxquant analysis
    enzyme : str,
        Give any chosen Protease from MQ. The default is "Trypsin/P".
    save : bool,
        While True table and fig will be saved in active filepath.
    ax: matplotlib axis, optional
        If provided, the plot will be drawn on the given axis.
    title: str, optional
        Title for the plot. If None, a default title will be used.

    Returns
    -------
    plt.Figure, pd.DataFrame
        Fig and table for missed cleavage analysis
    """
    # set plot style
    plt.style.use("seaborn-v0_8-whitegrid")

    # set parameters
    today = date.today().isoformat()

    if "Experiment" not in df_evidence.columns.tolist():
        print(
            "Warning: Column [Experiment] either not unique or missing,\n\
              column [Raw file] used"
        )
        experiments = list(set((df_evidence["Raw file"])))
    else:
        experiments = list(set((df_evidence["Experiment"])))

    rawfiles = list(set((df_evidence["Raw file"])))
    if len(experiments) != len(rawfiles):
        experiments = rawfiles
        print(
            "Warning: Column [Experiment] either not unique or missing,\n\
              column [Raw file] used"
        )

    # calculate miss cleavage for each raw file in df_evidence
    df_missed_cleavage_summary = pd.DataFrame()
    for raw, df_group in df_evidence.groupby("Raw file"):
        if enzyme == "Trypsin/P":
            df_missed_cleavage = df_group["Missed cleavages"].value_counts()
        else:
            df_missed_cleavage = df_group[
                "Missed cleavages ({0})".format(enzyme)
            ].value_counts()
        df_missed_cleavage_summary = pd.concat(
            [df_missed_cleavage_summary, df_missed_cleavage], axis=1
        )
    try:
        df_missed_cleavage_summary.columns = experiments
    except Exception as e:
        print(f"unexpected error in col [Experiment]: {e}")
    df_missed_cleavage_summary = (
        df_missed_cleavage_summary
        / df_missed_cleavage_summary.apply(np.sum, axis=0)
        * 100
    )
    df_missed_cleavage_summary = df_missed_cleavage_summary.round(2)

    # making the barchart figure missed cleavage
    x_ax = len(experiments) + 1
    if ax is None:
        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(x_ax, 4))
    else:
        fig = ax.get_figure()
        ax1 = ax

    if title is None:
        fig.suptitle(
            "% Missed cleavage per run",
            fontdict=None,
            horizontalalignment="center",
            size=14,
        )
    else:
        fig.suptitle(title)
    df_missed_cleavage_summary.T.plot(kind="bar", stacked=True, ax=ax1)
    ax1.set_xlabel("Experiment assinged in MaxQuant", size=12)
    ax1.set_ylabel("Missed cleavage [%]", size=12)
    ax1.legend(bbox_to_anchor=(1.5, 1), loc="upper right", borderaxespad=0.0)

    if save:
        # save fig in cwd with date
        plt.savefig(f"{today}_BarChart_missed-cleavage.pdf", dpi=600)
        # save df missed cleavage summery as .csv
        df_missed_cleavage_summary.to_csv(
            f"{today}_Missed-cleavage_result-table.csv", sep="\t", index=False
        )

    return fig, df_missed_cleavage_summary


def enrichment_specificity(
    df_evidence,
    mod_col="Phospho (STY)",
    groupby="Experiment",
    save=True,
    ax=None,
    title=None,
):
    """

    Parameters
    ----------
    df_evidence : cleaned pandas DataFrame from Maxquant analysis
    mod_col : str,
          Give type of enrichment for analysis. The default is 'Phospho (STY)'.
          ('Met--> Phosphonate', 'Cys--> Phosphonate', 'Met --> Biotin')
    groupby : str,
        Column name to group the data by. The default is 'Experiment'.
    save : bool,
        While True table and fig will be saved in active filepath.
    ax: matplotlib axis, optional
        If provided, the plot will be drawn on the given axis.
    title: str, optional
        Title for the plot. If None, a default title will be used.

    Returns
    -------
    plt.Figure, pd.DataFrame
        Fig and table for enrichment specificity analysis

    """
    # set parameters
    today = date.today().isoformat()

    if groupby not in df_evidence.columns.tolist():
        if "Raw file" in df_evidence.columns.tolist():
            print(
                f"Warning: Column [{groupby}] either not unique or missing, column [Raw file] used"
            )
            groupby = "Raw file"
        else:
            raise KeyError(
                "Columns [Experiment] and [Raw file] are missing. Is this a MaxQuant evidence table?"
            )

    df_summary = pd.DataFrame()

    for name, group in df_evidence.groupby(groupby):
        nonmod = round(
            ((group[mod_col] == 0).astype(int).sum() / group.shape[0] * 100), 2
        )
        mod = round(((group[mod_col] > 0).sum() / group.shape[0] * 100), 2)

        df_summary.loc[name, "Modified peptides [%]"] = mod
        df_summary.loc[name, "Non-modified peptides [%]"] = nonmod

    # make barchart
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if title is None:
        fig.suptitle(
            f"Enrichment specificity [%] for {mod_col}",
            fontdict=None,
            horizontalalignment="center",
            size=14,
        )
    else:
        fig.suptitle(title)

    df_summary.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("peptides [%]")
    ax.legend(bbox_to_anchor=(1.5, 1), loc="upper right", borderaxespad=0.0)

    if save:
        # save fig in cwd with date
        plt.savefig(f"{today}_BarPlot_enrichmentSpecificity.pdf", dpi=600)
        # save df missed cleavage summery as .csv
        df_summary.T.to_csv(
            f"{today}_enrichmentSpecificity_result-table.csv", sep="\t", index=False
        )

    return fig, df_summary


def SILAC_labeling_efficiency(
    df_evidence: pd.DataFrame,
    label: Literal["L", "M", "H"] = None,
    lys_ax: plt.Axes = None,
    arg_ax: plt.Axes = None,
    r_to_p_conversion: Literal["Pro5", "Pro6"] = None,
    r_to_p_return: bool = True,
    r_to_p_ax: plt.Axes = None,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parameters
    ----------
    df_evidence : MaxQuant evidence table
        clean reverse and contaminant first autoprot.preprocessing.cleaning()
    label : str, optional
        The label type used in the experiment ('L', 'M', 'H'). The default is None (will analyze 'H').
    lys_ax : matplotlib axis, optional
        If provided, the Lys labeling plot will be drawn on the given axis.
    arg_ax : matplotlib axis, optional
        If provided, the Arg labeling plot will be drawn on the given axis.
    r_to_p_conversion : str, optional
        variable modifications e.g. ["Pro5", "Pro6"] set in MaxQuant.
        If provided, Arg to Pro conversion will be calculated. The default is None.
    r_to_p_return : bool, optional
        If True, the Arg to Pro conversion table will be returned. The default is True.
    r_to_p_ax : matplotlib axis, optional
        If provided, the Arg to Pro conversion plot will be drawn on the given axis.

    Notes
    -----
    To generate the evidence table, search the files in MaxQuant specifying the labels as labels (i.e.
    multiplicity > 1) and the Arg to Pro conversion as variable modification. There is no need to separate into
    different analysis groups for this analysis.

    Returns
    -------
    Fig, table for SILAC label incorporation
    """
    # work on a copy of the dataframe
    df_evidence: pd.DataFrame = df_evidence.copy()  # noqa

    if label is None:
        label = "H"

    if not isinstance(r_to_p_conversion, str):
        raise TypeError("r_to_p_conversion is not a string")

    if r_to_p_return and r_to_p_conversion is None:
        raise ValueError(
            "r_to_p_return is True but r_to_p_conversion is None. Please provide a modification name."
        )

    df_evidence.sort_values(["Raw file"], inplace=True)
    experiments = list(df_evidence["Experiment"].unique())  # type: ignore
    runs = list(df_evidence["Raw file"].unique())  # type: ignore

    # mapping raw file names to experiments
    raw2exp = {}
    for key, val in zip(runs, experiments):
        raw2exp[key] = val

    # Extracted from old function labeling_efficiency
    # Create column names for the intensity and ratio columns.
    intensity_col = f"Intensity {label}"
    ratio_col_name = f"Ratio Intensity {label}/total"

    # Create empty DataFrames to store the results.
    df_labeling_eff_k = pd.DataFrame()
    df_labeling_eff_r = pd.DataFrame()

    # Remove NaN values from the intensity column.
    df_evidence[intensity_col] = df_evidence[intensity_col].dropna()  # type: ignore

    # Calculate the SILAC labeling ratio for each peptide.
    df_evidence[ratio_col_name] = (
        df_evidence[intensity_col] / df_evidence["Intensity"] * 100  # type: ignore
    )

    # Iterate through each sample (i.e., raw file).
    for raw, df_group in df_evidence.groupby("Raw file"):
        # Calculate the SILAC labeling efficiency for Lysine.
        k_filter = (df_group["R Count"] == 0) & (df_group["K Count"] > 0)
        only_k = df_group.loc[k_filter, ratio_col_name].dropna()
        s_k_binned = only_k.value_counts(bins=range(0, 101, 10), sort=False)
        k_count = only_k.shape[0]
        s_relative_k_binned = s_k_binned / k_count * 100
        df_labeling_eff_k[raw] = s_relative_k_binned

        # Calculate the SILAC labeling efficiency for Arginine.
        r_filter = (df_group["R Count"] > 0) & (df_group["K Count"] == 0)
        only_r = df_group.loc[r_filter, ratio_col_name].dropna()
        s_r_binned = only_r.value_counts(bins=range(0, 101, 10), sort=False)
        r_count = only_r.shape[0]
        s_relative_r_binned = s_r_binned / r_count * 100
        df_labeling_eff_r[raw] = s_relative_r_binned

    # Rename the columns to match the experimental setup.
    exp = []
    for elem in df_labeling_eff_k.columns:
        exp.append(raw2exp[elem])
    df_labeling_eff_k.columns = exp
    df_labeling_eff_r.columns = exp

    # Combine the two DataFrames into one and return it.
    df_labeling_eff = pd.concat(
        [df_labeling_eff_k, df_labeling_eff_r],
        keys=["Lys incorporation", "Arg incorporation"],
        names=["Amino acid", "bins"],
    )

    if lys_ax is None and arg_ax is None:
        fig, (lys_ax, arg_ax) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    elif (lys_ax is None) or (arg_ax is None):
        raise ValueError("Both lys_ax and arg_ax must be provided or neither.")

    for ax, (aa, df) in zip((lys_ax, arg_ax), (df_labeling_eff.groupby(level=0))):
        df.plot(kind="bar", ax=ax, legend=True)

        ax.set_xticklabels(
            [
                "0-10",
                "11-20",
                "21-30",
                "31-40",
                "41-50",
                "51-60",
                "61-70",
                "71-80",
                "81-90",
                "91-100",
            ]
        )
        ax.set_xlabel(
            "bins",
        )

        # set minor yticks every 10 percent
        ax.yaxis.set_minor_locator(plt.MultipleLocator(10))
        # set major yticks every 20 percent
        ax.yaxis.set_major_locator(plt.MultipleLocator(20))
        # activate grid on all yticks
        ax.grid(which="both", axis="y", linestyle="--", linewidth=0.5)

        ax.set_ylabel(f"{label} {aa} [%]")
        ax.set_ylim(0, 100)
        plt.tight_layout()

    df_r_to_p_summary = None

    if r_to_p_conversion is not None:
        # calculate Arg to Pro for each raw file in df_evidence
        # collect per-rawfile summaries in a list of dicts then build DataFrame
        df_evidence["P count"] = df_evidence["Sequence"].str.count("P")  # type: ignore
        rows = []
        for raw, df_group in df_evidence.groupby("Raw file"):
            p_count = df_group.loc[df_group[r_to_p_conversion] == 0, "P count"].sum()
            r_count = df_group.loc[
                df_group[r_to_p_conversion] > 0, r_to_p_conversion
            ].sum()
            # map raw file to experiment name if available
            exp_name = raw2exp[raw]
            rows.append(
                {"Experiment": exp_name, "P count": p_count, r_to_p_conversion: r_count}
            )

        df_r_to_p_summary = pd.DataFrame(rows).set_index("Experiment")
        df_r_to_p_summary.dropna(inplace=True)
        df_r_to_p_summary["RtoP [%]"] = (
            df_r_to_p_summary[r_to_p_conversion] / df_r_to_p_summary["P count"] * 100
        )

        # making the box plot Arg to Pro conversion
        if r_to_p_ax is None:
            fig, r_to_p_ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))

        df_r_to_p_summary["RtoP [%]"].plot(kind="bar", ax=r_to_p_ax)
        r_to_p_ax.set_ylabel(f"Arg to {r_to_p_conversion} [%]")
        plt.tight_layout()

    if r_to_p_return and r_to_p_conversion is not None:
        return df_labeling_eff, df_r_to_p_summary

    return df_labeling_eff


def dimethyl_labeling_efficieny(df_evidence, label, save=True) -> pd.DataFrame:
    """
    This function calculates the labeling efficiency of dimethyl labeled samples using a MaxQuant evidence table.

    Parameters
    ----------
    df_evidence : pd.DataFrame
        MaxQuant evidence table as pandas.Dataframe
    label : str
        The label type used in the experiment ('L', 'M', 'H')
    save : bool
        If True table and fig will be saved in active filepath.

    Returns
    -------
    pd.DataFrame
        Results from the analysis
    """
    # set plot style
    plt.style.use("seaborn-v0_8-whitegrid")

    # set parameters
    today = date.today().isoformat()

    df_evidence.sort_values(["Raw file"], inplace=True)
    if "Experiment" in df_evidence.columns.tolist():
        experiments = list((df_evidence["Experiment"].unique()))
    else:
        experiments = list((df_evidence["Raw file"].unique()))
        print(
            "Warning: Column [Experiment] either not unique or missing,\n\
              column [Raw file] used"
        )

    df_labeling_eff = pd.DataFrame()

    df_evidence.dropna(subset=["Intensity"], inplace=True)
    df_evidence["Ratio Intensity {}/total".format(label)] = (
        df_evidence["Intensity {}".format(label)] / df_evidence["Intensity"] * 100
    )

    # build label ratio based on given label
    for raw, df_group in df_evidence.groupby("Raw file"):
        s_binned = df_group["Ratio Intensity {}/total".format(label)].value_counts(
            bins=range(0, 101, 10), sort=False
        )
        count = df_group["Ratio Intensity {}/total".format(label)].count()
        s_relative_binned = s_binned / count * 100
        df_labeling_eff = pd.concat([df_labeling_eff, s_relative_binned], axis=1)

    try:
        df_labeling_eff.columns = experiments
    except Exception as e:
        print(f"unexpected error in col [Experiment]: {e}")

    if save:
        df_labeling_eff.to_csv(
            "{0}_labeling_eff_{1}_summary.csv".format(today, label), sep="\t"
        )

    # plot labeling efficiency overview
    x_ax = len(experiments) + 1
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(x_ax * 2, 4))
    fig.suptitle(
        "Dimethyl Labeling efficiency {}".format(label),
        fontdict=None,
        horizontalalignment="center",
        size=14,
        # ,fontweight="bold"
    )
    df_labeling_eff.plot(kind="bar", ax=ax1)
    ax1.set_xlabel("bins", size=12)
    ax1.set_ylabel("{} labeling [%]".format(label), size=12)

    plt.tight_layout()
    if save:
        plt.savefig(
            "{0}_BoxPlot_Lab-eff-{1}_overview.pdf".format(today, label), dpi=600
        )

    # plot labeling efficiency Lys for each experiment separately
    # columns and rows from number of experiments in df_evidence
    number_of_subplots = len(experiments)

    if (number_of_subplots % 3) == 0:
        number_of_columns = 3
    elif (number_of_subplots % 2) == 0:
        number_of_columns = 2
    else:
        number_of_columns = 1

    number_of_rows = number_of_subplots // number_of_columns

    # adjust figsize
    # 8.3 *11.7 inch is the size of a dinA4
    fig = plt.figure(figsize=(2.76 * number_of_columns, 2.925 * number_of_rows))

    for col_name, plot in zip(experiments, range(number_of_subplots)):
        ax1 = fig.add_subplot(number_of_rows, number_of_columns, plot + 1)

        # filter for bins with low values: set 1%
        df_labeling_eff[col_name][df_labeling_eff[col_name].cumsum() > 1].plot(
            kind="bar", ax=ax1
        )

        ax1.set_title(col_name)
        ax1.set_xlabel("bins", size=8)
        ax1.set_ylabel("{} Dimethyl incorporation [%]".format(label), size=8)
        ax1.set_ylim(0, 100)
        ax1.axhline(95, linestyle="--", c="k")

    fig.suptitle(
        "Dimethyl Labeling efficiency {}".format(label), horizontalalignment="center"
    )
    plt.tight_layout()

    if save:
        plt.savefig(
            "{0}_BoxPlot_Lab-eff-{1}-seperately.pdf".format(today, label), dpi=1200
        )

    return df_labeling_eff


def tmt6plex_labeling_efficiency(evidence_under, evidence_sty_over, evidence_h_over):
    """
    Calculate TMT6plex labeling efficiency from 3 dedicated MaxQuant searches as described in Zecha et al. 2019.
    TMT6plex channels should be named in MQ experiments.
    @author: Johannes Zimmermann

    Parameters
    ----------
    evidence_under : evidence.txt as pd.DataFrame from under-labeling search,
                     label-free search with TMT as variable modification on peptide n-term and lysine
    evidence_sty_over : evidence.txt as pd.DataFrame from over-labeling search,
                       MS2-TMT experiment with TMT as variable modification on serine, threonine, tyrosine
    evidence_h_over : evidence.txt as pd.DataFrame from over-labeling search,
                     MS2-TMT experiment with TMT as variable modification on histidine

    Returns
    -------
    df : pd.DataFrame
         Results from labeling efficiency calculations as absolute and relative numbers.
    fig : Figure of labeling efficiency as stacked bars. Under/Over-labeling as separated axis.

    """

    # initiate DataFrame for results
    df_efficiency = pd.DataFrame()

    # delete N-terminal acetylated arginines without lysine (can't be modified)
    evidence_under = evidence_under[
        ~(
            evidence_under["Modified sequence"].str.contains(
                r"\_\(Acetyl \(Protein N\-term\)\)"
            )
            & evidence_under["Modified sequence"].str.contains("K")
        )
    ]

    # cal
    evidence_under["K count"] = evidence_under["Sequence"].str.count("K")
    evidence_sty_over["S count"] = evidence_sty_over["Sequence"].str.count("S")
    evidence_sty_over["T count"] = evidence_sty_over["Sequence"].str.count("T")
    evidence_sty_over["Y count"] = evidence_sty_over["Sequence"].str.count("Y")

    evidence_h_over["H count"] = evidence_h_over["Sequence"].str.count("H")

    for raw, group in evidence_under.groupby("Experiment"):
        (
            lysine,
            nterm,
            sty_over_experiment,
            under_experiment,
            sty_over,
            h_over_experiment,
            h_over,
        ) = ("",) * 7

        if str(126) in raw:
            nterm = r"\_\(TMT6plex\-Nterm126\)"
            lysine = (
                "TMT6plex-Lysine126"  # modifications have to be named after MQ mod.list
            )
            h_over = "TMT6plex (H)126"
            sty_over = "TMT6plex (STY)126"
            under_experiment = raw
            h_over_experiment = [
                entry
                for entry in evidence_h_over["Experiment"].unique()
                if str(126) in entry
            ][0]
            sty_over_experiment = [
                entry
                for entry in evidence_sty_over["Experiment"].unique()
                if str(126) in entry
            ][0]
        if str(127) in raw:
            nterm = r"\_\(TMT6plex\-Nterm127\)"
            lysine = "TMT6plex-Lysine127"
            h_over = "TMT6plex (H)127"
            sty_over = "TMT6plex (STY)127"
            under_experiment = raw
            h_over_experiment = [
                entry
                for entry in evidence_h_over["Experiment"].unique()
                if str(127) in entry
            ][0]
            sty_over_experiment = [
                entry
                for entry in evidence_sty_over["Experiment"].unique()
                if str(127) in entry
            ][0]
        if str(128) in raw:
            nterm = r"\_\(TMT6plex\-Nterm128\)"
            lysine = "TMT6plex-Lysine128"
            h_over = "TMT6plex (H)128"
            sty_over = "TMT6plex (STY)128"
            under_experiment = raw
            h_over_experiment = [
                entry
                for entry in evidence_h_over["Experiment"].unique()
                if str(128) in entry
            ][0]
            sty_over_experiment = [
                entry
                for entry in evidence_sty_over["Experiment"].unique()
                if str(128) in entry
            ][0]
        if str(129) in raw:
            nterm = r"\_\(TMT6plex\-Nterm129\)"
            lysine = "TMT6plex-Lysine129"
            h_over = "TMT6plex (H)129"
            sty_over = "TMT6plex (STY)129"
            under_experiment = raw
            h_over_experiment = [
                entry
                for entry in evidence_h_over["Experiment"].unique()
                if str(129) in entry
            ][0]
            sty_over_experiment = [
                entry
                for entry in evidence_sty_over["Experiment"].unique()
                if str(129) in entry
            ][0]
        if str(130) in raw:
            nterm = r"\_\(TMT6plex\-Nterm130\)"
            lysine = "TMT6plex-Lysine130"
            h_over = "TMT6plex (H)130"
            sty_over = "TMT6plex (STY)130"
            under_experiment = raw
            h_over_experiment = [
                entry
                for entry in evidence_h_over["Experiment"].unique()
                if str(130) in entry
            ][0]
            sty_over_experiment = [
                entry
                for entry in evidence_sty_over["Experiment"].unique()
                if str(130) in entry
            ][0]
        if str(131) in raw:
            nterm = r"\_\(TMT6plex\-Nterm131\)"
            lysine = "TMT6plex-Lysine131"
            h_over = "TMT6plex (H)131"
            sty_over = "TMT6plex (STY)131"
            under_experiment = raw
            h_over_experiment = [
                entry
                for entry in evidence_h_over["Experiment"].unique()
                if str(131) in entry
            ][0]
            sty_over_experiment = [
                entry
                for entry in evidence_sty_over["Experiment"].unique()
                if str(131) in entry
            ][0]

        df_efficiency.loc[raw, ["fully labeled"]] = (
            (group["K count"] == group[lysine])
            & (
                ~(
                    group["Modified sequence"].str.contains(
                        r"\_\(Acetyl \(Protein N\-term\)\)"
                    )
                )
                & (group["Modified sequence"].str.contains(nterm))
            )
        ).sum()

        df_efficiency.loc[raw, ["partially labeled"]] = (
            group["Modified sequence"].str.contains(r"\(TMT6plex").sum()
            - df_efficiency.loc[raw, ["fully labeled"]].values
        )

        df_efficiency.loc[raw, ["not labeled"]] = (
            ~group["Modified sequence"].str.contains(r"\(TMT6plex")
        ).sum()

        df_efficiency.loc[[under_experiment], "sum all labeled"] = (
            df_efficiency["not labeled"]
            + df_efficiency["fully labeled"]
            + df_efficiency["partially labeled"]
        )

        df_efficiency.loc[[under_experiment], "PSM STY"] = (
            evidence_sty_over[evidence_sty_over["Experiment"] == sty_over_experiment][
                "S count"
            ].sum()
            + evidence_sty_over[evidence_sty_over["Experiment"] == sty_over_experiment][
                "T count"
            ].sum()
            + evidence_sty_over[evidence_sty_over["Experiment"] == sty_over_experiment][
                "Y count"
            ].sum()
        )
        df_efficiency.loc[[under_experiment], "TMT (STY)"] = evidence_sty_over[
            evidence_sty_over["Experiment"] == sty_over_experiment
        ][sty_over].sum()

        df_efficiency.loc[[under_experiment], "PSM H"] = evidence_h_over[
            evidence_h_over["Experiment"] == h_over_experiment
        ]["H count"].sum()
        df_efficiency.loc[[under_experiment], "TMT (H)"] = evidence_h_over[
            evidence_h_over["Experiment"] == h_over_experiment
        ][h_over].sum()

    df_efficiency["% fully labeled"] = (
        df_efficiency["fully labeled"] / df_efficiency["sum all labeled"] * 100
    )
    df_efficiency["% partially labeled"] = (
        df_efficiency["partially labeled"] / df_efficiency["sum all labeled"] * 100
    )
    df_efficiency["% not labeled"] = (
        df_efficiency["not labeled"] / df_efficiency["sum all labeled"] * 100
    )
    df_efficiency["% overlabeled STY"] = (
        (df_efficiency["TMT (STY)"]) / df_efficiency["PSM STY"] * 100
    )
    df_efficiency["% overlabeled H"] = (
        (df_efficiency["TMT (H)"]) / df_efficiency["PSM H"] * 100
    )
    df_efficiency["% overlabeled STY+H"] = (
        df_efficiency["% overlabeled H"] + df_efficiency["% overlabeled STY"]
    )

    # make figure TMT6plex labeling efficiency
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        nrows=2, ncols=2, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.suptitle(
        "Comparison of labeling efficiency in TMT6plex",
        fontdict=None,
        horizontalalignment="center",
    )

    sns.barplot(
        x=df_efficiency.index,
        y=df_efficiency["% fully labeled"]
        + df_efficiency["% partially labeled"]
        + df_efficiency["% not labeled"],
        ax=ax1,
        color="#dd4e26",
        **{"label": "% not labeled"},
    )

    sns.barplot(
        x=df_efficiency.index,
        y=df_efficiency["% fully labeled"] + df_efficiency["% partially labeled"],
        ax=ax1,
        color="#2596be",
        **{"label": "% partially labeled"},
    )

    sns.barplot(
        x=df_efficiency.index,
        y=df_efficiency["% fully labeled"],
        ax=ax1,
        color="#063970",
        **{"label": "% fully labeled"},
    )

    plt.xticks(
        np.arange(len(df_efficiency.index)), rotation=45, horizontalalignment="right"
    )

    sns.barplot(
        x=df_efficiency.index, y=df_efficiency["% not labeled"], ax=ax3, color="#dd4e26"
    )

    ax1.set_ylabel("Peptides [%]")
    ax3.set_ylabel("Peptides [%]")
    ax1.legend(bbox_to_anchor=(-0.75, 1), loc="upper left", borderaxespad=0.0)
    ax3.set_xlabel("channel", horizontalalignment="center", fontsize=12)
    ax1.set_xticklabels([])
    ax3.set_xticklabels(df_efficiency.index, rotation=45, horizontalalignment="right")

    sns.barplot(
        x=df_efficiency.index,
        y=df_efficiency["% overlabeled STY+H"],
        ax=ax2,
        color="#cce7e8",
        **{"label": "% overlabeled STY+H"},
    )

    sns.barplot(
        x=df_efficiency.index,
        y=df_efficiency["% overlabeled STY"],
        ax=ax2,
        color="#44bcd8",
        **{"label": "% overlabeled STY"},
    )

    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    ax2.set_ylabel("AA Residues [%]")

    ax2.set_xticklabels(df_efficiency.index, rotation=90, horizontalalignment="center")

    ax4.remove()

    return df_efficiency, fig
