# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 15:44:07 2021
Updated on Fr Jan 27 14:39:00 2022
@author: jzimmermann
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date

from autoprot import preprocessing as pp


def MissedCleavage(df_evidence, enzyme="Trypsin/P"):
    """
    Parameters
    ----------
    df_evidence : cleaned pandas DataFrame from Maxquant analysis
    enzyme : str,
        Give any chosen Protease from MQ. The default is "Trypsin/P".

    Figure in Pdf format in given filepath,
    Result table as csv
    -------
    None.
    """
    # set plot style
    plt.style.use('seaborn-whitegrid')

    # set parameters
    today = date.today().isoformat()

    if "Experiment" not in df_evidence:
        print("Warning: Column [Experiment] either not unique or missing,\n\
              column [Raw file] used")
        experiments = None
    else:
        experiments = list(set((df_evidence["Experiment"])))

    rawfiles = list(set((df_evidence["Raw file"])))
    if len(experiments) != len(rawfiles):
        experiments = rawfiles
        print("Warning: Column [Experiment] either not unique or missing,\n\
              column [Raw file] used")

    # calculate miss cleavage for each raw file in df_evidence
    df_missed_cleavage_summary = pd.DataFrame()
    for raw, df_group in df_evidence.groupby("Raw file"):
        if enzyme == "Trypsin/P":
            df_missed_cleavage = df_group["Missed cleavages"].value_counts()
        else:
            df_missed_cleavage = df_group["Missed cleavages ({0})".format(enzyme)].value_counts()
        df_missed_cleavage_summary = pd.concat([df_missed_cleavage_summary, df_missed_cleavage],
                                               axis=1)
    try:
        df_missed_cleavage_summary.columns = experiments
    except Exception as e:
        print(f"unexpected error in col [Experiment]: {e}")
    df_missed_cleavage_summary = df_missed_cleavage_summary / df_missed_cleavage_summary.apply(np.sum, axis=0) * 100
    df_missed_cleavage_summary = df_missed_cleavage_summary.round(2)

    # making the barchart figure missed cleavage
    x_ax = len(experiments) + 1
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(x_ax, 4))
    fig.suptitle("% Missed cleavage per run", fontdict=None,
                 horizontalalignment='center', size=14
                 # ,fontweight="bold"
                 )
    df_missed_cleavage_summary.T.plot(kind="bar", stacked=True, ax=ax1)
    ax1.set_xlabel("Experiment assinged in MaxQuant", size=12)
    ax1.set_ylabel("Missed cleavage [%]", size=12)
    ax1.legend(bbox_to_anchor=(1.5, 1),
               loc='upper right', borderaxespad=0.)

    # plt.tight_layout()
    plt.savefig("{0}_BarChart_missed-cleavage.pdf".format(today), dpi=600)

    # save df missed cleavage summery as .csv
    df_missed_cleavage_summary.to_csv(f"{today}_Missed-cleavage_result-table.csv", sep='\t', index=False)

    # return results
    return print(df_missed_cleavage_summary, ax1)


def enrichmentSpecifity(df_evidence, typ="Phospho"):
    # sourcery skip: raise-specific-error
    """

    Parameters
    ----------
    df_evidence : cleaned pandas DataFrame from Maxquant analysis
    typ : str,
          Give type of enrichment for analysis. The default is "Phospho".

    Figure in Pdf format in given filepath,
    Result table as csv
    -------
    None.

    """
    # set plot style
    plt.style.use('seaborn-whitegrid')

    # set parameters
    today = date.today().isoformat()

    if "Experiment" not in df_evidence:
        print("Warning: Column [Experiment] either not unique or missing,\n\
              column [Raw file] used")
        experiments = None
    else:
        experiments = list(set((df_evidence["Experiment"])))

    rawfiles = list(set((df_evidence["Raw file"])))
    if len(experiments) != len(rawfiles):
        print("Warning: Column [Experiment] either not unique or missing,\n\
              column [Raw file] used")

    if not typ:
        print("Error: Choose type of enrichment")

    if typ == "AHA-Phosphonate":
        colname = 'Met--> Phosphonate'
    elif typ == "CPT":
        colname = 'Cys--> Phosphonate'
    elif typ == "Phospho":
        colname = 'Phospho (STY)'
    else:
        raise Exception("Invalid type specified. Must be 'AHA-Phosphonate', 'CPT', or 'Phospho'")
    df = pd.DataFrame()
    df_summary = pd.DataFrame()

    for name, group in df_evidence.groupby("Experiment"):
        nonmod = round(((group[colname] == 0).sum() / group.shape[0] * 100), 2)
        mod = round(((group[colname] > 0).sum() / group.shape[0] * 100), 2)

        # print(name)
        # print("% peptides without modification: ",nonmod)
        # print("% peptides with modification: ",mod)

        df.loc[name, "Modified peptides [%]"] = mod
        df.loc[name, "Non-modified peptides [%]"] = nonmod

    df_summary = pd.concat([df_summary, df], axis=0)

    # make barchart
    fig, ax = plt.subplots()
    fig.suptitle('Enrichment specificty [%]', fontdict=None,
                 horizontalalignment='center', size=14
                 # ,fontweight="bold"
                 )

    df_summary.plot(kind="bar", stacked=True, ax=ax)

    ax.set_ylabel('peptides [%]')
    ax.legend(bbox_to_anchor=(1.5, 1),
              loc='upper right', borderaxespad=0.)

    plt.savefig("{0}_BarPlot_enrichmentSpecifity.pdf".format(today), dpi=600)

    # save df missed cleavage summery as .csv
    df_summary.T.to_csv(f"{today}_enrichmentSpecifity_result-table.csv", sep='\t', index=False)

    # return results
    return print(df.T, ax)


def TMT6plex_labeling_efficiency(evidence_under, evidence_sty_over, evidence_h_over):
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
        ~(evidence_under["Modified sequence"].str.contains('\_\(Acetyl \(Protein N\-term\)\)') &
          evidence_under["Modified sequence"].str.contains('K'))]

    # cal
    evidence_under["K count"] = evidence_under["Sequence"].str.count('K')
    evidence_sty_over["S count"] = evidence_sty_over["Sequence"].str.count('S')
    evidence_sty_over["T count"] = evidence_sty_over["Sequence"].str.count('T')
    evidence_sty_over["Y count"] = evidence_sty_over["Sequence"].str.count('Y')

    evidence_h_over["H count"] = evidence_h_over["Sequence"].str.count('H')

    for raw, group in evidence_under.groupby("Experiment"):
        lysine, nterm, sty_over_experiment, under_experiment, sty_over, h_over_experiment, h_over = ('',) * 7

        if str(126) in raw:
            nterm = '\_\(TMT6plex\-Nterm126\)'
            lysine = 'TMT6plex-Lysine126'  # modifications have to be named after MQ mod.list
            h_over = 'TMT6plex (H)126'
            sty_over = 'TMT6plex (STY)126'
            under_experiment = raw
            h_over_experiment = [entry for entry in evidence_h_over["Experiment"].unique() if str(126) in entry][0]
            sty_over_experiment = [entry for entry in evidence_sty_over["Experiment"].unique() if str(126) in entry][0]
        if str(127) in raw:
            nterm = '\_\(TMT6plex\-Nterm127\)'
            lysine = 'TMT6plex-Lysine127'
            h_over = 'TMT6plex (H)127'
            sty_over = 'TMT6plex (STY)127'
            under_experiment = raw
            h_over_experiment = [entry for entry in evidence_h_over["Experiment"].unique() if str(127) in entry][0]
            sty_over_experiment = [entry for entry in evidence_sty_over["Experiment"].unique() if str(127) in entry][0]
        if str(128) in raw:
            nterm = '\_\(TMT6plex\-Nterm128\)'
            lysine = 'TMT6plex-Lysine128'
            h_over = 'TMT6plex (H)128'
            sty_over = 'TMT6plex (STY)128'
            under_experiment = raw
            h_over_experiment = [entry for entry in evidence_h_over["Experiment"].unique() if str(128) in entry][0]
            sty_over_experiment = [entry for entry in evidence_sty_over["Experiment"].unique() if str(128) in entry][0]
        if str(129) in raw:
            nterm = '\_\(TMT6plex\-Nterm129\)'
            lysine = 'TMT6plex-Lysine129'
            h_over = 'TMT6plex (H)129'
            sty_over = 'TMT6plex (STY)129'
            under_experiment = raw
            h_over_experiment = [entry for entry in evidence_h_over["Experiment"].unique() if str(129) in entry][0]
            sty_over_experiment = [entry for entry in evidence_sty_over["Experiment"].unique() if str(129) in entry][0]
        if str(130) in raw:
            nterm = '\_\(TMT6plex\-Nterm130\)'
            lysine = 'TMT6plex-Lysine130'
            h_over = 'TMT6plex (H)130'
            sty_over = 'TMT6plex (STY)130'
            under_experiment = raw
            h_over_experiment = [entry for entry in evidence_h_over["Experiment"].unique() if str(130) in entry][0]
            sty_over_experiment = [entry for entry in evidence_sty_over["Experiment"].unique() if str(130) in entry][0]
        if str(131) in raw:
            nterm = '\_\(TMT6plex\-Nterm131\)'
            lysine = 'TMT6plex-Lysine131'
            h_over = 'TMT6plex (H)131'
            sty_over = 'TMT6plex (STY)131'
            under_experiment = raw
            h_over_experiment = [entry for entry in evidence_h_over["Experiment"].unique() if str(131) in entry][0]
            sty_over_experiment = [entry for entry in evidence_sty_over["Experiment"].unique() if str(131) in entry][0]

        df_efficiency.loc[raw, ["fully labeled"]] = ((group["K count"] == group[lysine]) &
                                                     (~(group["Modified sequence"].str.contains(
                                                         '\_\(Acetyl \(Protein N\-term\)\)')) &
                                                      (group["Modified sequence"].str.contains(nterm)))).sum()

        df_efficiency.loc[raw, ["partially labeled"]] = group["Modified sequence"].str.contains('\(TMT6plex').sum() -\
            df_efficiency.loc[raw, ["fully labeled"]].values

        df_efficiency.loc[raw, ["not labeled"]] = (~group["Modified sequence"].str.contains('\(TMT6plex')).sum()

        df_efficiency.loc[[under_experiment], "sum all labeled"] = df_efficiency["not labeled"] + df_efficiency[
            "fully labeled"] + df_efficiency["partially labeled"]

        df_efficiency.loc[[under_experiment], "PSM STY"] = \
            evidence_sty_over[evidence_sty_over["Experiment"] == sty_over_experiment]["S count"].sum() \
            + evidence_sty_over[evidence_sty_over["Experiment"] == sty_over_experiment]["T count"].sum() \
            + evidence_sty_over[evidence_sty_over["Experiment"] == sty_over_experiment]["Y count"].sum()
        df_efficiency.loc[[under_experiment], "TMT (STY)"] = \
            evidence_sty_over[evidence_sty_over["Experiment"] == sty_over_experiment][sty_over].sum()

        df_efficiency.loc[[under_experiment], "PSM H"] = \
            evidence_h_over[evidence_h_over["Experiment"] == h_over_experiment][
                "H count"].sum()
        df_efficiency.loc[[under_experiment], "TMT (H)"] = \
            evidence_h_over[evidence_h_over["Experiment"] == h_over_experiment][h_over].sum()

    df_efficiency["% fully labeled"] = df_efficiency["fully labeled"] / df_efficiency["sum all labeled"] * 100
    df_efficiency["% partially labeled"] = df_efficiency["partially labeled"] / df_efficiency["sum all labeled"] * 100
    df_efficiency["% not labeled"] = df_efficiency["not labeled"] / df_efficiency["sum all labeled"] * 100
    df_efficiency["% overlabeled STY"] = (df_efficiency["TMT (STY)"]) / df_efficiency["PSM STY"] * 100
    df_efficiency["% overlabeled H"] = (df_efficiency["TMT (H)"]) / df_efficiency["PSM H"] * 100
    df_efficiency["% overlabeled STY+H"] = df_efficiency["% overlabeled H"] + df_efficiency["% overlabeled STY"]

    # make figure TMT6plex labeling efficiency
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 8),
                                                 gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle("Comparison of labeling efficiency in TMT6plex", fontdict=None, horizontalalignment='center')

    sns.barplot(x=df_efficiency.index,
                y=df_efficiency["% fully labeled"] + df_efficiency["% partially labeled"] + df_efficiency[
                    "% not labeled"],
                ax=ax1, color="#dd4e26", **{"label": "% not labeled"})

    sns.barplot(x=df_efficiency.index, y=df_efficiency["% fully labeled"] + df_efficiency["% partially labeled"],
                ax=ax1, color="#2596be", **{"label": "% partially labeled"})

    sns.barplot(x=df_efficiency.index, y=df_efficiency["% fully labeled"],
                ax=ax1, color="#063970", **{"label": "% fully labeled"})

    plt.xticks(np.arange(len(df_efficiency.index)),
               rotation=45,
               horizontalalignment='right')

    sns.barplot(x=df_efficiency.index,
                y=df_efficiency["% not labeled"],
                ax=ax3, color="#dd4e26")

    ax1.set_ylabel("Peptides [%]")
    ax3.set_ylabel("Peptides [%]")
    ax1.legend(bbox_to_anchor=(-0.75, 1), loc='upper left', borderaxespad=0.)
    ax3.set_xlabel("channel",
                   horizontalalignment='center',
                   fontsize=12)
    ax1.set_xticklabels([])
    ax3.set_xticklabels(df_efficiency.index,
                        rotation=45,
                        horizontalalignment='right')

    sns.barplot(x=df_efficiency.index, y=df_efficiency["% overlabeled STY+H"],
                ax=ax2, color="#cce7e8", **{"label": "% overlabeled STY+H"})

    sns.barplot(x=df_efficiency.index, y=df_efficiency["% overlabeled STY"],
                ax=ax2, color="#44bcd8", **{"label": "% overlabeled STY"})

    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax2.set_ylabel("AA Residues [%]")

    ax2.set_xticklabels(df_efficiency.index,
                        rotation=90,
                        horizontalalignment='center')

    ax4.remove()

    return df_efficiency, fig
