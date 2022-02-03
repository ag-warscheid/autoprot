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
    '''

    Parameters
    ----------
    df_evidence : cleaned pandas DataFrame from Maxquant analysis
    enzyme : str, 
        Give any chosen Protease from MQ. The default is "Trypsin/P".

    Figure in Pdf format in given filepath,
    Result table as csv
    -------
    None.

    '''
    ##set plot style
    plt.style.use('seaborn-whitegrid')
    
    ##set parameters
    today = date.today().isoformat()
    try:
        experiments = list(set((df_evidence["Experiment"])))
    except:
        print("Warning: Column [Experiment] either not unique or missing,\n\
              column [Raw file] used")
    rawfiles = list(set((df_evidence["Raw file"])))
    if len(experiments) != len(rawfiles):
        experiments = rawfiles
        print("Warning: Column [Experiment] either not unique or missing,\n\
              column [Raw file] used")
 
    ####calculate miss cleavage for each raw file in df_evidence

    df_missed_cleavage_summary = pd.DataFrame()
    for raw, df_group in df_evidence.groupby("Raw file"):
        if enzyme == "Trypsin/P":
            df_missed_cleavage = df_group["Missed cleavages"].value_counts()
        elif enzyme != "Trypsin/P":
            df_missed_cleavage = df_group["Missed cleavages ({0})".format(enzyme)].value_counts()
        else:
            print("unexpected column name or enzyme")
       
        df_missed_cleavage_summary = pd.concat([df_missed_cleavage_summary, df_missed_cleavage], axis=1)
    
    try:
        df_missed_cleavage_summary.columns = experiments
    except:
        print("unexpected error in col [Experiment]")
    df_missed_cleavage_summary = df_missed_cleavage_summary/df_missed_cleavage_summary.apply(np.sum, axis=0)*100
    df_missed_cleavage_summary = df_missed_cleavage_summary.round(2)
    
    #### making the box plot figure missed cleavage
    x_ax=len(experiments)+1
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(x_ax, 4))
    fig.suptitle("% Missed cleavage per run", fontdict=None,
                 horizontalalignment='center', size=14
                 #,fontweight="bold"
                 )
    df_missed_cleavage_summary.T.plot(kind="bar", stacked=True, ax=ax1)
    ax1.set_xlabel("Experiment assinged in MaxQuant", size=12)
    ax1.set_ylabel("Missed cleavage [%]", size=12)
    ax1.legend(bbox_to_anchor=(1.5, 1),
               loc='upper right', borderaxespad=0.)
    
    #plt.tight_layout()
    plt.savefig("{0}_BoxPlot_missed-cleavage.pdf".format(today), dpi=600)
    
    #### save df missed cleavage summery as .csv
    df_missed_cleavage_summary.to_csv("{}_Missed-cleavage_result-table.csv".format(today), sep='\t', index=False)
    
    #### return results
    return print(df_missed_cleavage_summary, ax1)