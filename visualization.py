# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 10:56:14 2019

@author: Wignand
"""
from scipy import stats
from scipy.stats import zscore
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.image as mplimg
import matplotlib.ticker as ticker
import pylab as pl
import importlib
from autoprot import venn
from matplotlib_venn import venn2
from matplotlib_venn import venn3
import logomaker
import colorsys
import matplotlib.patches as patches
import colorsys
from itertools import chain

from datetime import date

from wordcloud import WordCloud
from wordcloud import STOPWORDS

from pdfminer3.layout import LAParams, LTTextBox
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer3.converter import PDFPageAggregator, TextConverter

import io

from PIL import Image

import plotly
import plotly.express as px
import plotly.graph_objects as go

plt.rcParams['pdf.fonttype'] = 42

from io import BytesIO
from scipy.stats import ttest_1samp

"""
To do: Add functionality of embedding all the plots as subplots in figures by providing ax parameter
"""

def correlogram(df, columns=None, file="proteinGroups",log=True,saveDir = None, saveType="pdf", saveName="pairPlot", lowerTriang="scatter",
sampleFrac=None, bins=100):

    """
    function plots a pair plot of the dataframe
    intensity columns in order to assess the reproducibility
    :params df: dataframe from MaxQuant file
    :params columns: the columns to be visualized 
    :params file: proteinGroups or Phospho(STY) (does only change annotation)
    :params log: whehter provided intensities are already log transformed
    :params saveDir: where the plots are saved,
    :params saveType: what format the saved plots have (pdf, png)
    :params saveName: the name of the saved file
    :params lowerTriang: scatter, hexBin, hist2d, the kind of plot displayed in the lower triang
    :sampleFrac: float; fraction between 0 and 1 to indicate fraction of entries to be shown in scatter
                 might be useful for large correlograms in order to make it possible to work with those in illustrator
    """
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
        0.91 : "#ec7bea",
        0.92 : "#e57cee",
        0.93 : "#dc7cf0",
        0.94 : "#d27df2",
        0.95 : "#c87df4",
        0.96 : "#be7df6",
        0.97 : "#b47ef9",
        0.98 : "#a97efb",
        0.99 : "#9e7ffd",
        1 : "#927fff"
            }
        if r <= 0.8:
            return "#D63D40"
        else:
            return colors[np.round(r,2)]


    def corrfunc(x, y, **kws):
        df = pd.DataFrame({"x":x, "y":y})
        df = df.dropna()
        x = df["x"].values
        y = df["y"].values
        r, _ = stats.pearsonr(x, y)
        ax = plt.gca()
        ax.annotate("r = {:.2f}".format(r),
                    xy=(.1, .9), xycoords=ax.transAxes)


    def heatmap(x,y,**kws):
        df = pd.DataFrame({"x":x, "y":y})
        df = df.replace(-np.inf, np.nan).dropna()
        x = df["x"].values
        y = df["y"].values
        r, _ = stats.pearsonr(x,y)
        ax = plt.gca()
        ax.add_patch(mpl.patches.Rectangle((0,0),5,5,  color=getColor(r), transform=ax.transAxes))
        ax.tick_params(axis = "both", which = "both", length=0)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)


    def lowerScatter(x,y,**kws):
        data = pd.DataFrame({"x":x, "y":y})
        if sampleFrac is not None:
            data = data.sample(int(data.shape[0]*sampleFrac))
        ax = plt.gca()
        ax.scatter(data['x'],data['y'], linewidth=0)
        
        
    def lowerHexBin(x,y,**kws):
        plt.hexbin(x,y, cmap="Blues", bins=bins,
        gridsize=50)
        
        
    def lowerhist2D(x,y,**kws):
        df = pd.DataFrame({"x":x, "y":y})
        df = df.dropna()
        x = df["x"].values
        y = df["y"].values
        plt.hist2d(x,y, bins=bins, cmap="Blues", vmin=0, vmax=1)

    def proteins_found(x,y,**kws):
        df = pd.DataFrame({"x":x, "y":y})
        df = df.dropna()
        x = df["x"].values
        y = df["y"].values
        r, _ = stats.pearsonr(x,y)
        ax = plt.gca()
        if file == "proteinGroups":
            ax.annotate("{} proteins identified".format(str(len(y))),
                    xy=(.1,.9), xycoords=ax.transAxes)
            ax.annotate("R: {}".format(str(round(r,2))),
                    xy=(.25,.5),size=18, xycoords=ax.transAxes)
        elif file == "Phospho (STY)":
            ax.annotate("{} peptides identified".format(str(len(y))),
                    xy=(.1,.9), xycoords=ax.transAxes)
            ax.annotate("R: {}".format(str(round(r,2))),
                    xy=(.25,.5),size=18, xycoords=ax.transAxes)
        
        
    if len(columns)==0:
        raise ValueError("No columns provided!")
    else:
        temp_df = df[columns]

    if log == False:
        temp_df[columns] = np.log10(temp_df[columns])


    y = temp_df
    y.replace(-np.inf, np.nan, inplace=True)
    corr = y.corr()
    
    g = sns.PairGrid(y)
    g.map_lower(corrfunc)
#    g.map_lower(plt.scatter) #very strange when using sns.scatterplot heatmap not showing?
    if lowerTriang == "scatter":
        g.map_lower(lowerScatter)
    elif lowerTriang == "hexBin":
        g.map_lower(lowerHexBin)
    elif lowerTriang == "hist2d":
        g.map_lower(lowerhist2D)
    g.map_diag(sns.histplot)
    g.map_upper(heatmap)
    g.map_upper(proteins_found)
    
    if saveDir is not None:
        if saveType == "pdf":
            plt.savefig(f"{saveDir}/{saveName}.pdf")
        elif saveType == "png":
            plt.savefig(f"{saveDir}/{saveName}.png")


def corrMap(df, columns, cluster=False, annot=None, cmap="YlGn", figsize=(7,7),
            saveDir = None, saveType="pdf", saveName="pairPlot", ax=None, **kwargs):
    corr = df[columns].corr()
    if cluster == False:
        if ax is None:
            plt.figure(figsize=figsize)
            sns.heatmap(corr, cmap=cmap, square=True, cbar=False, annot=annot, **kwargs)
        else:
            sns.heatmap(corr, cmap=cmap, square=True, cbar=False, annot=annot, ax=ax, **kwargs)
    else:
        sns.clustermap(corr, cmap=cmap, annot=annot, **kwargs)

    if saveDir is not None:
        if saveType == "pdf":
            plt.savefig(f"{saveDir}/{saveName}.pdf")
        elif saveType == "png":
            plt.savefig(f"{saveDir}/{saveName}.png")


def probPlot(df, col, dist = "norm",figsize=(6,6)):
    """
    function plots a QQ_plot of the provided column
    Here the data is compared against a theoretical distribution (default is normal)
    """
    t = stats.probplot(df[col].replace([-np.inf, np.inf], [np.nan, np.nan]).dropna(), dist=dist)
    label = f"R²: {round(t[1][2],4)}"
    y=[]
    x=[]
    for i in np.linspace(min(t[0][0]),max(t[0][0]), 100):
        y.append(t[1][0] * i + t[1][1])
        x.append(i)
    plt.figure(figsize=figsize)
    plt.scatter(t[0][0], t[0][1], alpha=.3, color="purple",
    label=label)
    plt.plot(x,y, color="teal")
    sns.despine()
    plt.title(f"Probability Plot\n{col}")
    plt.xlabel("Theorectical Quantiles")
    plt.ylabel("Sample Quantiles")
    plt.legend()


def boxplot(df, reps, title=None, labels=[], compare=False,
            data="logFC", file=None, retFig=False, figsize=(15,5),**kwargs):
    """
    function plots boxplots of intensities
    :param df: dataframe to test
    :param reps: columns which are replicates
    :param title: optional provide a list with titles for the blots
    :param labels: optional provide a list with labels for the axis
    :param compare: if False expects a single list, if True expects two list (e.g. normalized and nonnormalized Ratios)
    :param data: either logFC or Intensity
    :kwargs: arguments passed to pandas boxplot
    """

    

    # check if inputs make sense
    if compare==True:
        if len(reps) != 2:
            raise ValueError("You want to compare two sets, provide two sets.")

    #set ylabel based on data
    if data == "logFC":
        ylabel="logFC"
    elif data == "Intensity":
        ylabel="Intensity"

    if compare == True:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        ax[0].set_ylabel(ylabel)
        ax[1].set_ylabel(ylabel)
        if title:
            for idx, t in enumerate(title):
                ax[idx].set_title("{}".format(t))

        for idx, rep in enumerate(reps):
            df[rep].boxplot(ax=ax[idx], **kwargs)
            ax[idx].grid(False)
            if data == "logFC":
                 ax[idx].axhline(0,0,1, color="gray", ls="dashed")
            
        if len(labels)>0:
            for idx in [0,1]:
                temp = ax[idx].set_xticklabels(labels)
                tlabel = ax[idx].get_xticklabels()
                for i, label in enumerate(tlabel):
                    label.set_y(label.get_position()[1]-(i%2)*.05)
        else:
            ax[0].set_xticklabels([str(i+1) for i in range(len(reps[0]))])
            ax[1].set_xticklabels([str(i+1) for i in range(len(reps[1]))])
        sns.despine()

    else: 
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        
        df[reps].boxplot(**kwargs)
        ax.grid(False)
        plt.title(title)
        plt.ylabel(ylabel)

        if len(labels)>0:
            temp = ax.set_xticklabels(labels)
            ax.set_xticklabels(labels)
            for i, label in enumerate(temp):
                    label.set_y(label.get_position()[1]-(i%2)*.05)
        else:
            ax.set_xticklabels(str(i+1) for i in range(len(reps)))
        if data == "logFC":
            ax.axhline(0,0,1, color="gray", ls="dashed")
        sns.despine()

    if file is not None:
        plt.savefig(fr"{file}/BoxPlot.pdf")
    if retFig == True:
        return fig


def intensityRank(data, rankCol="log10_Intensity" ,label=None, n=5, title="Rank Plot", figsize=(15,7), file=None,
                  hline=None, **kwargs):
    """
    function that draws a rank plot
    :param data: pandas dataframe
    :param rankCol: the column with the values to be ranked (e.g. Intensity values)
    :param label: the column with the labels
    
    ToDo: add option to highlight a set of datapoints 
            could be alternative to topN labeling
    """
    
    if data.shape[1] > 1:
        data = data.sort_values(by=rankCol, ascending=True)
        y = data[rankCol]
    else:
        y = data.sort_values(ascending=True)
    
    x = range(data.shape[0])

    plt.figure(figsize=figsize)
    sns.scatterplot(x=x,y=y,
                   linewidth=0, **kwargs)
                   
    if hline is not None:
        plt.axhline(hline,0,1, ls="dashed", color="lightgray")
                   
    if label is not None:
        top_y = y.iloc[-1]
        top_yy = np.linspace(top_y-n*0.4, top_y, n)
        top_oy = y[-n:]
        top_xx = x[-n:]
        top_ss = data[label].iloc[-n:]
                
        for ys,xs,ss,oy in zip(top_yy, top_xx, top_ss, top_oy):
            plt.plot([xs,xs+len(x)*.1], [oy, ys], color="gray")
            plt.text(x=xs+len(x)*.1, y=ys, s=ss)
            
        low_y = y.iloc[0]
        low_yy = np.linspace(low_y, low_y+n*0.4, n)
        low_oy = y[:n]
        low_xx = x[:n]
        low_ss = data[label].iloc[:n]

        for ys,xs,ss,oy in zip(low_yy, low_xx, low_ss, low_oy):
            plt.plot([xs,xs+len(x)*.1], [oy, ys], color="gray")
            plt.text(x=xs+len(x)*.1, y=ys, s=ss)
    
    sns.despine()
    plt.xlabel("# rank")
    plt.title(title)
    
    if file is not None:
        plt.savefig(fr"{file}/RankPlot.pdf")


def vennDiagram(df, figsize=(10,10), retFig=False, proportional=True):
    """
    draws vennDiagrams, if proportional is set Trup matplotlib_venn implementation is used
    which draws proportional venn diagrams (venn2 and venn3)
    """
    data = df.copy(deep=True)
    n = data.shape[1]
    if n>6:
        raise ValueError("You cannot analyse more than 6 conditions in a venn diagram!")
    elif n==1:
        raise ValueError("You should at least provide 2 conditions to compare in a venn diagram!")
    reps = data.columns.to_list()
    data["UID"] = range(data.shape[0])
    if n == 2:
        g1 = data[[reps[0]] + ["UID"]]
        g2 = data[[reps[1]] + ["UID"]]
        g1 = set(g1["UID"][g1[reps[0]].notnull()].values)
        g2 = set(g2["UID"][g2[reps[1]].notnull()].values)
        if proportional:
            venn2([g1,g2], set_labels=reps)
        else:
            labels = venn.get_labels([g1, g2], fill=["number", "logic"])
            fig, ax = venn.venn2(labels, names=[reps[0], reps[1]],figsize=figsize)
        
    elif n == 3:
        g1 = data[[reps[0]] + ["UID"]]
        g2 = data[[reps[1]] + ["UID"]]
        g3 = data[[reps[2]] + ["UID"]]
        g1 = set(g1["UID"][g1[reps[0]].notnull()].values)
        g2 = set(g2["UID"][g2[reps[1]].notnull()].values)
        g3 = set(g3["UID"][g3[reps[2]].notnull()].values)
        if proportional:
            venn3([g1,g2,g3], set_labels=reps)
        else:
            labels = venn.get_labels([g1, g2, g3], fill=["number", "logic"])
            fig, ax = venn.venn3(labels, names=[reps[0], reps[1], reps[2]],figsize=figsize)

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
        fig, ax = venn.venn4(labels, names=[reps[0], reps[1], reps[2], reps[3]],figsize=figsize)
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
        fig, ax = venn.venn5(labels, names=[reps[0], reps[1], reps[2], reps[3], reps[4]],figsize=figsize)
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
        fig, ax = venn.venn6(labels, names=[reps[0], reps[1], reps[2], reps[3], reps[4], reps[5]],figsize=figsize)
        
    if retFig == True:
        return fig


def volcano(df, logFC, p=None, score=None, pt=0.05, fct=None, annot=None,interactive=False,
    sig_col="green", bg_col="lightgray", title="Volcano Plot", figsize=(6,6), hover_name=None, 
    highlight=None, highlight_col = "red", annotHighlight="all",
    custom_bg = {}, 
    custom_fg = {},
    custom_hl = {},
    retFig = False,
    ax=None,
    legend=True):
    """
    Function that draws Volcano plot. This function can either plot a static or an interactive version of the volcano.
    Further it allows the user to set the desired logFC and p value threshold as well as toggle the annotation of the plot.
    If provided it is possible to highlight a selection of datapoints in the plot. Those will then be annotated instead of 
    all significant entries.
    @params:
    ::df:: dataframe which contains the data
    ::logFC:: column of the dataframe with the log fold change
    ::p:: column of the dataframe containing p values (provide score or p)
    ::score:: column of the dataframe containing -log10(p values) (provide score or p)
    ::pt:: float; p-value threshold under which a entry is deemed significantly regulated
    ::fct:: floag; fold change threshold at which an entry is deemed significant regulated
    ""annot"" boolean; whether or not to annotate the plot 
    ::retFig: boolean, whether orr not to return the figure, can be used to further customize it afterwards
    """
        
    def setAesthetic(d, typ, interactive):
        """
        Function that sets standard aesthetics of volcano
        and integrates those with user defined ones
        @params
        ::d:: user provided dictionary
        ::typ:: whether foreground or background
        """
        if typ == "bg":
            standard = {"alpha":0.33, 
                           "s":2,
                           "label":"background",
                           "linewidth":0} #this hugely improves performance in illustrator
                           
        elif typ == "fg":
            standard = {"alpha":1, 
                           "s":6,
                           "label":"sig",
                           "linewidth":0}
                           
        elif typ == "hl":
            standard = {"alpha":1, 
                           "s":20,
                           "label":"POI",
                           "linewidth":0}

        if interactive == False:
            for k in standard.keys():
                if k in d:
                    pass
                else:
                    d[k] = standard[k]
                    
        return d
        
        
    def checkData(df,logFC, score, p, pt, fct):
        if score is None and p is None:
            raise ValueError("You have to provide either a score or a (adjusted) p value.")
        elif score is None:
            df["score"] = -np.log10(df[p])
            score = "score"
        else:
            df.rename(columns={score:"score"}, inplace=True)
            score = "score"
            p = "p"
            df["p"] = 10**(df["score"]*-1)

        # define the significant eintries in dataframe
        df["SigCat"] = "-"
        if fct is not None:
            df.loc[(df[p] < pt) & (abs(df[logFC]) > fct),"SigCat"] = '*'
        else:
            df.loc[(df[p] < pt), "SigCat"] = '*'
        sig = df[df["SigCat"]=='*'].index
        unsig = df[df["SigCat"]=="-"].index

        return df, score, sig, unsig

    
    df = df.copy(deep=True)
    # set up standard aesthetics
    custom_bg = setAesthetic(custom_bg, typ="bg", interactive=interactive)
    custom_fg = setAesthetic(custom_fg, typ="fg", interactive=interactive)
    if highlight is not None:
        custom_hl = setAesthetic(custom_hl, typ="hl", interactive=interactive)

    # check for input correctness and make sure score is present in df for plot
    df, score, sig, unsig = checkData(df,logFC, score, p, pt, fct)
        
    if interactive == False:
        #darw figure
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax=plt.subplot() #for a bare minimum plot you do not need this line
        #the following lines of code generate the scatter the rest is styling 

        ax.scatter(df[logFC].loc[unsig], df["score"].loc[unsig], color=bg_col, **custom_bg)
        ax.scatter(df[logFC].loc[sig], df["score"].loc[sig], color=sig_col, **custom_fg)
        if highlight is not None:
            ax.scatter(df[logFC].loc[highlight], df["score"].loc[highlight], color=highlight_col, **custom_hl)

        #draw threshold lines
        if fct:
            ax.axvline(fct,0,1,ls="dashed", color="lightgray")
            ax.axvline(-fct,0,1,ls="dashed", color="lightgray")
        ax.axhline(-np.log10(pt), 0, 1,ls="dashed", color="lightgray")

        #remove of top and right plot boundary
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        #seting x and y labels and title
        ax.set_ylabel("score")
        ax.set_xlabel("logFC")
        ax.set_title(title, size=18)

        #add legend
        if legend==True:
            ax.legend()

            #Annotation
        if annot is not None:
            #get x and y coordinates as well as strings to plot
            if highlight is None:
                xs = df[logFC].loc[sig]
                ys = df["score"].loc[sig]
                ss = df[annot].loc[sig]
            else:
                if annotHighlight == "all":
                    xs = df[logFC].loc[highlight]
                    ys = df["score"].loc[highlight]
                    ss = df[annot].loc[highlight]
                elif annotHighlight == "sig":
                    xs = df[logFC].loc[set(highlight)&set(sig)]
                    ys = df["score"].loc[set(highlight)&set(sig)]
                    ss = df[annot].loc[set(highlight)&set(sig)]

            #annotation
            for idx, (x,y,s) in enumerate(zip(xs,ys,ss)):
                if idx%2 != 0:
                    if x < 0:
                        ax.plot([x,x-.2],[y,y+.2],color="gray")
                        ax.text(x-.3,y+.25,s)
                    else:
                        ax.plot([x,x+.2],[y,y+.2],color="gray")
                        ax.text(x+.2,y+.2,s)
                else:
                    if x < 0:
                        ax.plot([x,x-.2],[y,y-.2],color="gray")
                        ax.text(x-.3,y-.25,s)
                    else:
                        ax.plot([x,x+.2],[y,y-.2],color="gray")
                        ax.text(x+.2,y-.2,s)
                        
        if retFig == True:
            return fig

    if interactive == True:
    
        colors = [bg_col,sig_col]
        if highlight is not None:
            
            df["SigCat"] = "-"
            df.loc[highlight, "SigCat"] = "*"
            if hover_name is not None:
                fig = px.scatter(data_frame=df,x=logFC, y=score, hover_name=hover_name, 
                          color="SigCat",color_discrete_sequence=colors,
                                 opacity=0.5,category_orders={"SigCat":["-","*"]}, title=title)
            else:
                fig = px.scatter(data_frame=df,x=logFC, y=score,
                          color="SigCat",color_discrete_sequence=colors,
                                 opacity=0.5,category_orders={"SigCat":["-","*"]}, title=title)
                                 
        else:
            if hover_name is not None:
                fig = px.scatter(data_frame=df,x=logFC, y=score, hover_name=hover_name, 
                          color="SigCat",color_discrete_sequence=colors,
                                 opacity=0.5,category_orders={"SigCat":["-","*"]}, title=title)
            else:
                fig = px.scatter(data_frame=df,x=logFC, y=score,
                          color="SigCat",color_discrete_sequence=colors,
                                 opacity=0.5,category_orders={"SigCat":["-","*"]}, title=title)
                                 
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
        #add fold change visualization
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

        fig.update_layout({
            'plot_bgcolor': 'rgba(70,70,70,1)',
            'paper_bgcolor': 'rgba(128, 128, 128, 0.25)',
        })
        return fig


def logIntPlot(df, logFC, Int, fct=None, annot=False, interactive=False,
    sig_col="green", bg_col="lightgray", title="LogFC Intensity Plot", figsize=(6,6), hover_name=None):
    
    """
    logIntPlot -> still lacks functionality. Copy from volcano function (highlight etc)
    also add option to not highlight anything
    """
    
    df = df.copy(deep=True)

    df = df[~df[Int].isin([-np.inf, np.nan])]
    df["SigCat"] = "-"
    if fct is not None:
        df.loc[abs(df[logFC])>fct,"SigCat"] = "*" 
    unsig = df[df["SigCat"] == "-"].index
    sig = df[df["SigCat"] == "*"].index
    
    if interactive == False:
        #draw figure
        plt.figure(figsize=figsize)
        ax=plt.subplot()
        plt.scatter(df[logFC].loc[unsig], df[Int].loc[unsig], color=bg_col,alpha=.75, s=5, label="background")
        plt.scatter(df[logFC].loc[sig], df[Int].loc[sig], color=sig_col, label="POI")

        #draw threshold lines
        if fct:
            plt.axvline(fct,0,1,ls="dashed", color="lightgray")
            plt.axvline(-fct,0,1,ls="dashed", color="lightgray")
        plt.axvline(0,0,1,ls="dashed", color="gray")

        #remove of top and right plot boundary
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        #seting x and y labels and title
        plt.ylabel("log Intensity")
        plt.xlabel("logFC")
        plt.title(title, size=18)

        #add legend
        plt.legend()

        if annot==True:
            #Annotation
            #get x and y coordinates as well as strings to plot
            xs = df[logFC].loc[sig]
            ys = df[Int].loc[sig]
            ss = df["Gene names"].loc[sig]
            
            #annotation
            for idx, (x,y,s) in enumerate(zip(xs,ys,ss)):
                if idx%2 != 0:
                    if x < 0:
                        plt.plot([x,x-.2],[y,y+.2],color="gray")
                        plt.text(x-.3,y+.25,s)
                    else:
                        plt.plot([x,x+.2],[y,y+.2],color="gray")
                        plt.text(x+.2,y+.2,s)
                else:
                    if x < 0:
                        plt.plot([x,x-.2],[y,y-.2],color="gray")
                        plt.text(x-.3,y-.25,s)
                    else:
                        plt.plot([x,x+.2],[y,y-.2],color="gray")
                        plt.text(x+.2,y-.2,s)

    if interactive == True:
        if hover_name is not None:
            fig = px.scatter(data_frame=df,x=logFC, y=Int, hover_name=hover_name, 
                      color="SigCat",color_discrete_sequence=["cornflowerblue","mistyrose"],
                             opacity=0.5,category_orders={"SigCat":["*","-"]}, title="Volcano plot")
        else:
            fig = px.scatter(data_frame=df,x=logFC, y=Int,
                      color="SigCat",color_discrete_sequence=["cornflowerblue","mistyrose"],
                             opacity=0.5,category_orders={"SigCat":["*","-"]}, title="Volcano plot")

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
        fig.show()


def MAPlot(df, x, y, interactive=False, fct=None,
    sig_col="green", bg_col="lightgray", title="MA Plot", figsize=(6,6), hover_name=None):
    """
    needs docstring !
    """
    df = df.copy(deep=True)
    df["M"] = df[x] - df[y]
    df["A"] = 1/2 * (df[x]+df[y])
    df["M"].replace(-np.inf, np.nan, inplace=True)
    df["A"].replace(-np.inf, np.nan, inplace=True)
    df["SigCat"] = False
    if fct is not None:
        df.loc[abs(df["M"]) > fct, "SigCat"] = True
    if interactive == False:
        #draw figure
        plt.figure(figsize=figsize)
        sns.scatterplot(data=df,x='A',y='M', linewidth=0, hue="SigCat")
        plt.axhline(0,0,1, color="black", ls="dashed")
        plt.title(title)
        plt.ylabel("M")
        plt.xlabel("A")
        
        if fct is not None:
            plt.axhline(fct,0,1,color="gray", ls="dashed")
            plt.axhline(-fct,0,1,color="gray", ls="dashed")

    if interactive == True:
        if hover_name is not None:
            fig = px.scatter(data_frame=df,x='A', y='M', hover_name=hover_name, 
                      color="SigCat",color_discrete_sequence=["cornflowerblue","mistyrose"],
                             opacity=0.5,category_orders={"SigCat":["*","-"]}, title=title)
        else:
            fig = px.scatter(data_frame=df,x='A', y='M',
                      color="SigCat",color_discrete_sequence=["cornflowerblue","mistyrose"],
                             opacity=0.5,category_orders={"SigCat":["*","-"]}, title=title)

        fig.update_yaxes(showgrid=False, zeroline=True)
        fig.update_xaxes(showgrid=False, zeroline=False)

        #fig.add_trace(
        #    go.Scatter(
        #        y=[df['A'].min(), df['A'].max()],
        #        x=[0,0],
        #        mode="lines",
        #        line=go.scatter.Line(color="teal", dash="longdash"),
        #        showlegend=False)
        #)
        
        
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

    def hexa(x,y):
        plt.hexbin(x,y, cmap="BuPu",
                  gridsize=40)
        plt.plot(x,y.rolling(window=200, min_periods=10).mean(), color="teal")
        plt.xlabel("rank (mean)")

    df = df.copy(deep=True)
    df["mean"] = abs(df[reps].mean(1))
    df["sd"] = df[reps].std(1)
    df = df.sort_values(by="mean")

    p = sns.JointGrid(
    x = range(df.shape[0]),
    y = df['sd']
    )

    p = p.plot_joint(
    hexa
    )

    p.ax_marg_y.hist(
    df['sd'],
    orientation = 'horizontal',
    alpha = 0.5,
    bins=50
    )
    
    p.ax_marg_x.get_xaxis().set_visible(False)
    p.ax_marg_x.set_title("Mean SD plot", fontsize=18)


def plotTraces(df, cols, labels=None, colors=None, zScore=None,
              xlabel="", ylabel="logFC", title="", ax=None,
              plotSummary=False, plotSummaryOnly=False, summaryColor="red",
              summaryType="Mean", summaryStyle="solid", **kwargs):
    """
    Function to plot traces. 
    @params:
    ::df: dataframe
    ::cols: columns with numeric data
    ::labels: corresponds to data, used to label traces
    ::zScore: 0 or 1, whether to apply zscore transformation
    
    ToDo:
    Add parameter to plot yerr
    """
    
    x = range(len(cols))
    y = df[cols].T.values
    if zScore is not None and zScore in [0,1]:
        y = zscore(y, axis=zScore)
    
    if ax is None:
        plt.figure()
        ax = plt.subplot()
    ax.set_title(title)
    if plotSummaryOnly == False:
        if colors is None:
            f = ax.plot(x,y, **kwargs)
        else:
            f = []
            for i, yi in enumerate(y.T):
                f += ax.plot(x,yi, color=colors[i],**kwargs)
    if (plotSummary == True) or (plotSummaryOnly == True):
        if summaryType == "Mean":
            f=ax.plot(x, np.mean(y,1), color=summaryColor,
            lw=3, linestyle=summaryStyle)
        elif summaryType == "Median":
            f=ax.plot(x, np.median(y,1), color=summaryColor,
            lw=3, linestyle=summaryStyle)
    
    if labels is not None:
        for s,line in zip(labels,f):
            #get last point for annotation
            ly = line.get_data()[1][-1]
            lx = line.get_data()[0][-1] + 0.1
            plt.text(lx,ly,s)
    
    sns.despine(ax=ax)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)


def sequenceLogo(df, motif, file, ST=False):
    """
    function that generates sequence logos 
    needs sequence matrix as input
    motif  example: 
    motif = ("..R.R..s.......", "MK_down")
    """
    # Note: Function has to be reworked somewhat
    # file should be optional
    # motif and name should be provided in 2 parameter

    def generateSequenceLogo(seq, file, motif=""):




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

        seq = [i for i in seq if len(i)==15] 
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
        m = np.empty((15,20))
        for i in range(m.shape[0]):
            x = [j for j in scoreMatrix[i].values()]
            m[i] = x

        # create Logo object
        kinase_motif_df = pd.DataFrame(m).fillna(0)
        kinase_motif_df.columns = aa_dic.keys()
        k_logo = logomaker.Logo(kinase_motif_df,
                                 font_name='Arial',
                                 color_scheme='dmslogo_funcgroup',
                                 vpad=0,
                                 width=.8)

        k_logo.highlight_position(p=7, color='purple', alpha=.5)
        plt.title("{} SequenceLogo".format(motif))
        #labels=k_logo.ax.get_xticklabels()
        k_logo.ax.set_xticklabels(labels=[-7,-7,-5,-3,-1,1,3,5,7]);
        sns.despine()
        plt.savefig(file)


    def find_motif(x, motif, typ, ST=False):
        import re
        d = x["Sequence window"]
        #In Sequence window the aa of interest is always at pos 15 
        #This loop will check if the motif we are interested in is 
        #centered with its phospho residue at pos 15 of the sequence window
        checkLower = False
        for j,i in enumerate(motif):
            if i.islower() == True:
                pos1 = len(motif)-j
                checkLower = True
        if checkLower == False:
            raise ValueError("Phosphoresidue has to be lower case!")
        if ST == True:
            exp = motif[:pos1-1] + "(S|T)" + motif[pos1:]
        else:
            exp = motif.upper()
            
        pos2 = re.search(exp.upper(),d)
        if pos2:
            pos2 = pos2.end()
            pos = pos2-pos1
            if pos == 15:
                return typ
        else:
            pass


    df[motif[0]] = np.nan
    df[motif[0]] = df.apply(lambda x: find_motif(x, motif[0], motif[0], ST), 1)

    generateSequenceLogo(df["Sequence window"][df[motif[0]].notnull()].apply(lambda x: x[8:23]),
                        file+"/{}_{}.svg".format(motif[0], motif[1]),
                        "{} - {}".format(motif[0], motif[1]))


def visPs(name, length, domain_position, ps, pl,plc, pls=4):
    """
    Function to visualize domains and phosphosites on 
    a protein of interst
    :@param length: int, length of the protein
    :@domain_positions: list, the amino acids at which domains begin and end (protein start and end have not to be included)
    :@ps: list, position of phosphosites
    :@pl: list, label for ps (has to be in same order as ps)
    :@color: list, optionally one can provide a list of colors for the domais, otherwise random color for each new domain

    """
    
    def get_N_HexCol(N=5):

        HSV_tuples = [(x*1/N, 0.75, 0.6) for x in range(N)]
        RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
        return list(RGB_tuples)
    
    
    textColor = {"A"  : "gray",
                 "Ad" : "gray",
                 "B"  : "#dc86fa",
                 "Bd" : "#6AC9BE",
                 "C"  : "#aa00d7",
                 "Cd" : "#239895",
                 "D"  : "#770087",
                 "Dd" : "#008080"}

    color = None #No color options yet

    lims = (1, length)
    height = lims[1]/25


    #start and end positions of domains
    a=[0]+domain_position#beginning
    c= domain_position + [length]#end 

    #colors for domains
    if not color:
        #color2 = get_N_HexCol(N=len(domain_position)+1)
        color2 = ["#EEF1F4"]*50
        color = ["gray"]*(len(domain_position)+1)
        color = list(chain.from_iterable(([(i,j) for i,j in zip(color,color2)])))

    fig1 = plt.figure(figsize=(15,2))
    ax1 = fig1.add_subplot(111, aspect='equal')
    for i in range(len(a)):
        width = c[i] - a[i]
        ax1.add_patch(
            patches.Rectangle((a[i], 0), width, height,color=color[i]) )
    for i, site in enumerate(ps):
        plt.axvline(site,0,1, color="red")
        plt.text(site-1, height-(height+height*0.15),pl[i], fontsize=pls, rotation=90,
                color=textColor[plc[i]])

    plt.subplots_adjust(left=0.25)
    plt.ylim(height)
    plt.xlim(lims)
    ax1.axes.get_yaxis().set_visible(False)
    plt.title(name+'\n', size=18)
    plt.tight_layout()


def styCountPlot(df, figsize=(12,8), typ="bar", retFig=False):
    """
    function that draws a overview of Number of Phospho (STY) distribution of Phospho(STY) file
    Provide dataframe containing "Number of Phospho (STY)" column
    """
    noOfPhos = [int(i) for i in list(pl.flatten([str(i).split(';') for i in df["Number of Phospho (STY)"].fillna(0)]))]
    count = [(noOfPhos.count(i),i) for i in set(noOfPhos)]
    counts_perc = [(round(noOfPhos.count(i)/len(noOfPhos)*100,2), i) for i in set(noOfPhos)]
        
    print("Number of phospho (STY) [total] - (count / # Phospho)")    
    print(count)
    print("Percentage of phospho (STY) [total] - (% / # Phospho)")    
    print(counts_perc)
    df = pd.DataFrame(noOfPhos, columns=["Number of Phospho (STY)"])

    if typ=="bar":
        fig = plt.figure(figsize=figsize)
        ax = sns.countplot(x="Number of Phospho (STY)", data=df)
        plt.title('Number of Phospho (STY)')
        plt.xlabel('Number of Phospho (STY)')
        ncount = df.shape[0]

        # Make twin axis
        ax2=ax.twinx()

        ax2.yaxis.tick_left()
        ax.yaxis.tick_right()

        ax.yaxis.set_label_position('right')
        ax2.yaxis.set_label_position('left')

        ax2.set_ylabel('Frequency [%]')

        for p in ax.patches:
            x=p.get_bbox().get_points()[:,0]
            y=p.get_bbox().get_points()[1,1]
            ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
                    ha='center', va='bottom') # set the alignment of the text
            
        ax.yaxis.set_major_locator(ticker.LinearLocator(11))
        ax2.set_ylim(0,100)
        ax.set_ylim(0,ncount)
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))
    elif typ=="pie":
        fig = plt.figure(figsize=figsize)
        plt.pie([i[0] for i in count], labels=[i[1] for i in count]);
        plt.title("Number of Phosphosites")
    if retFig==True:
        return fig


def chargePlot(df, figsize=(12,8), typ="bar", retFig=False):
    """
    function that draws a overview of charge distribution of Phospho(STY) file
    Provide dataframe containing "charge" column
    """
    df = df.copy(deep=True)
    noOfPhos = [int(i) for i in list(pl.flatten([str(i).split(';') for i in df["Charge"].fillna(0)]))]
    count = [(noOfPhos.count(i),i) for i in set(noOfPhos)]
    counts_perc = [(round(noOfPhos.count(i)/len(noOfPhos)*100,2), i) for i in set(noOfPhos)]
        
    print("charge [total] - (count / # charge)")    
    print(count)
    print("Percentage of charge [total] - (% / # charge)")    
    print(counts_perc)
    df = pd.DataFrame(noOfPhos, columns=["charge"])

    if typ=="bar":
        fig = plt.figure(figsize=figsize)
        ax = sns.countplot(x="charge", data=df)
        plt.title('charge')
        plt.xlabel('charge')
        ncount = df.shape[0]

        # Make twin axis
        ax2=ax.twinx()

        ax2.yaxis.tick_left()
        ax.yaxis.tick_right()

        ax.yaxis.set_label_position('right')
        ax2.yaxis.set_label_position('left')

        ax2.set_ylabel('Frequency [%]')

        for p in ax.patches:
            x=p.get_bbox().get_points()[:,0]
            y=p.get_bbox().get_points()[1,1]
            ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
                    ha='center', va='bottom') # set the alignment of the text
            
        ax.yaxis.set_major_locator(ticker.LinearLocator(11))
        ax2.set_ylim(0,100)
        ax.set_ylim(0,ncount)
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))
    elif typ=="pie":
        fig = plt.figure(figsize=figsize)
        plt.pie([i[0] for i in count], labels=[i[1] for i in count]);
        plt.title("charge")
    if retFig==True:
        return fig


def modAa(df, figsize=(6,6), retFig=False):
    labels = [str(i)+'\n'+str(round(j/df.shape[0]*100,2))+'%' 
              for i,j in zip(df["Amino acid"].value_counts().index, 
                             df["Amino acid"].value_counts().values)]

    fig = plt.figure(figsize=figsize)
    plt.pie(df["Amino acid"].value_counts().values,
           labels=(labels));
    plt.title("Modified AAs")
    if retFig == True:
        return fig


def wordcloud(text, exlusionwords=None, background_color="white", mask=None, file="",
              contour_width=0, **kwargs):
    """
    Parameters
    ----------
    text : text input as a string
    @exlusionwords : list of words to exclude from wordcloud, default: None
    @background_color :: The default is "white".
    @mask :: default is false, set it either to round or true and add a .png file
    @file :: file is given as path with path/to/file.filetype
    Returns
    -------
    figure
​    """

    def extractPDF(file):
        """
        function extract text from PDF files
        @params
        @file :: file is given as path with path/to/file.filetype
        @returns :: returns extracted text as string
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
        wc = WordCloud(background_color="white",stopwords=exlusionwords,width=1800, height=500).generate(text)

    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()

    return wc
    
    
def BHplot(df, ps, adj_ps,title=None, alpha=0.05, zoom=20):
    """
    Function that visualizes FDR correction
    :@param df: dataframe in which p values are stored
    :@param ps: column with p values
    :@param adj_ps: column with adj_p values
    """
    
    n = len(df[ps][df[ps].notnull()])
    x = range(n)
    y = [((i+1)*alpha)/n for i in x]
    
    idx = df[ps][df[ps].notnull()].sort_values().index
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
    ax[0].set_title(title)
    ax[0].plot(x,y,color='gray', label=r'$\frac{i * \alpha}{n}$')
    ax[0].scatter(x, df[ps].loc[idx].sort_values(), label="p_values", color="teal",alpha=0.5)
    ax[0].scatter(x, df[adj_ps].loc[idx],label="adj. p_values", color="purple",alpha=0.5)
    ax[0].legend(fontsize=12)
    
    ax[1].plot(x[:zoom],y[:zoom],color='gray')
    ax[1].scatter(x[:zoom], df[ps].loc[idx].sort_values().iloc[:zoom], label="p_values", color="teal")
    ax[1].scatter(x[:zoom], df[adj_ps].loc[idx][:zoom],label="adj. p_values", color="purple")
    
    sns.despine(ax=ax[0])
    sns.despine(ax=ax[1])