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