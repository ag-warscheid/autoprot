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