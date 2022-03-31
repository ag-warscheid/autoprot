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

phos = ana.ttest(df=phos_expanded, reps=twitchVsmild, cond="TvM", mean=True)
phos = ana.ttest(df=phos_expanded, reps=twitchVsctrl, cond="TvC", mean=True)

ksea = ana.KSEA(phos)
ksea.annotate(organism="mouse", onlyInVivo=True)
ksea.getKinaseOverview(kois=["Akt1","MKK4", "P38A", "Erk1"])
ksea.ksea(col="logFC_TvC", minSubs=5)

ksea.plotEnrichment(up_col="salmon",
                    bg_col="pink",
                    down_col="hotpink")