phos = pp.read_csv("../data/Phospho (STY)Sites_minimal.zip")
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

vis.pval_hist(phos,'pValue_MvC', 'adj.pValue_MvC', alpha=0.05, zoom=7)