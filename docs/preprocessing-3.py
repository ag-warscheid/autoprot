import autoprot.preprocessing as pp
import autoprot.visualization as vis
import pandas as pd
phos = pd.read_csv("_static/testdata/Phospho (STY)Sites_mod.zip", sep="\t", low_memory=False)
phosRatio = phos.filter(regex="^Ratio .\/.( | normalized )R.___").columns
phosLog = pp.log(phos, phosRatio, base=2)
noNorm = phosLog.filter(regex="log2_Ratio ./. R.___").columns
phos_norm_r = pp.quantileNorm(phosLog, noNorm, backend='r')
vis.boxplot(phos_norm_r, [noNorm, phos_norm_r.filter(regex="_norm").columns], compare=True)
plt.show()