phos = pd.read_csv("../data/Phospho (STY)Sites_minimal.zip", sep="\t", low_memory=False)
phosRatio = phos.filter(regex="^Ratio .\/.( | normalized )R.___").columns
phosLog = pp.log(phos, phosRatio, base=2)
noNorm = phosLog.filter(regex="log2_Ratio ./. R.___").columns
phos_norm_r = pp.cyclic_loess(phosLog, noNorm)
vis.boxplot(phos_norm_r, [noNorm, phos_norm_r.filter(regex="_norm").columns], compare=True)
plt.show()