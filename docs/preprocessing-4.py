import autoprot.preprocessing as pp
import autoprot.visualization as vis
import pandas as pd
phos_lfq = pd.read_csv("_static/testdata/Phospho (STY)Sites_lfq.zip", sep="\t", low_memory=False)
noNorm = phos_lfq.filter(regex="Intensity .").columns
phos_lfq[noNorm] = phos_lfq.filter(regex="Intensity .").replace(0, np.nan)
phos_lfq = pp.vsn(phos_lfq, noNorm)
vis.boxplot(phos_lfq, [noNorm, phos_lfq.filter(regex="_norm").columns], data='Intensity', compare=True)