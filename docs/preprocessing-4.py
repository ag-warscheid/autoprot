import autoprot.preprocessing as pp
import autoprot.visualization as vis
import pandas as pd
phos_lfq = pd.read_csv("_static/testdata/Phospho (STY)Sites_lfq.zip", sep="\t", low_memory=False)
intens_cols = phos_lfq.filter(regex="Intensity .").columns.to_list()
phos_lfq[intens_cols] = phos_lfq[intens_cols].replace(0, np.nan)
phos_lfq = pp.vsn(phos_lfq, intens_cols)
norm_cols = phos_lfq.filter(regex="_norm").columns.to_list()
phos_lfq, log_cols = pp.log(phos_lfq, intens_cols, base=2, return_cols=True)
vis.boxplot(phos_lfq, reps=[log_cols, norm_cols], compare=True)