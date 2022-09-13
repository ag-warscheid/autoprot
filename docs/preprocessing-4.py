import autoprot.preprocessing as pp
import autoprot.visualization as vis
import pandas as pd
phos_lfq = pd.read_csv("_static/testdata/Phospho (STY)Sites_lfq.zip", sep="\t", low_memory=False)