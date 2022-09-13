import autoprot.preprocessing as pp
import autoprot.visualization as vis
import pandas as pd

phos = pd.read_csv("_static/testdata/Phospho (STY)Sites_mod.zip", sep="\t", low_memory=False)
phos = pp.cleaning(phos, file = "Phospho (STY)")
vis.sty_count_plot(phos, typ="bar")
plt.show()