import autoprot.preprocessing as pp
import autoprot.visualization as vis
import autoprot.analysis as ana
import pandas as pd

prot = pd.read_csv("_static/testdata/proteinGroups.zip", sep='\t', low_memory=False)
prot = pp.cleaning(prot, "proteinGroups")
protInt = prot.filter(regex='Intensity').columns
prot = pp.log(prot, protInt, base=10)

x = "log10_Intensity BC4_3"
y = "log10_Intensity BC36_1"

vis.MAPlot(prot, x, y, fct=2,interactive=False)
plt.show()