import autoprot.preprocessing as pp
import autoprot.visualization as vis
import autoprot.analysis as ana
import pandas as pd

prot = pd.read_csv("_static/testdata/proteinGroups.zip", sep='\t', low_memory=False)
prot = pp.cleaning(prot, "proteinGroups")
protInt = prot.filter(regex='Intensity').columns
prot = pp.log(prot, protInt, base=10)

twitchInt = ['log10_Intensity H BC18_1','log10_Intensity M BC18_2','log10_Intensity H BC18_3',
         'log10_Intensity BC36_1','log10_Intensity H BC36_2','log10_Intensity M BC36_2']

vis.mean_sd_plot(prot, twitchInt)