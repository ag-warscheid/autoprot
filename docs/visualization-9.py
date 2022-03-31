import pandas as pd
import autoprot.preprocessing as pp
import autoprot.visualization as vis

twitchInt = ['Intensity H BC18_1','Intensity M BC18_2','Intensity H BC18_3',
             'Intensity H BC36_1','Intensity H BC36_2','Intensity M BC36_2']
ctrlInt = ["Intensity L BC18_1","Intensity L BC18_2","Intensity L BC18_3",
           "Intensity L BC36_1", "Intensity L BC36_2","Intensity L BC36_2"]
mildInt = ["Intensity M BC18_1","Intensity H BC18_2","Intensity M BC18_3",
           "Intensity M BC36_1","Intensity M BC36_2","Intensity H BC36_2"]

prot = pd.read_csv("_static/testdata/proteinGroups.zip", sep='\t', low_memory=False)
prot = pp.log(prot, twitchInt+ctrlInt+mildInt, base=10)
twitchLogInt = [f"log10_{i}" for i in twitchInt]
mildLogInt = [f"log10_{i}" for i in mildInt]

vis.correlogram(prot,mildLogInt, file='proteinGroups', lowerTriang="hist2d")
plt.show()