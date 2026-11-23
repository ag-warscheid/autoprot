twitchInt = ['Intensity H BC18_1','Intensity M BC18_2','Intensity H BC18_3',
             'Intensity H BC36_1','Intensity H BC36_2','Intensity M BC36_3']
ctrlInt = ["Intensity L BC18_1","Intensity L BC18_2","Intensity L BC18_3",
           "Intensity L BC36_1", "Intensity L BC36_2","Intensity L BC36_3"]
mildInt = ["Intensity M BC18_1","Intensity H BC18_2","Intensity M BC18_3",
           "Intensity M BC36_1","Intensity M BC36_2","Intensity H BC36_3"]

prot = pp.read_csv("../data/proteinGroups_minimal.zip")
prot = pp.log(prot, twitchInt+ctrlInt+mildInt, base=10)
twitchLogInt = [f"log10_{i}" for i in twitchInt]
mildLogInt = [f"log10_{i}" for i in mildInt]

vis.correlogram(prot,mildLogInt, file='proteinGroups', lower_triang="hist2d")
plt.show()