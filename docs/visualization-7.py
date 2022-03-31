import autoprot.preprocessing as pp
import autoprot.visualization as vis

prot = pd.read_csv("_static/testdata/proteinGroups.zip", sep='\t', low_memory=False)
mildInt = ["Intensity M BC18_1","Intensity H BC18_2","Intensity M BC18_3",
           "Intensity M BC36_1","Intensity M BC36_2","Intensity H BC36_2"]
prot = pp.log(prot, mildInt, base=10)
mildLogInt = [f"log10_{i}" for i in mildInt]
vis.corrMap(prot,mildLogInt, annot=True)
plt.show()