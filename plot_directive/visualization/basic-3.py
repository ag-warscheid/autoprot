prot = pp.read_csv("../data/proteinGroups_minimal.zip")
mildInt = ["Intensity M BC18_1","Intensity H BC18_2","Intensity M BC18_3",
           "Intensity M BC36_1","Intensity M BC36_2","Intensity H BC36_2"]
prot = pp.log(prot, mildInt, base=10)
mildLogInt = [f"log10_{i}" for i in mildInt]
vis.corr_map(prot,mildLogInt, annot=True)
plt.show()