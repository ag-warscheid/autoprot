prot = pp.read_csv("../data/proteinGroups_minimal.zip")
prot = pp.cleaning(prot, "proteinGroups")
protInt = prot.filter(regex='Intensity').columns
prot = pp.log(prot, protInt, base=10)

x = "log10_Intensity BC4_3"
y = "log10_Intensity BC36_1"

vis.ma_plot(prot, x, y, fct=2)
plt.show()