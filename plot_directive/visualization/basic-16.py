prot = pp.read_csv("../data/proteinGroups_minimal.zip")
prot = pp.cleaning(prot, "proteinGroups")
protRatio = prot.filter(regex="Ratio .\/. BC.*").columns
prot = pp.log(prot, protRatio, base=2)

twitchVsmild = ['Ratio H/M BC18_1','Ratio M/L BC18_2','Ratio H/M BC18_3',
                'Ratio H/L BC36_1','Ratio H/M BC36_2','Ratio M/L BC36_2']

data = prot[twitchVsmild[:3]]
vis.venn_diagram(data, figsize=(5,5))
plt.show()