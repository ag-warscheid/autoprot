prot = pp.read_csv("../data/proteinGroups_minimal.zip")
prot = pp.cleaning(prot, "proteinGroups")
protRatio = prot.filter(regex="Ratio .\/. BC.*_1").columns
prot = pp.log(prot, protRatio, base=2)
protRatio = prot.filter(regex="log2_Ratio.*").columns
prot = pp.vsn(prot, protRatio)
protRatio = prot.filter(regex="log2_Ratio.*_1$").columns
labels = [i.split(" ")[1]+"_"+i.split(" ")[-1] for i in protRatio]
vis.boxplot(df=prot,reps=protRatio, compare=False, labels=labels, title="Unnormalized Ratios Boxplot",
        ylabel="log_fc")
plt.show()