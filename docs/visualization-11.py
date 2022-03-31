data = pp.log(prot,["Intensity"], base=10)
data = data[["log10_Intensity", "Gene names"]]
data = data[data["log10_Intensity"]!=-np.inf]

vis.intensityRank(data, rankCol="log10_Intensity", label="Gene names", n=15, title="Rank Plot",
                 hline=8, marker="d")