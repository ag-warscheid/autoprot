temp["color"] = temp["Multiplicity"]
temp["color"].replace([1,2,3], ["teal", "purple", "salmon"], inplace=True)
rc = {"multiplicity" : temp["color"]}

clusterRes.clusterMap(nCluster=5, makeTraces=True, rowColors=rc,
                      colors=["green", "chartreuse", "blue", "hotpink", "gold"])