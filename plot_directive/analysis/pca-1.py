prot = pd.read_csv("../data/proteinGroups_minimal.zip", sep="\t", low_memory=False)
protRatio = prot.filter(regex="Ratio .\/. normalized").columns
protLog = pp.log(prot, protRatio, base=2)
temp = protLog[~protLog.filter(regex="log2.*norm").isnull().any(axis=1)]
dataframe = temp.filter(regex="log2.*norm.*_1$")
clabels = dataframe.columns
rlabels = None
autopca = ana.AutoPCA(dataframe=dataframe, clabels=clabels, rlabels=rlabels)
autopca.scree()