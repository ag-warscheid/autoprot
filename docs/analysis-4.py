import autoprot.analysis as ana
import autoprot.preprocessing as pp
import pandas as pd

prot = pd.read_csv("_static/testdata/proteinGroups.zip", sep="\t", low_memory=False)
protRatio = prot.filter(regex="Ratio .\/. normalized")
protLog = pp.log(prot, protRatio, base=2)
temp = protLog[~protLog.filter(regex="log2.*norm").isnull().any(1)]
X = temp.filter(regex="log2.*norm.*_1$")
clabels = X.columns
rlabels = np.nan
autopca = ana.autoPCA(X, rlabels, clabels)
autopca.scree()