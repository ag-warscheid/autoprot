import autoprot.analysis as ana
import autoprot.preprocessing as pp
import pandas as pd
twitchVsmild = ['log2_Ratio H/M normalized BC18_1','log2_Ratio M/L normalized BC18_2','log2_Ratio H/M normalized BC18_3',
                'log2_Ratio H/L normalized BC36_1','log2_Ratio H/M normalized BC36_2','log2_Ratio M/L normalized BC36_2']
prot = pd.read_csv("_static/testdata/proteinGroups.zip", sep='\t', low_memory=False)
protRatio = prot.filter(regex="Ratio .\/. normalized")
protLog = pp.log(prot, protRatio, base=2)
prot_tt = ana.ttest(df=protLog, reps=twitchVsmild, cond="TvM", mean=True, adjustPVals=True)
prot_tt["pValue_TvM"].hist(bins=50)
plt.show()