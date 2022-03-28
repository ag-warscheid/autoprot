import autoprot.preprocessing as pp
import autoprot.visualization as vis
import pandas as pd
phos = pd.read_csv("_static/testdata/Phospho (STY)Sites_mod.zip", sep="\t", low_memory=False)
forImp = np.log10(phos.filter(regex="Int.*R1").replace(0, np.nan))
impProt = pp.impMinProb(forImp, phos.filter(regex="Int.*R1").columns, width=.4, downshift=2.5)
fig, ax1 = plt.subplots(1)
impProt.filter(regex="Int.*R1")[impProt["Imputed"]==False].mean(1).hist(density=True, bins=50, label="not Imputed", ax=ax1)
impProt.filter(regex="Int.*R1")[impProt["Imputed"]==True].mean(1).hist(density=True, bins=50, label="Imputed", ax=ax1)
plt.legend()
plt.show()