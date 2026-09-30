phos = pd.read_csv("../data/Phospho (STY)Sites_minimal.zip", sep="\t", low_memory=False)
forImp = np.log10(phos.filter(regex="Int.*R1").replace(0, np.nan))
impProt = pp.imp_min_prob(forImp, phos.filter(regex="Int.*R1").columns, width=.4, downshift=2.5)
fig, ax1 = plt.subplots(1)
imputed_values = impProt.filter(regex="Int.*R1$").isnull()
ax1.hist(impProt.filter(regex="Int.*R1_min_imputed").values[~imputed_values],
          density=True, bins=50, label="not Imputed", alpha=.5)
ax1.hist(impProt.filter(regex="Int.*R1_min_imputed").values[imputed_values],
          density=True, bins=50, label="Imputed", alpha=.5)
ax1.set_xlabel("log10 Intensity")
ax1.set_ylabel("Density")

plt.legend()
plt.show()