fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15,10))
vis.volcano(df=prot_limma, logFC="logFC_TvM", p="P.Value_TvM", highlight=idx, annot="Gene names",
           figsize=(15,5), ax=ax[0])
vis.volcano(df=prot_limma, logFC="logFC_TvM", p="P.Value_TvM", highlight=idx, annot="Gene names",
           figsize=(15,5), ax=ax[1])
ax[1].set_ylim(2,4)
ax[1].set_xlim(0,4)
plt.show()