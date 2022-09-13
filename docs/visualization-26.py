idx = prot_limma[prot_limma['logFC_TvM'] > 1].sample(10).index
vis.volcano(df=prot_limma, log_fc="logFC_TvM", p="P.Value_TvM", highlight=idx, annot="Gene names",
            figsize=(15,5))
plt.show()