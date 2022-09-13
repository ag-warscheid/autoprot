vis.volcano(df=prot_limma, log_fc="logFC_TvM", p="P.Value_TvM", pt=0.01,
   fct=2, annot="Gene names", sig_col="purple", bg_col="teal",
   title="Custom Title", figsize=(15,5))
plt.show()