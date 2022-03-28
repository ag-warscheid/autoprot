vis.volcano(df=prot_limma, logFC="logFC_TvM", p="P.Value_TvM", highlight=idx, annot="Gene names",
   figsize=(15,5), highlight_col = "teal", sig_col="lightgray",
   custom_bg = {"s":1, "alpha":.1},
   custom_fg = {"s":5, "alpha":.33},
   custom_hl = {"s":40, "linewidth":1, "edgecolor":"purple"})
plt.show()