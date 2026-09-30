fig = vis.volcano(
   df=prot_limma,
   log_fc_colname="logFC_TvM",
   p_colname="P.Value_TvM",
   title="Volcano Plot",
   annotate_colname="Gene names 1st",
)

ax = fig.gca()
ax.axhline(y=3, color='red', linestyle=':')
ax.axhline(y=4, color='blue', linestyle=':')

fig.show()