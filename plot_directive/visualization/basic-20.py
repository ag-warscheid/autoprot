fig = vis.volcano(
    df=prot_limma,
    log_fc_colname="logFC_TvM",
    p_colname="P.Value_TvM",
    p_thresh=0.01,
    title="Volcano Plot",
    annotate_colname="Gene names 1st",
)

fig.show()