fig = vis.volcano(
    df=prot_limma,
    log_fc_colname="logFC_TvM",
    p_colname="P.Value_TvM",
    pointsize_colname='iBAQ',
    pointsize_scaler=5,
    title="Volcano Plot",
    annotate_colname="Gene names 1st",
)

fig.show()