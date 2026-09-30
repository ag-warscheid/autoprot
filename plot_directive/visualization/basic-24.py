to_highlight = prot_limma[prot_limma['iBAQ'] > 10e8].index

fig = vis.volcano(
    df=prot_limma,
    log_fc_colname="logFC_TvM",
    p_colname="P.Value_TvM",
    highlight=to_highlight,
    annotate='highlight',
    title="Volcano Plot",
    annotate_colname="Gene names 1st",
)

fig.show()