non_sig_kwargs = dict(color="black", marker="x")
sig_kwargs = dict(color="red", marker=7, s=100)

fig = vis.volcano(
    df=prot_limma,
    log_fc_colname="logFC_TvM",
    p_colname="P.Value_TvM",
    p_thresh=0.01,
    title="Customised Volcano Plot",
    annotate_colname="Gene names 1st",
    kwargs_ns=non_sig_kwargs,
    kwargs_p_sig=non_sig_kwargs,
    kwargs_log_fc_sig=non_sig_kwargs,
    kwargs_both_sig=sig_kwargs,
)

fig.show()