prot = pp.read_csv("../data/proteinGroups_minimal.zip")
prot = pp.cleaning(prot, "proteinGroups")
protRatio = prot.filter(regex=r"^Ratio .\/.( | normalized )B").columns
prot = pp.log(prot, protRatio, base=2)
twitchVsmild = ['log2_Ratio H/M normalized BC18_1','log2_Ratio M/L normalized BC18_2',
                'log2_Ratio H/M normalized BC18_3',
                'log2_Ratio H/L normalized BC36_1','log2_Ratio H/M normalized BC36_2',
                'log2_Ratio M/L normalized BC36_2']
prot_limma = ana.limma(prot, twitchVsmild, cond="_TvM")
prot_limma['Gene names 1st'] = prot_limma['Gene names'].str.split(';').str[0]

fig = vis.volcano(
    df=prot_limma,
    log_fc_colname="logFC_TvM",
    p_colname="P.Value_TvM",
    title="Volcano Plot",
    annotate_colname="Gene names 1st",
)

fig.show()