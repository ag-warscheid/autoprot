prot = pp.read_csv("../data/proteinGroups_minimal.zip")
prot = pp.cleaning(prot, "proteinGroups")
protRatio = prot.filter(regex=r"^Ratio .\/.( | normalized )B").columns
prot = pp.log(prot, protRatio, base=2)
prot['Gene names 1st'] = prot['Gene names'].str.split(';').str[0]

fig = vis.ratio_plot(
    prot,
    col_name1='Ratio M/L BC18_1',
    col_name2='Ratio M/L BC18_2',
    ratio_thresh= 3,
    annotate_colname='Gene names 1st',
    xlabel= 'Ratio Rep1',
    ylabel = 'Ratio Rep2',
    annotate_density=20)

fig.show()