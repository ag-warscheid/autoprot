simplify = {"ERK":["ERK1","ERK2"],
            "GSK3":["GSK3A", "GSK3B"],
            "AMPKA":["AMPKA1","AMPKA2"]}
ksea.ksea(col="logFC_TvC", min_subs=5, simplify=simplify)
ksea.plot_enrichment()