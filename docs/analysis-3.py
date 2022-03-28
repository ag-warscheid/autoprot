simplify = {"ERK":["ERK1","ERK2"],
            "GSK3":["GSK3A", "GSK3B"]}
ksea.ksea(col="logFC_TvC", minSubs=5, simplify=simplify)
ksea.plotEnrichment()