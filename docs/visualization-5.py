protRatioNorm = prot.filter(regex="log2_Ratio.*normalized").columns
vis.boxplot(prot,[protRatio, protRatioNorm], compare=True, labels=labels, title=["unormalized", "normalized"],
           data="logFC")