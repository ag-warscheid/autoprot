c.nclusters = 3
c.makeCluster()
c.cmap = 'coolwarm'
c.visCluster(rowColors={'species': labels}, makeTraces=True, file=None, makeHeatmap=True)