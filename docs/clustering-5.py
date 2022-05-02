labels.replace(['setosa', 'virginica', 'versicolor'], ["teal", "purple", "salmon"], inplace=True)
rc = {"species" : labels}
c.visCluster(rowColors={'species': labels})