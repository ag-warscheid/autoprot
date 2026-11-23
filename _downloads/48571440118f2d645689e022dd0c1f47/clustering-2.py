labels.replace(['setosa', 'virginica', 'versicolor'], ["teal", "purple", "salmon"], inplace=True)
rc = {"species" : labels}
c.vis_cluster(row_colors={'species': labels})