# -*- coding: utf-8 -*-
"""
Autoprot Clustering Functions.

@author: Wignand, Julian

@documentation: Julian
"""

import seaborn as sns
import pandas as pd
from scipy.stats import zscore
from scipy.spatial import distance
from scipy import cluster as clst
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn import cluster as clstsklearn
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import numpy as np
import os


class Cluster:
    r"""
    Base class for clustering pipelines.
    """

    def __init__(self, data, clabels=None, rlabels=None, zs=None,
                 linkage=None):
        """
        Initialise the class.

        Parameters
        ----------
        data : np.array or pd.DataFrame
            The data to be clustered.
        clabels : list
            Column labels. Must be present in the in input df.
            Defaulting to RangeIndex(0, 1, 2, …, n). 
        rlabels : list
            Row labels. Must be present in the in input df.
            Will default to RangeIndex if no indexing information part of
            input data and no index provided.
        zs : int or None, optional
            Axis along which to calculate the zscore.
            The default is None.
        linkage : scipy.cluster.hierarchy.linkage object, optional
            Precalculated linkage object.
            The default is None.

        Returns
        -------
        None.

        """

        def _sanitizeData(data, clabels, rlabels, zs):
            """
            Check if data contains missing values and remove them.

            Parameters
            ----------
            data : np.array or pd.DataFrame
                The data to be clustered.
            clabels : list-like, optional
                Column labels. May not be present in the input df.
            rlabels : list-like, optional. Row labels.
                May not be present in the input df.
            zscore : int or None, optional
                Axis along which to calculate the zscore.

            Raises
            ------
            ValueError
                If the length of the labels does not fit the data size.

            Returns
            -------
            data.values : np.ndarray
                data without NaN or ztransformed depending on parameters
            rlabels : list
                row labels of the reduced dataset
            clabels : list
                column labels of the reduced dataset

            """
            # make sure this is a DataFrame
            data = pd.DataFrame(data, index=rlabels, columns=clabels)

            # if the zscore is to be calculated (i.e. if zs != None)
            # a dataframe with zscores instead of values is calculated
            if zs is not None:
                temp = data.copy(deep=True).values
                X = zscore(temp, axis=zs)
                data = pd.DataFrame(X, index=rlabels, columns=clabels)

            print(f'Removed {data.isnull().values.sum()} NaN values from the dataframe to prepare for clustering.')
            # no NA values should remain during cluster analysis
            data.dropna(how='any', axis=1, inplace=True)

            return data.values, data.index.tolist(), data.columns.tolist()

        self.data, self.rlabels, self.clabels = _sanitizeData(data, clabels, rlabels, zs)

        # the linkage object for hierarchical clustering
        self.linkage = linkage
        # the number of clusters
        self.nclusters = None
        # list of len(data) with IDs of clusters corresponding to rows
        self.clusterId = None
        # the standard colormap
        # self.cmap = sns.diverging_palette(150, 275, s=80, l=55, n=9)
        self.cmap = matplotlib.cm.viridis

    def visCluster(self, col_cluster=False, make_traces=False,
                   make_heatmap=False, file=None, row_colors=None,
                   colors: list = None, yticklabels="", **kwargs):
        """
        Visualise the clustering.

        Parameters
        ----------
        col_cluster : bool, optional
            Whether to cluster the columns. The default is False.
        make_traces : bool, optional
            Whether to generate traces of each cluster. The default is False.
        make_heatmap : bool, optional
            Whether to generate a summery heatmap.
            The default is False.
        file : str, optional
            Path to the output plot file. The default is None.
        row_colors : dict, optional
            dictionary of mapping a row title to a list of colours.
            The list must have the same length as the data has rows.
            Generates an additional column in the heatmeap showing
            the indicated columns values as colors.
            Has to be same length as provided data.
            The default is None.
        colors : list of str, optional
            Colors for the annotated clusters.
            Has to be the same size as the number of clusters.
            The default is None.
        yticklabels : list of str, optional
            Labels for the y ticks. The default is "".
        **kwargs :
            passed to seaborn.clustermap.
            See https://seaborn.pydata.org/generated/seaborn.clustermap.html
            May also contain 'z-score' that is used during making of
            cluster traces.

        Returns
        -------
        None.

        """

        def makeClusterTraces(self, file, colors: list, zs=None):
            """
            Plot RMSD vs colname line plots.

            Shaded areas representing groups of RMSDs are plotted.

            Parameters
            ----------
            file : str
                Filename with extension to save file to.
                Will be extended by FNAME_traces.EXT.
            colors : list of str or None.
                Colours for the traces. If none, the same predefined colours will
                be used for all n traces.
            zs : int or None, optional
                Axis along which to standardise the data by z-score transformation.
                The default is None.

            Returns
            -------
            None.

            """
            plt.figure(figsize=(5, 5 * self.nclusters))
            temp = pd.DataFrame(self.data.copy())
            if zs is not None:
                # calculate the z-score using scipy using the other axis (i.e. axis=0 if
                # 1 was provided and vice versa)
                temp = pd.DataFrame(zscore(temp, axis=1 - zs))  # seaborn and scipy work opposite
            # ndarray containing the cluster numbers for each data point
            temp["cluster"] = self.clusterId

            labels = list(set(self.clusterId))

            # iterate over the generated clusters
            for idx, i in enumerate(labels):

                ax = plt.subplot(self.nclusters, 1, idx + 1)
                # slice data points belonging to a certain cluster number
                temp2 = temp[temp["cluster"] == i].drop("cluster", axis=1)

                # compute the root mean square deviation of the z-scores or the protein log fold changes
                # as a helper we take the -log of the rmsd in order to plot in the proper sequence
                # i.e. low RMSDs take on large -log values and thereofore are plotted
                # last and on the top
                temp2["distance"] = temp2.apply(lambda x: -np.log(np.sqrt(sum((x - temp2.mean()) ** 2))), 1)

                if temp2.shape[0] == 1:
                    # if cluster contains only 1 entry i.e. one condition
                    ax.set_title(f"Cluster {i + 1}")
                    ax.set_ylabel("")
                    ax.set_xlabel("")
                    # ax.plot(range(temp2.shape[1]-1),temp2.drop("distance", axis=1).values.reshape(3), color=color[idx], alpha=alpha[idx])
                    ax.plot(range(temp2.shape[1] - 1),
                            temp2.drop("distance", axis=1).values.reshape(3))
                    plt.xticks(range(len(self.clabels)), self.clabels)
                    continue

                # bin the RMSDs into five groups
                temp2["distance"] = pd.cut(temp2["distance"], 5)

                # get aestethics for traces
                if colors is None:
                    color = ["#C72119", "#D67155", "#FFC288", "#FFE59E", "#FFFDBF"]
                else:
                    color = [colors[i]] * 5
                color = color[::-1]
                alpha = [0.1, 0.2, 0.25, 0.4, 0.6]

                # group into the five RMSD bins
                grouped = temp2.groupby("distance")

                ax.set_title(f"Cluster {i + 1}")
                if zs is None:
                    ax.set_ylabel("value")
                else:
                    ax.set_ylabel("z-score")
                ax.set_xlabel("Condition")

                # for every RMSD group
                for idx, (i, group) in enumerate(grouped):
                    # for every condition (i.e. colname)
                    for j in range(group.shape[0]):
                        ax.plot(range(temp2.shape[1] - 1),
                                group.drop("distance", axis=1).iloc[j],
                                color=color[idx],
                                alpha=alpha[idx])
                # set the tick labels as the colnames
                plt.xticks(range(len(self.clabels)), self.clabels, rotation=90)
                plt.tight_layout()

                # save to file if asked
                if file is not None:
                    name, ext = file.split('.')
                    filet = f"{name}_traces.{ext}"
                    plt.savefig(filet)

        def makeClusterHeatmap(self, file=None):
            """
            Make summary heatmap of clustering.
    
            Parameters
            ----------
            file : str
                Path to write summary.
    
            Returns
            -------
            None.
            """

            temp = pd.DataFrame(self.data, index=self.rlabels, columns=self.clabels)
            temp["cluster"] = self.clusterId
            grouped = temp.groupby("cluster")[self.clabels].mean()

            ylabel = [f"Cluster{i + 1} (n={j})" for i, j in
                      enumerate(temp.groupby("cluster").count().iloc[:, 0].values)]
            plt.figure()
            plt.title("Summary Of Clustering")
            sns.heatmap(grouped,
                        cmap=self.cmap)
            plt.yticks([i + 0.5 for i in range(len(ylabel))], ylabel, rotation=0)

            plt.tight_layout()

            if file is not None:
                name, ext = file.split('.')
                filet = f"{name}_summary.{ext}"
                plt.savefig(filet)

        # there should be as many colours as clusters
        norm = clrs.Normalize(vmin=self.clusterId.min(),
                              vmax=self.clusterId.max())

        if (colors is not None) and len(colors) == self.ncluster:
            cmap = clrs.ListedColormap(colors)
            mapper = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        else:
            mapper = plt.cm.ScalarMappable(norm=norm, cmap=self.cmap)
        a = mapper.to_rgba(self.clusterId)
        # A 1darray as long as the number of rows in data wtih
        # colours matching the cluster IDs
        clusterColors = np.apply_along_axis(clrs.to_hex, 1, a)

        if "cmap" not in kwargs.keys():
            kwargs["cmap"] = self.cmap

        if row_colors is not None:
            # transform the dict to dataframe
            rowColors_df = pd.DataFrame(row_colors)
            # add a new column with the original cluster colours
            rowColors_df['Cluster'] = clusterColors
            # set the same index as the data to plot
            rowColors_df.index = self.rlabels
        else:
            # if no extra labeling row is provided, use only the list of
            # colours corresponding to cluster names
            # transform to df for use with seaborn
            rowColors_df = pd.DataFrame(clusterColors,
                                        columns=['Cluster'],
                                        index=self.rlabels)

        # We plot usually ratios or zscores
        value_type = 'z-score' if "z_score" in kwargs.keys() else 'value'

        sns.clustermap(pd.DataFrame(self.data,
                                    index=self.rlabels,
                                    columns=self.clabels),  # Rectangular data for clustering. Cannot contain NAs.
                       row_linkage=self.linkage,  # cluster linkage
                       row_colors=rowColors_df,  # list of colours or pd.DataFrame with colours
                       col_cluster=col_cluster,  # cluster the columns (y/n)
                       yticklabels=yticklabels,  # set coname labels
                       cbar_kws={'label': value_type},
                       **kwargs)

        # save the file if necessary
        if file is not None:
            plt.savefig(file)

        # compute RMSD traces for the deviations of values in each cluster
        # separated by condition
        if make_traces == True:
            if "z_score" in kwargs.keys():
                makeClusterTraces(self, file, zs=kwargs["z_score"], colors=colors)
            else:
                makeClusterTraces(self, file, colors=colors)

        # generate a summary
        if make_heatmap == True:
            makeClusterHeatmap(self, file)

    def returnCluster(self):
        """Return dataframe with clustered data."""
        temp = pd.DataFrame(self.data, index=self.rlabels, columns=self.clabels)
        temp["cluster"] = self.clusterId
        return temp

    def writeClusterFiles(self, rootdir):
        """
        Generate a folder with text files for each cluster.

        Parameters
        ----------
        rootdir : str
            Path to target dir.
            If the folder is named clusterResults, text files will be saved
            within.
            Else a new folder clusterResults will be created.

        Returns
        -------
        None.

        """
        path = os.path.join(rootdir, "clusterResults")
        if "clusterResults" not in os.listdir(rootdir):
            os.mkdir(path)

        temp = pd.DataFrame(self.data, index=self.rlabels, columns=self.clabels)
        temp["cluster"] = self.clusterId
        for cluster in temp["cluster"].unique():
            pd.DataFrame(temp[temp["cluster"] == cluster].index).to_csv(f"{path}/cluster_{cluster}.tsv", header=False,
                                                                        index=False)


class HCA(Cluster):
    r"""
    Conduct hierarchical cluster analysis.

    Parameters
    ----------
    startProcessingAt : int
        Position in the autoHCA to start after user intervention
    stopProcessingAt : int
        Position in the autoHCA to stop so that the user can intervene

    Notes
    -----
    User provides dataframe and can afterwards use various metrics and methods to perfom and evaluate
    clustering.
    
    StandarWorkflow:
    makeLinkage() -> findNClusters() -> makeCluster()
    
    Examples
    --------
    First grab a dataset that will be used for clustering such as the iris dataset.
    Extract the species labelling from the dataframe as it cannot be used for
    clustering and will be used later to evaluate the result.
    
    >>> import seaborn as sns
    >>> df = sns.load_dataset('iris')
    >>> labels = df.pop('species')
    
    Initialise the clustering class with the data and find the optimum number of
    clusters and generate the final clustering with the autoRun method.
    
    >>> import autoprot.clustering as clst
    >>> c = clst.HCA(df)
    Removed 0 NaN values from the dataframe to prepare for clustering.
    
    >>> c.autoRun()
    Best Davies Boulding at 2 with 0.38275284210068616
    Best Silhouoette_score at 2 with 0.6867350732769781
    Best Harabasz/Calinski at 2 with 502.82156350235897
    Using Davies Boulding Score for setting # clusters: 2
    You may manually overwrite this by setting self.nclusters
    
    .. plot::
        :context: close-figs
    
        import seaborn as sns
        import autoprot.clustering as clst
        
        df = sns.load_dataset('iris')
        labels = df.pop('species')
        c = clst.HCA(df)
        c.autoRun()
    
    Finally visualise the clustering using the visCluster method and include the
    previously extracted labeling column from the original dataframe.
    
    >>> labels.replace(['setosa', 'virginica', 'versicolor'], ["teal", "purple", "salmon"], inplace=True)
    >>> rc = {"species" : labels}
    >>> c.visCluster(row_colors={'species': labels})
    
     .. plot::
         :context: close-figs
    
         labels.replace(['setosa', 'virginica', 'versicolor'], ["teal", "purple", "salmon"], inplace=True)    
         rc = {"species" : labels}
         c.visCluster(rowColors={'species': labels})
         
    HCA separates the setosa quite well but virginica and versicolor are harder.
    When we manually pick true the number of clusters, HCA performs only slightly
    better von this dataset. Note that you can change the default cmap for the
    class by changing the cmap attribute.
    
    >>> c.nclusters = 3
    >>> c.makeCluster()
    >>> c.cmap = 'coolwarm'
    >>> c.visCluster(row_colors={'species': labels}, make_traces=True, file=None, make_heatmap=True)
    
     .. plot::
         :context: close-figs
    
            c.nclusters = 3  
            c.makeCluster()
            c.cmap = 'coolwarm'
            c.visCluster(rowColors={'species': labels}, makeTraces=True, file=None, make_heatmap=True)
    """

    def makeLinkage(self, method='single', metric: str = 'euclidean'):
        """
        Perform hierarchical clustering on the data.

        Parameters
        ----------
        method : str
            Which method is used for the clustering.
            Possible are 'single', 'average' and 'complete' and all values
            for method of scipy.cluster.hierarchy.linkage
            See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
        metric : str
            Which metric is used to calculate distance.
            Possible values are 'pearson', 'spearman' and all metrics
            implemented in scipy.spatial.distance.pdist
            See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html

        Returns
        -------
        None.

        """

        def asDist(c):
            """
            Convert a matrix (i.e. correlation matrix) into a distance matrix for hierachical clustering.

            Parameters
            ----------
            c : np.ndarray
                Input matrix.

            Returns
            -------
            list
                List corresponsding to left off-diagonal elememnts of the
                correlation matrix.

            Examples
            --------
            >>> a = [
            ...     [0.1, .32, .2,  0.4, 0.8],
            ...     [.23, .18, .56, .61, .12],
            ...     [.9,   .3,  .6,  .5,  .3],
            ...     [.34, .75, .91, .19, .21]
            ...      ]
            >>> c = np.corrcoef(a)
            >>> c
            array([[ 1.        , -0.35153114, -0.74736506, -0.48917666],
                   [-0.35153114,  1.        ,  0.23810227,  0.15958285],
                   [-0.74736506,  0.23810227,  1.        , -0.03960706],
                   [-0.48917666,  0.15958285, -0.03960706,  1.        ]])
            >>> autoprot.autoHCA.asDist(c)
            [-0.3515311393849671,
             -0.7473650573493561,
             -0.4891766567441463,
             0.23810227412143423,
             0.15958285448266604,
             -0.03960705975653923]
            """
            return [c[i][j] for i in (range(c.shape[0])) for j in (range(c.shape[1])) if i < j]

        # First calculate a distance metric between the points
        if metric in ["pearson", "spearman"]:
            corr = pd.DataFrame(self.data).T.corr(metric).values
            dist = asDist(1 - corr)
        else:
            dist = distance.pdist(self.data, metric=metric)
        # perform hierarchical clustering using the distance metric
        # the returned matrix self.linkage contains n-1 x 4 elements
        # with each row representing
        # cluster1, cluster2, distance_between_1_and_2,
        # number_of_observations_in_the_cluster
        self.linkage = clst.hierarchy.linkage(dist, method=method)

    def findNClusters(self, start=2, upTo=20, figsize=(15, 5), plot=True):
        """
        Evaluate number of clusters.

        Parameters
        ----------
        start : int, optional
            The minimum number of clusters to plot. The default is 2.
        upTo : int, optional
            The maximum number of clusters to plot. The default is 20.
        figsize : tuple of float or int, optional
            The size of the plotted figure.
            The default is (15,5).
        plot : bool, optional
            Whether to plot the corresponding figures for the cluster scores

        Notes
        -----
        Davies-Bouldin score:
            The score is defined as the average similarity measure of each
            cluster with its most similar cluster, where similarity is the
            ratio of within-cluster distances to between-cluster distances.
            Thus, clusters which are farther apart and less dispersed will
            result in a better score.
            The minimum score is zero, with lower values indicating better
            clustering.
        Silhouette score:
            The Silhouette Coefficient is calculated using the mean
            intra-cluster distance (a) and the mean nearest-cluster
            distance (b) for each sample. The Silhouette Coefficient for a
            sample is (b - a) / max(a, b). To clarify, b is the distance
            between a sample and the nearest cluster that the sample is not a
            part of. Note that Silhouette Coefficient is only defined if
            number of labels is 2 <= n_labels <= n_samples - 1.
            The best value is 1 and the worst value is -1. Values near 0
            indicate overlapping clusters. Negative values generally indicate
            that a sample has been assigned to the wrong cluster, as a
            different cluster is more similar.
        Harabasz score:
            It is also known as the Variance Ratio Criterion.
            The score is defined as ratio between the within-cluster dispersion
            and the between-cluster dispersion.

        Returns
        -------
        None.

        """
        upTo += 1
        pred = []
        for i in range(start, upTo):
            # return the assigned cluster labels for each data point
            cluster = clst.hierarchy.fcluster(self.linkage, t=i, criterion='maxclust')
            # calculate scores based on assigned cluster labels and
            # the original data points
            pred.append((davies_bouldin_score(self.data, cluster),
                         silhouette_score(self.data, cluster),
                         calinski_harabasz_score(self.data, cluster)))

        pred = np.array(pred)
        plt.figure(figsize=figsize)
        plt.subplot(131)
        plt.title("Davies_boulding_score")
        plt.plot(pred[::, 0])
        plt.xticks(range(upTo - start), range(start, upTo), rotation=90)
        plt.grid(axis='x')
        plt.subplot(132)
        plt.title("Silhouoette_score")
        plt.plot(pred[::, 1])
        plt.xticks(range(upTo - start), range(start, upTo), rotation=90)
        plt.grid(axis='x')
        plt.subplot(133)
        plt.title("Harabasz score")
        plt.plot(pred[::, 2])
        plt.xticks(range(upTo - start), range(start, upTo), rotation=90)
        plt.grid(axis='x')
        print(f"Best Davies Boulding at {start + list(pred[::, 0]).index(min(pred[::, 0]))} with {min(pred[::, 0])}")
        print(f"Best Silhouoette_score at {start + list(pred[::, 1]).index(max(pred[::, 1]))} with {max(pred[::, 1])}")
        print(f"Best Harabasz/Calinski at {start + list(pred[::, 2]).index(max(pred[::, 2]))} with {max(pred[::, 2])}")
        self.nclusters = start + list(pred[::, 0]).index(min(pred[::, 0]))
        print(f"Using Davies Boulding Score for setting # clusters: {self.nclusters}")
        print("You may manually overwrite this by setting self.nclusters")

    def makeCluster(self):
        """
        Form flat clusters from the hierarchical clustering of linkage.

        Parameters
        ----------
        n : int
            Max number of clusters.
        colors : None or list of RGB_tuples
            Colors for the clusters.
            If none, new colors are generated.

        Returns
        -------
        None.

        """
        if self.nclusters is None:
            raise Exception('No. of clusters is None. Cannot perform flattening.')

        # self.cluster is an array of length x
        # with x = number of original data points containing the ID
        # of the corresponding cluster
        self.clusterId = \
            clst.hierarchy.fcluster(self.linkage,  # the hierarchical clustering
                                    t=self.nclusters,  # max number of clusters
                                    criterion="maxclust")  # forms maximumum n=t clusters

    def autoRun(self, startProcessingAt=1, stopProcessingAt=5):
        """
        Automatically run the clustering pipeline with standard settings.

        Parameters
        ----------
        startProcessingAt : int, optional
            Step of the pipeline to start. The default is 1.
        stopProcessingAt : int, optional
            Step of the pipeline to stop. The default is 5.

        Notes
        -----
        The pipeline currently consists of (1) makeLinkage, (2) findNClusters
        and (3) makeCluster.

        Returns
        -------
        None.

        """
        if startProcessingAt <= 1:
            self.makeLinkage()
        if startProcessingAt <= 2 and stopProcessingAt >= 2:
            self.findNClusters()
        if startProcessingAt <= 3 and stopProcessingAt >= 3:
            self.makeCluster()


class KMeans(Cluster):
    """
    Perform KMeans clustering on a dataset.

    Parameters
    ----------
    n_clusters : int
        The number of clusters to separate.

    Returns
    -------
    None.

    Notes
    -----
    The functions uses scipy.cluster.vq.kmeans2
    (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.kmeans2.html#scipy.cluster.vq.kmeans2)

    References
    ----------
    D. Arthur and S. Vassilvitskii, “k-means++: the advantages of careful seeding”, Proceedings of the Eighteenth Annual ACM-SIAM Symposium on Discrete Algorithms, 2007.

    Examples
    --------

    First grab a dataset that will be used for clustering such as the iris dataset.
    Extract the species labelling from the dataframe as it cannot be used for
    clustering and will be used later to evaluate the result.
    
    >>> import seaborn as sns
    >>> df = sns.load_dataset('iris')
    >>> labels = df.pop('species')

    Initialise the clustering class with the data and find the optimum number of
    clusters and generate the final clustering with the autoRun method.
    
    >>> import autoprot.clustering as clst
    >>> c = clst.KMeans(df)
    Removed 0 NaN values from the dataframe to prepare for clustering.
    >>> c.autoRun()
    Best Davies Boulding at 2 with 0.40429283717304343
    Best Silhouette_score at 2 with 0.6810461692117465
    Best Harabasz/Calinski at 3 with 561.5937320156642
    Using Davies Boulding Score for setting # clusters: 2
    You may manually overwrite this by setting self.nclusters
    
    .. plot::
        :context: close-figs

        import seaborn as sns
        import autoprot.clustering as clst
        
        df = sns.load_dataset('iris')
        labels = df.pop('species')
        c = clst.KMeans(df)
        c.autoRun()
    
    Finally visualise the clustering using the visCluster method and include the
    previously extracted labeling column from the original dataframe.
    
    >>> labels.replace(['setosa', 'virginica', 'versicolor'], ["teal", "purple", "salmon"], inplace=True)
    >>> rc = {"species" : labels}
    >>> c.visCluster(row_colors={'species': labels})

     .. plot::
         :context: close-figs

         labels.replace(['setosa', 'virginica', 'versicolor'], ["teal", "purple", "salmon"], inplace=True)    
         rc = {"species" : labels}
         c.visCluster(rowColors={'species': labels})
         
    As you can see can KMeans quite well separate setosa but virginica and versicolor are harder.
    When we manually pick the number of clusters, it gets a bit better
    
    >>> c.nclusters = 3
    >>> c.makeCluster()
    >>> c.visCluster(row_colors={'species': labels}, make_traces=True, file=None, make_heatmap=True)
    
     .. plot::
         :context: close-figs

            c.nclusters = 3  
            c.makeCluster()
            c.visCluster(rowColors={'species': labels}, makeTraces=True, file=None, make_heatmap=True)
    """

    def findNClusters(self, start=2, upTo=20, figsize=(15, 5), plot=True, algo='scipy'):
        """
        Evaluate number of clusters.

        Parameters
        ----------
        start : int, optional
            The minimum number of clusters to plot. The default is 2.
        upTo : int, optional
            The maximum number of clusters to plot. The default is 20.
        figsize : tuple of float or int, optional
            The size of the plotted figure.
            The default is (15,5).
        plot : bool, optional
            Whether to plot the corresponding figures for the cluster scores
        algo : str, optional
            Algorith to use for KMeans Clustering. Either "scipy" or "sklearn"

        Notes
        -----
        Davies-Bouldin score:
            The score is defined as the average similarity measure of each
            cluster with its most similar cluster, where similarity is the
            ratio of within-cluster distances to between-cluster distances.
            Thus, clusters which are farther apart and less dispersed will
            result in a better score.
            The minimum score is zero, with lower values indicating better
            clustering.
        Silhouette score:
            The Silhouette Coefficient is calculated using the mean
            intra-cluster distance (a) and the mean nearest-cluster
            distance (b) for each sample. The Silhouette Coefficient for a
            sample is (b - a) / max(a, b). To clarify, b is the distance
            between a sample and the nearest cluster that the sample is not a
            part of. Note that Silhouette Coefficient is only defined if
            number of labels is 2 <= n_labels <= n_samples - 1.
            The best value is 1 and the worst value is -1. Values near 0
            indicate overlapping clusters. Negative values generally indicate
            that a sample has been assigned to the wrong cluster, as a
            different cluster is more similar.
        Harabasz score:
            It is also known as the Variance Ratio Criterion.
            The score is defined as ratio between the within-cluster dispersion
            and the between-cluster dispersion.

        Returns
        -------
        None.

        """
        upTo += 1
        pred = []
        for i in range(start, upTo):

            if algo == 'scipy':
                # return the assigned cluster labels for each data point
                _, cluster = clst.vq.kmeans2(data=self.data,
                                             k=i,
                                             minit='++')
            elif algo == 'sklearn':
                model = clstsklearn.KMeans(n_clusters=i)
                model.fit(self.data)
                cluster = model.labels_
            else:
                raise ValueError('Provide either "sklearn" or "scipy" as parameter for the algo kwarg.')

            # calculate scores based on assigned cluster labels and
            # the original data points
            pred.append((davies_bouldin_score(self.data, cluster),
                         silhouette_score(self.data, cluster),
                         calinski_harabasz_score(self.data, cluster)))

        pred = np.array(pred)
        plt.figure(figsize=figsize)
        plt.subplot(131)
        plt.title("Davies_boulding_score")
        plt.plot(pred[::, 0])
        plt.xticks(range(upTo - start), range(start, upTo), rotation=90)
        plt.grid(axis='x')
        plt.subplot(132)
        plt.title("Silhouoette_score")
        plt.plot(pred[::, 1])
        plt.xticks(range(upTo - start), range(start, upTo), rotation=90)
        plt.grid(axis='x')
        plt.subplot(133)
        plt.title("Harabasz score")
        plt.plot(pred[::, 2])
        plt.xticks(range(upTo - start), range(start, upTo), rotation=90)
        plt.grid(axis='x')
        print(f"Best Davies Boulding at {start + list(pred[::, 0]).index(min(pred[::, 0]))} with {min(pred[::, 0])}")
        print(f"Best Silhouette_score at {start + list(pred[::, 1]).index(max(pred[::, 1]))} with {max(pred[::, 1])}")
        print(f"Best Harabasz/Calinski at {start + list(pred[::, 2]).index(max(pred[::, 2]))} with {max(pred[::, 2])}")
        self.nclusters = start + list(pred[::, 0]).index(min(pred[::, 0]))
        print(f"Using Davies Boulding Score for setting # clusters: {self.nclusters}")
        print("You may manually overwrite this by setting self.nclusters")

    def makeCluster(self, algo='scipy', **kwargs):
        """
        Perform k-means clustering and store the resulting labels in self.clusterId.
        
        Parameters
        ----------
        algo : str, optional
            Algorith to use for KMeans Clustering. Either "scipy" or "sklearn"
        **kwargs:
            passed to either scipy or sklearn kmeans

        Returns
        -------
        None.

        """
        if algo == 'scipy':
            centroids, self.clusterId = clst.vq.kmeans2(data=self.data,
                                                        k=self.nclusters,
                                                        minit='++',
                                                        **kwargs)
        elif algo == 'sklearn':
            # initialise model
            model = clstsklearn.KMeans(n_clusters=self.nclusters,
                                       **kwargs)
            model.fit(self.data)
            self.clusterId = model.labels_
        else:
            raise ValueError('Provide either "sklearn" or "scipy" as parameter for the algo kwarg.')

    def autoRun(self, startProcessingAt=1, stopProcessingAt=5):
        """
        Automatically run the clustering pipeline with standard settings.

        Parameters
        ----------
        startProcessingAt : int, optional
            Step of the pipeline to start. The default is 1.
        stopProcessingAt : int, optional
            Step of the pipeline to stop. The default is 5.

        Notes
        -----
        The pipeline currently consists of (1) findNClusters
        and (2) makeCluster.

        Returns
        -------
        None.

        """
        if startProcessingAt <= 1:
            self.findNClusters()
        if startProcessingAt <= 2 and stopProcessingAt >= 2:
            self.makeCluster()
