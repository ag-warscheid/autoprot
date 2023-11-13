# -*- coding: utf-8 -*-
"""
Autoprot Analysis Functions.

@author: Wignand, Julian, Johannes

@documentation: Julian
"""
import os
from typing import Union, Literal
from numpy.typing import ArrayLike
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pylab as plt
import matplotlib.colors as clrs
import seaborn as sns
from scipy.stats import zscore
from scipy.spatial import distance
from scipy import cluster as clst
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn import cluster as clstsklearn
import warnings

from .. import r_helper

from gprofiler import GProfiler

gp = GProfiler(
    user_agent="autoprot",
    return_dataframe=True)
RFUNCTIONS, R = r_helper.return_r_path()

# check where this is actually used and make it local
cmap = sns.diverging_palette(150, 275, s=80, l=55, n=9)


class _Cluster:
    r"""
    Base class for clustering pipelines.
    """

    def __init__(self, data: Union[np.array, pd.DataFrame], clabels: Union[None, list] = None,
                 rlabels: Union[None, list] = None, zs: Union[None, int] = None,
                 linkage: Union[None, ArrayLike] = None):
        """
        Initialise the class.

        Parameters
        ----------
        data : np.array or pd.DataFrame
            The data to be clustered.
        clabels : list or None
            Column labels. Must be present in the in input df.
            Defaulting to RangeIndex(0, 1, 2, …, n). 
        rlabels : list or None
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

        def _sanitize_data(data: Union[np.ndarray, pd.DataFrame], clabels: list, rlabels: list,
                           zs: Union[int, None]) -> tuple[ArrayLike, list, list]:
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
            zs : int or None, optional
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
            dataframe = pd.DataFrame(data, index=rlabels, columns=clabels)

            # if the zscore is to be calculated (i.e. if zs != None)
            # a dataframe with zscores instead of values is calculated
            if zs is not None:
                temp = dataframe.copy(deep=True).to_numpy()
                temp_transformed = zscore(temp, axis=zs)
                dataframe = pd.DataFrame(temp_transformed, index=dataframe.index, columns=dataframe.columns)

            print(f'Removed {dataframe.isnull().values.sum()} NaN values from the dataframe to prepare for clustering.')
            # no NA values should remain during cluster analysis
            dataframe.dropna(how='any', axis=1, inplace=True)

            return dataframe.values, dataframe.index.tolist(), dataframe.columns.tolist()

        # noinspection PyTupleAssignmentBalance
        self.data, self.rlabels, self.clabels = _sanitize_data(data=data, clabels=clabels, rlabels=rlabels, zs=zs)

        # the linkage object for hierarchical clustering
        self.linkage = linkage
        # the number of clusters
        self.nclusters = None
        # list of len(data) with IDs of clusters corresponding to rows
        self.clusterId = None
        # the standard colormap
        # self.cmap = sns.diverging_palette(150, 275, s=80, l=55, n=9)
        self.cmap = matplotlib.cm.viridis

    def vis_cluster(self, col_cluster=False, make_traces=False, make_heatmap=False, file=None, row_colors=None,
                    colors: list = None, ytick_labels="", **kwargs):
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
        ytick_labels : list of str, optional
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

        def make_cluster_traces(file, colors: list, zs=None):
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
                temp = pd.DataFrame(zscore(temp, axis=1 - zs))
            temp["cluster"] = self.clusterId
            labels = list(set(self.clusterId))
            for idx, i in enumerate(labels):
                ax = plt.subplot(self.nclusters, 1, idx + 1)
                temp2 = temp[temp["cluster"] == i].drop("cluster", axis=1)
                temp2["distance"] = temp2.apply(lambda x: -np.log(np.sqrt(sum((x - temp2.mean()) ** 2))), 1)

                if temp2.shape[0] == 1:
                    ax.set_title(f"Cluster {i}")
                    ax.set_ylabel("")
                    ax.set_xlabel("")
                    ax.plot(range(temp2.shape[1] - 1), temp2.drop("distance", axis=1).values.reshape(-1))

                    plt.xticks(range(len(self.clabels)), self.clabels)
                    continue
                temp2["distance"] = pd.cut(temp2["distance"], 5)
                if colors is None:
                    color = ["#C72119", "#D67155", "#FFC288", "#FFE59E", "#FFFDBF"]
                else:
                    color = [colors[i]] * 5
                color = color[::-1]
                alpha = [0.1, 0.2, 0.25, 0.4, 0.6]
                grouped = temp2.groupby("distance")
                ax.set_title(f"Cluster {i}")
                if zs is None:
                    ax.set_ylabel("value")
                else:
                    ax.set_ylabel("z-score")
                ax.set_xlabel("Condition")
                for jdx, (_, group) in enumerate(grouped):
                    for j in range(group.shape[0]):
                        ax.plot(range(temp2.shape[1] - 1), group.drop("distance", axis=1).iloc[j], color=color[jdx],
                                alpha=alpha[jdx])

                plt.xticks(range(len(self.clabels)), self.clabels, rotation=90)
                plt.tight_layout()
                if file is not None:
                    name, ext = file.split('.')
                    filet = f"{name}_traces.{ext}"
                    plt.savefig(filet)

        def make_cluster_heatmap(file=None):
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
            sns.heatmap(grouped, cmap=self.cmap)
            plt.yticks([i + 0.5 for i in range(len(ylabel))], ylabel, rotation=0)
            plt.tight_layout()
            if file is not None:
                name, ext = file.split('.')
                filet = f"{name}_summary.{ext}"
                plt.savefig(filet)

        norm = clrs.Normalize(vmin=self.clusterId.min(), vmax=self.clusterId.max())
        if colors is not None and len(colors) == self.nclusters:
            cmap = clrs.ListedColormap(colors)
            mapper = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        else:
            mapper = plt.cm.ScalarMappable(norm=norm, cmap=self.cmap)
        a = mapper.to_rgba(self.clusterId)
        cluster_colors = np.apply_along_axis(clrs.to_hex, 1, a)
        if "cmap" not in kwargs.keys():
            kwargs["cmap"] = self.cmap
        if row_colors is not None:
            row_colors_df = pd.DataFrame(row_colors)
            row_colors_df['Cluster'] = cluster_colors
            row_colors_df.index = self.rlabels
        else:
            row_colors_df = pd.DataFrame(cluster_colors, columns=['Cluster'], index=self.rlabels)

        value_type = 'z-score' if "z_score" in kwargs else 'value'
        sns.clustermap(pd.DataFrame(self.data, index=self.rlabels, columns=self.clabels), row_linkage=self.linkage,
                       row_colors=row_colors_df, col_cluster=col_cluster, yticklabels=ytick_labels,
                       cbar_kws={'label': value_type}, **kwargs)

        if file is not None:
            plt.savefig(file)
        if make_traces:
            if "z_score" in kwargs:
                make_cluster_traces(file, zs=kwargs["z_score"], colors=colors)
            else:
                make_cluster_traces(file, colors=colors)
        if make_heatmap:
            make_cluster_heatmap(file)

    def return_cluster(self):
        """Return dataframe with clustered data."""
        temp = pd.DataFrame(self.data, index=self.rlabels, columns=self.clabels)
        temp["cluster"] = self.clusterId
        return temp

    def write_cluster_files(self, root_dir):
        """
        Generate a folder with text files for each cluster.

        Parameters
        ----------
        root_dir : str
            Path to target dir.
            If the folder is named clusterResults, text files will be saved
            within.
            Else a new folder clusterResults will be created.

        Returns
        -------
        None.

        """
        path = os.path.join(root_dir, "clusterResults")
        if "clusterResults" not in os.listdir(root_dir):
            os.mkdir(path)

        temp = pd.DataFrame(self.data, index=self.rlabels, columns=self.clabels)
        temp["cluster"] = self.clusterId
        for cluster in temp["cluster"].unique():
            pd.DataFrame(temp[temp["cluster"] == cluster].index).to_csv(f"{path}/cluster_{cluster}.tsv", header=False,
                                                                        index=False)

    def clustering_evaluation(self, pred, figsize, start, up_to, plot: bool):
        pred = np.array(pred)
        print(f"Best Davies Boulding at {start + list(pred[::, 0]).index(min(pred[::, 0]))} with {min(pred[::, 0])}")
        print(f"Best Silhouoette_score at {start + list(pred[::, 1]).index(max(pred[::, 1]))} with {max(pred[::, 1])}")
        print(f"Best Harabasz/Calinski at {start + list(pred[::, 2]).index(max(pred[::, 2]))} with {max(pred[::, 2])}")
        self.nclusters = start + list(pred[::, 0]).index(min(pred[::, 0]))
        print(f"Using Davies Boulding Score for setting # clusters: {self.nclusters}")
        print("You may manually overwrite this by setting self.nclusters")
        if plot:
            plt.figure(figsize=figsize)
            plt.subplot(131)
            plt.title("Davies_boulding_score")
            plt.plot(pred[::, 0])
            plt.xticks(range(up_to - start), range(start, up_to), rotation=90)
            plt.grid(axis='x')
            plt.subplot(132)
            plt.title("Silhouoette_score")
            plt.plot(pred[::, 1])
            plt.xticks(range(up_to - start), range(start, up_to), rotation=90)
            plt.grid(axis='x')
            plt.subplot(133)
            plt.title("Harabasz score")
            plt.plot(pred[::, 2])
            plt.xticks(range(up_to - start), range(start, up_to), rotation=90)
            plt.grid(axis='x')


class HCA(_Cluster):
    # noinspection PyUnresolvedReferences
    r"""
    Conduct hierarchical cluster analysis.

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

    >>> from autoprot import analysis as ana
    >>> c = ana.HCA(df)
    Removed 0 NaN values from the dataframe to prepare for clustering.

    >>> c.auto_run()
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
        c.auto_run()

    Finally visualise the clustering using the visCluster method and include the
    previously extracted labeling column from the original dataframe.

    >>> labels.replace(['setosa', 'virginica', 'versicolor'], ["teal", "purple", "salmon"], inplace=True)
    >>> rc = {"species" : labels}
    >>> c.vis_cluster(row_colors={'species': labels})

     .. plot::
         :context: close-figs

         labels.replace(['setosa', 'virginica', 'versicolor'], ["teal", "purple", "salmon"], inplace=True)
         rc = {"species" : labels}
         c.vis_cluster(row_colors={'species': labels})

    HCA separates the setosa quite well but virginica and versicolor are harder.
    When we manually pick true the number of clusters, HCA performs only slightly
    better von this dataset. Note that you can change the default cmap for the
    class by changing the cmap attribute.

    >>> c.nclusters = 3
    >>> c.make_cluster()
    >>> c.cmap = 'coolwarm'
    >>> c.vis_cluster(row_colors={'species': labels}, make_traces=True, file=None, make_heatmap=True)

     .. plot::
         :context: close-figs

            c.nclusters = 3
            c.make_cluster()
            c.cmap = 'coolwarm'
            c.vis_cluster(row_colors={'species': labels}, make_traces=True, file=None, make_heatmap=True)
        """

    def make_linkage(self, method='single',
                     metric: Literal['braycurtis', 'canberra', 'chebyshev', 'cityblock',
                                     'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard',
                                     'jensenshannon', 'kulczynski1', 'mahalanobis', 'matching', 'minkowski',
                                     'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
                                     'sokalsneath', 'sqeuclidean', 'yule', 'spearman', 'pearson'] = 'euclidean'):

        """
        Perform hierarchical clustering on the data.

        Parameters
        ----------
        method : str
            Which method is used for the clustering.
            Possible are 'single', 'average' and 'complete' and all values
            for method of scipy.cluster.hierarchy.linkage
            See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
        metric : str or function
            Which metric is used to calculate distance.
            Possible values are 'pearson', 'spearman' and all metrics
            implemented in scipy.spatial.distance.pdist
            See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html

        Returns
        -------
        None.

        """

        def as_dist(c):
            # noinspection PyUnresolvedReferences
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
            >>> np.corrcoef(np.array(a))
            array([[ 1.        , -0.35153114, -0.74736506, -0.48917666],
                   [-0.35153114,  1.        ,  0.23810227,  0.15958285],
                   [-0.74736506,  0.23810227,  1.        , -0.03960706],
                   [-0.48917666,  0.15958285, -0.03960706,  1.        ]])
            >>> autoprot.autoHCA.as_dist(c)
            [-0.3515311393849671,
             -0.7473650573493561,
             -0.4891766567441463,
             0.23810227412143423,
             0.15958285448266604,
             -0.03960705975653923]
            """
            return [c[i][j] for i in (range(c.shape[0])) for j in (range(c.shape[1])) if i < j]

        if self.linkage is not None:
            warnings.warn('Linkage is already present, using the already defined linkage. If you want to reset the '
                          'linkage, manually set HCA.linkage = None', UserWarning)
            # leave the function
            return None

        # First calculate a distance metric between the points
        if metric in {"pearson", "spearman"}:
            corr = pd.DataFrame(self.data).T.corr(metric).values
            dist = as_dist(1 - corr)
        else:
            dist = distance.pdist(X=self.data, metric=metric)
        # perform hierarchical clustering using the distance metric
        # the returned matrix self.linkage contains n-1 x 4 elements
        # with each row representing
        # cluster1, cluster2, distance_between_1_and_2,
        # number_of_observations_in_the_cluster
        self.linkage = clst.hierarchy.linkage(dist, method=method)

    def find_nclusters(self, start=2, up_to=20, figsize=(15, 5), plot=True):
        """
        Evaluate number of clusters.

        Parameters
        ----------
        start : int, optional
            The minimum number of clusters to plot. The default is 2.
        up_to : int, optional
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
        up_to += 1
        pred = []
        for i in range(start, up_to):
            # return the assigned cluster labels for each data point
            cluster = clst.hierarchy.fcluster(self.linkage, t=i, criterion='maxclust')
            # calculate scores based on assigned cluster labels and
            # the original data points
            pred.append((davies_bouldin_score(self.data, cluster),
                         silhouette_score(self.data, cluster),
                         calinski_harabasz_score(self.data, cluster)))

        self.clustering_evaluation(pred, figsize, start, up_to, plot)

    def make_cluster(self):
        """
        Form flat clusters from the hierarchical clustering of linkage.

        Returns
        -------
        None.

        """
        if self.nclusters is None:
            raise AttributeError('No. of clusters is None. Perform find_nclusters before.')

        # self.cluster is an array of length x
        # with x = number of original data points containing the ID
        # of the corresponding cluster
        self.clusterId = \
            clst.hierarchy.fcluster(self.linkage,  # the hierarchical clustering
                                    t=self.nclusters,  # max number of clusters
                                    criterion="maxclust")  # forms maximumum n=t clusters

    def auto_run(self, start_processing=1, stop_processing=5):
        """
        Automatically run the clustering pipeline with standard settings.

        Parameters
        ----------
        start_processing : int, optional
            Step of the pipeline to start. The default is 1.
        stop_processing : int, optional
            Step of the pipeline to stop. The default is 5.

        Notes
        -----
        The pipeline currently consists of (1) makeLinkage, (2) findNClusters
        and (3) makeCluster.

        Returns
        -------
        None.

        """
        if start_processing <= 1:
            self.make_linkage()
        if start_processing <= 2 <= stop_processing:
            self.find_nclusters()
        if start_processing <= 3 <= stop_processing:
            self.make_cluster()


class KMeans(_Cluster):
    # noinspection PyUnresolvedReferences
    """
    Perform KMeans clustering on a dataset.

    Returns
    -------
    None.

    Notes
    -----
    The functions uses scipy.cluster.vq.kmeans2
    (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.kmeans2.html#scipy.cluster.vq.kmeans2)

    References
    ----------
    D. Arthur and S. Vassilvitskii, “k-means++: the advantages of careful seeding”, Proceedings of the Eighteenth
    Annual ACM-SIAM Symposium on Discrete Algorithms, 2007.

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
    
    >>> from autoprot import analysis as ana
    >>> c = ana.KMeans(df)
    Removed 0 NaN values from the dataframe to prepare for clustering.
    >>> c.auto_run()
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
        c.auto_run()
    
    Finally visualise the clustering using the visCluster method and include the
    previously extracted labeling column from the original dataframe.
    
    >>> labels.replace(['setosa', 'virginica', 'versicolor'], ["teal", "purple", "salmon"], inplace=True)
    >>> rc = {"species" : labels}
    >>> c.vis_cluster(row_colors={'species': labels})

     .. plot::
         :context: close-figs

         labels.replace(['setosa', 'virginica', 'versicolor'], ["teal", "purple", "salmon"], inplace=True)    
         rc = {"species" : labels}
         c.vis_cluster(row_colors={'species': labels})
         
    As you can see can KMeans quite well separate setosa but virginica and versicolor are harder.
    When we manually pick the number of clusters, it gets a bit better
    
    >>> c.nclusters = 3
    >>> c.make_cluster()
    >>> c.vis_cluster(row_colors={'species': labels}, make_traces=True, file=None, make_heatmap=True)
    
     .. plot::
         :context: close-figs

            c.nclusters = 3  
            c.make_cluster()
            c.vis_cluster(row_colors={'species': labels}, make_traces=True, file=None, make_heatmap=True)
    """

    def find_nclusters(self, start=2, up_to=20, figsize=(15, 5), plot=True, algo='scipy'):
        """
        Evaluate number of clusters.

        Parameters
        ----------
        start : int, optional
            The minimum number of clusters to plot. The default is 2.
        up_to : int, optional
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
        up_to += 1
        pred = []
        for i in range(start, up_to):

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

        self.clustering_evaluation(pred, figsize, start, up_to, plot)

    def make_cluster(self, algo='scipy', **kwargs):
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

    def auto_run(self, start_processing=1, stop_processing=5):
        """
        Automatically run the clustering pipeline with standard settings.

        Parameters
        ----------
        start_processing : int, optional
            Step of the pipeline to start. The default is 1.
        stop_processing : int, optional
            Step of the pipeline to stop. The default is 5.

        Notes
        -----
        The pipeline currently consists of (1) findNClusters
        and (2) makeCluster.

        Returns
        -------
        None.

        """
        if start_processing <= 1:
            self.find_nclusters()
        if start_processing <= 2 <= stop_processing:
            self.make_cluster()
