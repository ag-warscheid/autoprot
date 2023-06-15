# -*- coding: utf-8 -*-
"""
Autoprot Analysis Functions.

@author: Wignand, Julian, Johannes

@documentation: Julian
"""
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from .. import r_helper

from gprofiler import GProfiler
gp = GProfiler(
    user_agent="autoprot",
    return_dataframe=True)
RFUNCTIONS, R = r_helper.return_r_path()

# check where this is actually used and make it local
cmap = sns.diverging_palette(150, 275, s=80, l=55, n=9)


def edm(matrix_a, matrix_b):
    """
    Calculate an euclidean distance matrix between two matrices.

    See:  https://medium.com/swlh/euclidean-distance-matrix-4c3e1378d87f

    Parameters
    ----------
    matrix_a : np.ndarray
        Matrix 1.
    matrix_b : np.ndarray
        Matrix 2.

    Returns
    -------
    np.ndarray
        Distance matrix.

    """
    p1 = np.sum(matrix_a ** 2, axis=1)[:, np.newaxis]
    p2 = np.sum(matrix_b ** 2, axis=1)
    p3 = -2 * np.dot(matrix_a, matrix_b.T)
    return np.sqrt(p1 + p2 + p3)


def loess(data, xvals, yvals, alpha, poly_degree=2):
    # noinspection PyUnresolvedReferences
    r"""
    Calculate a LOcally-Weighted Scatterplot Smoothing Fit.

    See: https://medium.com/@langen.mu/creating-powerfull-lowess-graphs-in-python-e0ea7a30b17a

    Parameters
    ----------
    data : pd.Dataframe
        Input dataframe.
    xvals : str
        Colname of x values.
    yvals : str
        Colname of y values.
    alpha : float
        Sensitivity of the estimation.
        Controls how much of the total number of values is used during weighing.
        0 <= alpha <= 1.
    poly_degree : int, optional
        Degree of the fitted polynomial. The default is 2.

    Returns
    -------
    None.

    Notes
    -----
    Loess normalisation (also referred to as Savitzky-Golay filter) locally approximates
    the data around every point using low-order functions and giving less weight to distant
    data points.

    Examples
    --------
    >>> np.random.seed(10)
    >>> x_values = np.random.randint(-50,110,size=250)
    >>> y_values = np.square(x_values)/1.5 + np.random.randint(-1000,1000, size=len(x_values))
    >>> df = pd.DataFrame({"Xvalue" : x_values,
                           "Yvalue" : y_values
                           })

    >>> evalDF = autoprot.analysis.loess(df, "Xvalue", "Yvalue", alpha=0.7, poly_degree=2)
    >>> fig, ax = plt.subplots(1,1)
    >>> sns.scatterplot(df["Xvalue"], df["Yvalue"], ax=ax)
    >>> ax.plot(eval_df['v'], eval_df['g'], color='red', linewidth= 3, label="Test")

    .. plot::
        :context: close-figs

        import autoprot.analysis as ana
        import seaborn as sns

        x_values = np.random.randint(-50,110,size=(250))
        y_values = np.square(x_values)/1.5 + np.random.randint(-1000,1000, size=len(x_values))
        df = pd.DataFrame({"Xvalue" : x_values,
                           "Yvalue" : y_values
                           })
        evalDF = ana.loess(df, "Xvalue", "Yvalue", alpha=0.7, poly_degree=2)
        fig, ax = plt.subplots(1,1)
        sns.scatterplot(df["Xvalue"], df["Yvalue"], ax=ax)
        ax.plot(evalDF['v'], evalDF['g'], color='red', linewidth= 3, label="Test")
        plt.show()
    """
    # generate x,y value pairs and sort them according to x
    all_data = sorted(zip(data[xvals].tolist(), data[yvals].tolist()), key=lambda x: x[0])
    # separate the values again into x and y cols
    xvals, yvals = zip(*all_data)
    # generate empty df for final fit
    eval_df = pd.DataFrame(columns=['v', 'g'])

    n = len(xvals)
    m = n + 1
    # how many data points to include in the weighing
    # alpha determines the relative proportion of values considered during weighing
    q = int(np.floor(n * alpha) if alpha <= 1.0 else n)
    # the average point to point distance in x direction
    avg_interval = ((max(xvals) - min(xvals)) / len(xvals))
    # calculate upper on lower boundaries
    v_lb = min(xvals) - (.5 * avg_interval)
    v_ub = (max(xvals) + (.5 * avg_interval))
    # coordinates for the fitting points
    v = enumerate(np.linspace(start=v_lb, stop=v_ub, num=m), start=1)
    # create an array of ones of the same length as xvals
    xcols = [np.ones_like(xvals)]

    for j in range(1, (poly_degree + 1)):
        xcols.append([i ** j for i in xvals])
    x_mtx = np.vstack(xcols).T
    for i in v:
        iterval = i[1]
        iterdists = sorted([(j, np.abs(j - iterval)) for j in xvals], key=lambda x: x[1])
        _, raw_dists = zip(*iterdists)
        scale_fact = raw_dists[q - 1]
        scaled_dists = [(j[0], (j[1] / scale_fact)) for j in iterdists]
        weights = [(j[0], ((1 - np.abs(j[1] ** 3)) ** 3 if j[1] <= 1 else 0)) for j in scaled_dists]
        _, weights = zip(*sorted(weights, key=lambda x: x[0]))
        _, raw_dists = zip(*sorted(iterdists, key=lambda x: x[0]))
        _, scaled_dists = zip(*sorted(scaled_dists, key=lambda x: x[0]))
        w = np.diag(weights)
        b = np.linalg.inv(x_mtx.T @ w @ x_mtx) @ (x_mtx.T @ w @ yvals)
        # loc_eval
        local_est = sum(i[1] * (iterval ** i[0]) for i in enumerate(b))
        iter_df2 = pd.DataFrame({
            'v': [iterval],
            'g': [local_est]
        })
        eval_df = pd.concat([eval_df, iter_df2])
    eval_df = eval_df[['v', 'g']]
    return eval_df


def make_psm(seq, seq_len):
    # noinspection PyUnresolvedReferences
    """
    Generate a position score matrix for a set of sequences.

    Returns the percentage of each amino acid for each position that
    can be further normalized using a PSM of unrelated/background sequences.

    Parameters
    ----------
    seq : list of str
        list of sequences.
    seq_len : int
        Length of the peptide sequences.
        Must match to the list provided.

    Returns
    -------
    pd.Dataframe
        Dataframe holding the prevalence for every amino acid per position in
        the input sequences.

    Examples
    --------
    >>> autoprot.analysis.make_psm(['PEPTIDE', 'PEGTIDE', 'GGGGGGG'], 7)
              0         1         2         3         4         5         6
    G  0.333333  0.333333  0.666667  0.333333  0.333333  0.333333  0.333333
    P  0.666667  0.000000  0.333333  0.000000  0.000000  0.000000  0.000000
    matrix_a  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
    V  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
    L  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
    I  0.000000  0.000000  0.000000  0.000000  0.666667  0.000000  0.000000
    M  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
    C  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
    F  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
    Y  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
    W  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
    H  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
    K  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
    R  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
    Q  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
    N  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
    E  0.000000  0.666667  0.000000  0.000000  0.000000  0.000000  0.666667
    D  0.000000  0.000000  0.000000  0.000000  0.000000  0.666667  0.000000
    S  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
    T  0.000000  0.000000  0.000000  0.666667  0.000000  0.000000  0.000000

    """
    aa_dic = {
        'G': 0,
        'P': 0,
        'matrix_a': 0,
        'V': 0,
        'L': 0,
        'I': 0,
        'M': 0,
        'C': 0,
        'F': 0,
        'Y': 0,
        'W': 0,
        'H': 0,
        'K': 0,
        'R': 0,
        'Q': 0,
        'N': 0,
        'E': 0,
        'D': 0,
        'S': 0,
        'T': 0,
    }

    seq = [i for i in seq if len(i) == seq_len]
    seq_t = [''.join(s) for s in zip(*seq)]
    score_matrix = []
    for pos in seq_t:
        d = aa_dic.copy()
        for aa in pos:
            aa = aa.upper()
            if aa not in ['.', '-', '_', "dataframe"]:
                d[aa] += 1
        score_matrix.append(d)

    for pos in score_matrix:
        for k in pos.keys():
            pos[k] /= len(seq)

    # empty array -> (sequenceWindow, aa)
    m = np.empty((seq_len, 20))
    for i in range(m.shape[0]):
        x = list(score_matrix[i].values())
        m[i] = x

    m = pd.DataFrame(m, columns=list(aa_dic.keys()))

    return m.T
