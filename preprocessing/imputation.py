# -*- coding: utf-8 -*-
"""
Autoprot Preprocessing Functions.

@author: Wignand, Julian, Johannes

@documentation: Julian
"""

import numpy as np
import pandas as pd
import os
from subprocess import run, PIPE, STDOUT
from typing import Union
from ..decorators import report
from .. import r_helper
from .. import preprocessing as pp

RFUNCTIONS, R = r_helper.return_r_path()


# =============================================================================
# Note: When using R functions provided column names might get changed
# Especially, do not use +,- or spaces in your column names. Maybe write decorator to
# validate proper column formatting and handle exceptions
# =============================================================================


# =============================================================================
# IMPUTATION ALGORITHMS
# =============================================================================
@report
def imp_min_prob(df: pd.DataFrame, cols_to_impute: Union[list[str], pd.Index], max_missing: int = None,
                 downshift: Union[int, float] = 1.8, width: Union[int, float] = .3):
    r"""
    Perform an imputation by modeling a distribution on the far left site of the actual distribution.

    The final distribution will be mean shifted and has a smaller variation.
    Intensities should be log transformed before being supplied to this function.

    Downsshift: mean - downshift*sigma
    Var: width*sigma

    Parameters
    ----------
    df : pd.dataframe
        Dataframe on which imputation is performed.
    cols_to_impute : list
        Columns to impute. Should correspond to a single condition (i.e. control).
    max_missing : int, optional
        How many missing values have to be missing across all columns to perfom imputation
        If None all values have to be missing. The default is None.
    downshift : float, optional
        How many Stds to lower values the mean of the new population is shifted. The default is 1.8.
    width : float, optional
        How to scale the Std of the new distribution with respect to the original. The default is .3.

    Returns
    -------
    pd.dataframe
        The dataframe with imputed values.

    Examples
    --------
    >>> forImp = np.log10(phos.filter(regex="Int.*R1").replace(0, np.nan))
    >>> impProt = pp.imp_min_prob(forImp, phos.filter(regex="Int.*R1").columns,
    ...                         width=.4, downshift=2.5)
    >>> impProt.filter(regex="Int.*R1")[impProt["Imputed"]==False].mean(1).hist(density=True, bins=50,
    ...                                                                         label="not Imputed")
    >>> impProt.filter(regex="Int.*R1")[impProt["Imputed"]==True].mean(1).hist(density=True, bins=50,
    ...                                                                        label="Imputed")
    >>> plt.legend()

    .. plot::
        :context: close-figs

        import autoprot.preprocessing as pp
        import autoprot.visualization as vis
        import pandas as pd
        phos = pd.read_csv("_static/testdata/Phospho (STY)Sites_mod.zip", sep="\t", low_memory=False)
        forImp = np.log10(phos.filter(regex="Int.*R1").replace(0, np.nan))
        impProt = pp.imp_min_prob(forImp, phos.filter(regex="Int.*R1").columns, width=.4, downshift=2.5)
        fig, ax1 = plt.subplots(1)
        impProt.filter(regex="Int.*R1")[impProt["Imputed"]==False].mean(1).hist(density=True, bins=50, label="not Imputed", ax=ax1)
        impProt.filter(regex="Int.*R1")[impProt["Imputed"]==True].mean(1).hist(density=True, bins=50, label="Imputed", ax=ax1)
        plt.legend()
        plt.show()
    """
    df = df.copy(deep=True)
    
    # test if cols_to_impute is list of columns
    if type(cols_to_impute) != list():
        cols_to_impute=[cols_to_impute]
        
    # idxs of rows imputation will be excluded
    if max_missing is not None:
        s_nan = df[cols_to_impute].isnull().sum(axis=1)
        s_nan = s_nan[s_nan <= max_missing]
        filter_idx = s_nan.index
    else:
        filter_idx = pd.Index([])

    for col in cols_to_impute:
        count_na = df[col].isna().sum()
        na_index = df[df[col].isna()].index
        if max_missing is not None:
            na_index = na_index.difference(filter_idx)
            count_na = len(na_index)

        #define values before imputation
        mean = df[col].mean()
        var  = df[col].std()
        #new mean, val for imputation
        minimp_mean = mean - downshift*var
        minimp_var = var*width

        rnd = np.random.normal(minimp_mean, minimp_var, size=count_na)
        imputed_s = pd.Series(data=rnd, index=na_index)
        
        col_new = col + "_min_imputed"
        df[col_new] = df[col].fillna(imputed_s)
    
    return df



def imp_seq(df, cols: Union[list[str], pd.Index], print_r=True):
    """
    Perform sequential imputation in R using impSeq from rrcovNA.

    See https://rdrr.io/cran/rrcovNA/man/impseq.html for a description of the
    algorithm.
    SEQimpute starts from a complete subset of the data set Xc and estimates sequentially the missing values in an incomplete observation, say x*, by minimizing the determinant of the covariance of the augmented data matrix X* = [Xc; x']. Then the observation x* is added to the complete data matrix and the algorithm continues with the next observation with missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    cols : list of str
        Colnames to perform imputation of.
    print_r : bool, optional
        Whether to print the output of R, default is True.

    Returns
    -------
    pd.DataFrame
        Dataframe with imputed values.
        Cols with imputed values are named _imputed.
        Contains a col UID that was used for processing.

    """
    d = os.getcwd()
    dataLoc = d + "/input.csv"
    outputLoc = d + "/output.csv"

    if "UID" not in df.columns:
        # UID is basically a row index starting at 1
        df["UID"] = range(1, df.shape[0] + 1)

    if not isinstance(cols, list):
        cols = cols.to_list()
    pp.to_csv(df[["UID"] + cols], dataLoc)

    command = [R, '--vanilla',
               RFUNCTIONS,  # script location
               "impSeq",  # functionName
               dataLoc,  # data location
               outputLoc  # output file
               ]

    p = run(command,
            stdout=PIPE,
            stderr=STDOUT,
            universal_newlines=True)

    if print_r:
        print(p.stdout)

    res = pp.read_csv(outputLoc)
    # append a string to recognise the cols
    res_cols = [f"{i}_imputed" if i != "UID" else i for i in res.columns]
    # change back the R colnames
    res_cols = [x.replace('.', ' ') for x in res_cols]
    res.columns = res_cols

    # merge and retain the rows of the original df
    df = df.merge(res, how='left', on="UID")
    # drop UID again
    df.drop("UID", axis=1, inplace=True)

    # os.remove(dataLoc)
    # os.remove(outputLoc)

    return df


def dima(df, cols: Union[list[str], pd.Index], selection_substr=None, ttest_substr='cluster', methods='fast',
         npat=20, performance_metric='RMSE', print_r=True, min_values_for_imputation=0):
    # noinspection PyUnresolvedReferences
    """
    Perform Data-Driven Selection of an Imputation Algorithm.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    cols : list of str or pd.Index
        Colnames to perform imputation on.
        NOTE: if used on intensities, use log-transformed values.
    selection_substr : str
        pattern to extract columns for processing during DIMA run.
    ttest_substr : 2-element list or str
        For statistical interpretation based on the t-test, the RMSEt ≔ RMSE(tR, tI) serves as rank criterion,
        where t is the t-test statistics calculated from the observed data R and the imputed data O.Todefine the null
        hypothesis H0, the group assignments of the samples have to be specified by the user.

        If string, two elements need to be separated by ','
        If list, concatenation will be done automatically.
        The two elements must be substrings of the columns to compare.
        Make sure that for each substring at least two matching colnames
        are present in the data.
    methods : str or list of str, optional
        Methods to evaluate. Default 'fast' for the 9 most used imputation
        methods. Possible values are 'impSeqRob','impSeq','missForest',
        'imputePCA','ppca','bpca', ...
    npat : int, optional
        Number of missing value patterns to evaluate
    performance_metric : str, optional
        Metric used to select the best algorithm. Possible values are
        Dev, RMSE, RSR, pF,  Acc, PCC, RMSEt.
    min_values_for_imputation : int, optional
        Minimum number of non-missing values for imputation.
        Default is 0, which means that all values will be imputed.
    print_r : bool
        Whether to print the R output to the Python console.

    Returns
    -------
    pd.DataFrame
        Input dataframe with imputed values.
    pd.DataFrame
        Overview of performance metrices of the different algorithms.

    Examples
    --------
    We will use a standard sample dataframe and generate some missing values to
    demonstrate the imputation.
    
    >>> from autoprot import preprocessing as pp
    >>> import seaborn as sns
    >>> import pandas as pd
    >>> import numpy as np
    >>> iris = sns.load_dataset('iris')
    >>> _ = iris.pop('species')
    >>> for col in iris.columns:
    ...     iris.loc[iris.sample(frac=0.1).index, col] = np.nan

    >>> imp, perf = pp.dima(
    ...     iris, iris.columns, performance_metric="RMSEt", ttest_substr=["petal", "sepal"]
    ... )
    
    >>> imp.head()
       sepal_length  sepal_width  petal_length  ...  sepal_width_imputed  petal_length_imputed  petal_width_imputed
    0           5.1          3.5           1.4  ...                  3.5                   1.4                  0.2
    1           4.9          3.0           1.4  ...                  3.0                   1.4                  0.2
    2           4.7          3.2           1.3  ...                  3.2                   1.3                  0.2
    3           4.6          3.1           1.5  ...                  3.1                   1.5                  0.2
    4           5.0          3.6           1.4  ...                  3.6                   1.4                  0.2
    
    [5 rows x 9 columns]
    
    >>> perf.head()
                Deviation      RMSE       RSR  p-Value_F-test   Accuracy       PCC  RMSEttest
    impSeqRob    0.404402  0.531824  0.265112        0.924158  94.735915  0.997449   0.222656
    impSeq       0.348815  0.515518  0.256984        0.943464  95.413732  0.997563   0.223783
    missForest   0.348815  0.515518  0.256984        0.943464  95.413732  0.997563   0.223783
    imputePCA    0.404402  0.531824  0.265112        0.924158  94.735915  0.997449   0.222656
    ppca         0.377638  0.500354  0.249424        0.933919  95.000000  0.997721   0.199830


    It is also possible to specify the minimum number of non-missing values that are required for imputation.

    >>> for col in iris.columns:
    ...     iris.loc[iris.sample(frac=0.4).index, col] = np.nan
    >>> imp, perf = pp.dima(
    ...     iris, iris.columns, performance_metric="RMSEt", min_values_for_imputation=2
    ... )

    References
    ----------
    Egert, J., Brombacher, E., Warscheid, B. & Kreutz, C. DIMA: Data-Driven Selection of an Imputation Algorithm.
        Journal of Proteome Research 20, 3489–3496 (2021-06).
    """
    if not df.isnull().values.any():
        raise ValueError('Your dataframe does not contain missing values. Will return as is.')
    df = df.copy(deep=True)

    d = os.getcwd()
    data_loc = d + "/input.csv"
    output_loc = d + "/output.csv"

    for col in cols:
        mvs = df[col].isna().sum() / df[col].size
        print(f"{mvs * 100:.2f}% MVs in column {col}")

    if selection_substr is not None:
        df = df.filter(regex=selection_substr)

    if "UID" not in df.columns:
        # UID is basically a row index starting at 1
        df["UID"] = range(1, df.shape[0] + 1)

    if isinstance(cols, pd.Index):
        cols = cols.to_list()
    pp.to_csv(df[["UID"] + cols], data_loc)

    if isinstance(ttest_substr, list):
        ttest_substr = ','.join(ttest_substr)

    if isinstance(methods, list):
        methods = ','.join(methods)

    command = [R, '--vanilla',
               RFUNCTIONS,  # script location
               "dima",  # functionName
               data_loc,  # data location
               output_loc,  # output file
               ttest_substr,  # substring for ttesting
               methods,  # method(s) aka algorithms to benchmark
               str(npat),  # number of patterns
               performance_metric,  # to select the best algorithm
               str(min_values_for_imputation)  # minimum number of non-missing values for imputation
               ]

    p = run(command,
            stdout=PIPE,
            stderr=STDOUT,
            universal_newlines=True)

    if print_r:
        print(p.stdout)

    res = pp.read_csv(output_loc)
    # keep only the columns added by DIMA and the UID for merging
    res = res.loc[:, (res.columns.str.contains('Imputation')) | (res.columns.str.contains('UID'))]
    # append a string to recognise the cols
    res_cols = [f"{i}_imputed" if i != "UID" else i for i in res.columns]
    # remove the preceding string Imputation
    res_cols = [x.replace('Imputation.', '') for x in res_cols]
    res.columns = res_cols

    # merge and retain the rows of the original df
    df = df.merge(res, how='left', on="UID")
    # drop UID again
    df.drop("UID", axis=1, inplace=True)

    perf = pp.read_csv(output_loc[:-4] + '_performance.csv')

    os.remove(data_loc)
    os.remove(output_loc)
    os.remove(output_loc[:-4] + '_performance.csv')

    return df, perf
