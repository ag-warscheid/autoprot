# -*- coding: utf-8 -*-
"""
Autoprot Preprocessing Functions.

@author: Wignand, Julian, Johannes

@documentation: Julian
"""

import numpy as np
import pandas as pd
from subprocess import run, PIPE, STDOUT
from typing import Union
from autoprot import r_helper
from autoprot import preprocessing as pp

RFUNCTIONS, R = r_helper.return_r_path()

# defines which functions are exposed at the module level
__all__ = ['imp_min_prob', 'imp_median', 'imp_seq', 'dima']
# =============================================================================
# Note: When using R functions provided column names might get changed
# Especially, do not use +,- or spaces in your column names. Maybe write decorator to
# validate proper column formatting and handle exceptions
# =============================================================================


# =============================================================================
# IMPUTATION ALGORITHMS
# =============================================================================
def imp_min_prob(
    df: pd.DataFrame,
    cols: Union[list[str], str],
    min_missing: int = None,
    downshift: Union[int, float] = 1.8,
    width: Union[int, float] = 0.3,
    return_cols: bool = False,
    gen_isimp_cols: bool = False,
    return_isimp_cols: bool = False,
):
    r"""
    Perform an imputation by modeling a distribution on the far left site of the actual distribution.

    The final distribution will be mean shifted and has a smaller variation.
    Intensities should be log-transformed before being supplied to this function.

    Downsshift: mean - downshift*sigma
    Var: width*sigma

    Parameters
    ----------
    df : pd.dataframe
        Dataframe on which imputation is performed.
    cols : list or str
        Columns to impute. Should correspond to a single condition (i.e. control).
    min_missing : int, optional
        How many missing values have to be missing across all columns to perfom imputation
        If None imputation will be performed on all cells. The default is None.
    downshift : float, optional
        How many Stds to lower values the mean of the new population is shifted. The default is 1.8.
    width : float, optional
        How to scale the Std of the new distribution with respect to the original. The default is .3.
    return_cols : bool, optional
        Whether to return the columns that were imputed. The default is False.
    gen_isimp_cols : bool, optional
        Whether to generate columns indicating which values were imputed. The default is False.
    return_isimp_cols : bool, optional
        Whether to return the columns indicating which values were imputed. The default is False.

    Returns
    -------
    pd.dataframe
        The dataframe with imputed values.
    list of str
        Columns that were imputed.

    Examples
    --------
    .. plot::
        :context: close-figs

        phos = pd.read_csv("../data/Phospho (STY)Sites_minimal.zip", sep="\t", low_memory=False)
        forImp = np.log10(phos.filter(regex="Int.*R1").replace(0, np.nan))
        impProt = pp.imp_min_prob(forImp, phos.filter(regex="Int.*R1").columns, width=.4, downshift=2.5)
        fig, ax1 = plt.subplots(1)
        imputed_values = impProt.filter(regex="Int.*R1$").isnull()
        ax1.hist(impProt.filter(regex="Int.*R1_min_imputed").values[~imputed_values],
                  density=True, bins=50, label="not Imputed", alpha=.5)
        ax1.hist(impProt.filter(regex="Int.*R1_min_imputed").values[imputed_values],
                  density=True, bins=50, label="Imputed", alpha=.5)
        ax1.set_xlabel("log10 Intensity")
        ax1.set_ylabel("Density")

        plt.legend()
        plt.show()
    """
    if return_isimp_cols and not gen_isimp_cols:
        raise ValueError(
            "You set return_isimp_cols to True but gen_isimp_cols is False. Cannot return columns that were not "
            "generated."
        )

    # test if cols_to_impute is iterable
    try:
        iter(cols)
    except TypeError:
        cols: list[str] = [cols]

    # idxs of rows in which imputation will be excluded
    if min_missing is not None:
        s_nan = df[cols].isnull().sum(axis=1)  # number of NaNs per row
        filter_idx = s_nan[
            s_nan < min_missing
        ].index  # index of rows with less than min_missing NaNs (i.e. to exclude)
        print(
            f"Excluding {len(filter_idx)} rows from imputation because they have <{min_missing} missing values."
        )
    else:
        filter_idx = pd.Index([])
        print("No rows are excluded from imputation.")

    imputed_cols = []
    isimp_cols = []
    for col in cols:
        count_na = (
            df[col].isna().sum()
        )  # per column count of NaNs (requried for random number generation)
        na_index = df[
            df[col].isna()
        ].index  # index of rows to impute in the current column
        if min_missing is not None:
            na_index = na_index.difference(
                filter_idx
            )  # remove idxs to exclude from imputation
            count_na = len(na_index)

        # define values before imputation
        mean = df[col].mean()
        var = df[col].std()
        # new mean, val for imputation
        minimp_mean = mean - downshift * var
        minimp_var = var * width

        rnd = np.random.normal(minimp_mean, minimp_var, size=count_na)
        imputed_s = pd.Series(
            data=rnd, index=na_index
        )  # new series with imputed values and index of NaNs

        col_new = col + "_min_imputed"
        df[col_new] = df[col].fillna(
            imputed_s
        )  # fillna will map the values based on the index
        imputed_cols.append(col_new)
        if gen_isimp_cols:
            isimp_col = col + "_is_imputed"
            isimp_cols.append(isimp_col)
            df[isimp_col] = False
            df.loc[na_index, isimp_col] = True  # set only the imputed rows to True

    if return_isimp_cols and return_cols:
        return df, imputed_cols, isimp_cols
    elif return_isimp_cols:
        return df, isimp_cols
    elif return_cols:
        return df, imputed_cols
    else:
        return df


def imp_median(
    df: pd.DataFrame,
    cols_to_impute: Union[list[str], pd.Index],
    min_missing: int = None,
    max_missing: int = None,
    return_cols: bool = False,
    gen_isimp_cols: bool = False,
    return_isimp_cols: bool = False,
) -> (
    Union[pd.DataFrame, tuple[pd.DataFrame, list[str], list[str]]]
    | Union[pd.DataFrame, tuple[pd.DataFrame, list[str]]]
    | pd.DataFrame
):
    """
    Perform an imputation by replacing missing values with the median of the row.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe on which imputation is performed.
    cols_to_impute : list of str or pd.Index
        Columns to impute. Should correspond to a single condition (i.e. control).
    min_missing : int, optional
        How many missing values have to be missing across all columns to perform imputation.
        If None one value has to be missing. The default is None.
    max_missing : int, optional
        How many missing values are allowed across all columns to perform imputation.
        If None the number of columns minus one is used (i.e. one value has to be present). The default is None.
    return_cols : bool, optional
        Whether to return the columns that were imputed. The default is False.
    gen_isimp_cols : bool, optional
        Whether to generate columns indicating which values were imputed. The default is False.
    return_isimp_cols : bool, optional
        Whether to return the columns indicating which values were imputed. The default is False.

    Returns
    -------
    pd.DataFrame
        The dataframe with imputed values.
    list of str
        Columns that were imputed.
    list of str
        Columns indicating which values were imputed.

    """
    if return_isimp_cols and not gen_isimp_cols:
        raise ValueError(
            "You set return_isimp_cols to True but gen_isimp_cols is False. Cannot return columns that were not "
            "generated."
        )

    # test if cols_to_impute is iterable
    try:
        iter(cols_to_impute)
    except TypeError:
        cols_to_impute: list[str] = [cols_to_impute]

    min_missing = min_missing if min_missing is not None else 1
    max_missing = max_missing if max_missing is not None else len(cols_to_impute) - 1

    # idxs of rows for imputation
    filter_idx = df[
        min_missing <= df[cols_to_impute].isnull().sum(axis=1)
    ].index.intersection(
        df[df[cols_to_impute].isnull().sum(axis=1) <= max_missing].index
    )

    imputed_rows = []
    imputed_cols = [x + "_median_imputed" for x in cols_to_impute]
    isimp_cols = [x + "_is_imputed" for x in cols_to_impute]
    isimp_rows = []
    print(f"Imputing {len(filter_idx)} rows out of {len(df)}")
    for row in df.loc[filter_idx, cols_to_impute].itertuples(index=False):
        row_median = np.nanmedian(row)
        # fill the NaN values with the median of the row
        row = pd.Series(np.nan_to_num(row, nan=row_median), index=imputed_cols)
        imputed_rows.append(row)
        if gen_isimp_cols:
            isimp_row = pd.Series(np.isnan(row), index=imputed_cols)
            isimp_rows.append(isimp_row)

    # create a new DataFrame with the imputed rows
    imputed_df = pd.DataFrame(imputed_rows, columns=imputed_cols, index=filter_idx)
    # join the imputed DataFrame with the original DataFrame
    df = df.join(imputed_df)
    # create a new DataFrame with the is_imputed rows
    if gen_isimp_cols:
        isimp_df = pd.DataFrame(isimp_rows, columns=isimp_cols, index=filter_idx)
        # join the is_imputed DataFrame with the original DataFrame
        df = df.join(isimp_df)

    # set the rows which were not imputed to the original values
    # the loop above only created the imputed rows
    untouched_rows = ~df.index.isin(filter_idx)
    # Create a temporary DataFrame with imputed values
    tmp = df.loc[untouched_rows, cols_to_impute].copy()
    tmp.columns = imputed_cols  # Rename if necessary
    # Update only the selected columns in the original df
    df.update(tmp)
    if gen_isimp_cols:
        # fill the unset values in the is_imputed cols with False
        for col in isimp_cols:
            df[col].fillna(False, inplace=True)

    if return_isimp_cols and return_cols:
        return df, imputed_cols, isimp_cols
    elif return_isimp_cols:
        return df, isimp_cols
    elif gen_isimp_cols:
        return df, imputed_cols
    else:
        return df


def imp_seq(
    df,
    cols: Union[list[str], pd.Index],
    print_r=False,
    return_cols=False,
    suffix="_imputed",
):
    """
    Perform sequential imputation in R using impSeq from rrcovNA.

    See https://rdrr.io/cran/rrcovNA/man/impseq.html for a description of the algorithm. SEQimpute starts from a
    complete subset of the data set Xc and estimates sequentially the missing values in an incomplete observation,
    say x*, by minimizing the determinant of the covariance of the augmented data matrix X* = [Xc; x']. Then the
    observation x* is added to the complete data matrix and the algorithm continues with the next observation with
    missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    cols : list of str
        Colnames to perform imputation of.
    print_r : bool, optional
        Whether to print the output of R, default is False.
    return_cols : bool, optional
        Whether to return the columns that were imputed. The default is False.
    suffix : str, optional
        Suffix to add to the imputed columns. Default is '_imputed'.

    Returns
    -------
    pd.DataFrame
        Dataframe with imputed values.
        Cols with imputed values are named _imputed.
        Contains a col UID that was used for processing.
    list of str
        Columns that were imputed.

    """
    data_loc, output_loc = r_helper.generate_paths_for_r(df, cols, tool="_imp_seq")

    command = [
        R,
        "--vanilla",
        RFUNCTIONS,  # script location
        "impSeq",  # functionName
        data_loc,  # data location
        output_loc,  # output file
    ]

    p = run(command, stdout=PIPE, stderr=STDOUT, universal_newlines=True)

    if print_r:
        print(p.stdout)

    res = pp.read_csv(output_loc)

    return r_helper.merge_data_from_r(
        res,
        df,
        suffix=suffix,
        locs_to_remove=[data_loc, output_loc],
        return_cols=return_cols,
    )


def dima(
    df,
    cols: Union[list[str], pd.Index],
    selection_substr=None,
    ttest_substr="cluster",
    methods="fast",
    npat=20,
    performance_metric="RMSE",
    print_r=True,
    min_values_for_imputation=0,
    return_cols=False,
    suffix="_imputed",
):
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

        If is string, two elements need to be separated by ','
        If is list, concatenation will be done automatically.
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
    return_cols : bool, optional
        Whether to return the columns that were imputed. The default is False.
    suffix : str, optional
        Suffix to add to the imputed columns. Default is '_imputed'.

    Returns
    -------
    pd.DataFrame
        Input dataframe with imputed values.
    pd.DataFrame
        Overview of performance metrices of the different algorithms.
    list of str
        Columns that were imputed.

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
        raise ValueError(
            "Your dataframe does not contain missing values. Will return as is."
        )
    df = df.copy(deep=True)

    data_loc, output_loc = r_helper.generate_paths_for_r(df, cols, tool="_dima")

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
        ttest_substr = ",".join(ttest_substr)

    if isinstance(methods, list):
        methods = ",".join(methods)

    command = [
        R,
        "--vanilla",
        RFUNCTIONS,  # script location
        "dima",  # functionName
        data_loc,  # data location
        output_loc,  # output file
        ttest_substr,  # substring for ttesting
        methods,  # method(s) aka algorithms to benchmark
        str(npat),  # number of patterns
        performance_metric,  # to select the best algorithm
        str(
            min_values_for_imputation
        ),  # minimum number of non-missing values for imputation
    ]

    p = run(command, stdout=PIPE, stderr=STDOUT, universal_newlines=True)

    if print_r:
        print(p.stdout)

    res = pp.read_csv(output_loc)
    # keep only the columns added by DIMA and the UID for merging
    res = res.loc[
        :, (res.columns.str.contains("Imputation")) | (res.columns.str.contains("UID"))
    ]
    res.columns = [x.replace("Imputation.", "") for x in res.columns]

    perf = pp.read_csv(output_loc[:-4] + "_performance.csv")

    imputed_and_cols = r_helper.merge_data_from_r(
        res,
        df,
        suffix=suffix,
        locs_to_remove=[data_loc, output_loc, output_loc[:-4] + "_performance.csv"],
        return_cols=return_cols,
    )

    # return the imputed df and the performance metrics
    return (
        (imputed_and_cols[0], perf, imputed_and_cols[1])
        if return_cols
        else (imputed_and_cols, perf)
    )
