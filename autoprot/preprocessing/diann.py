import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations

import numpy as np
import pandas as pd

# defines which functions are exposed at the module level
__all__ = ["load_parquet", "parquet_to_pg", "silac_protein_group_from_diann"]


def load_parquet(
    path, filters: dict = None, mbr: bool = True, crap_str: str | list[str] = "cRAP"
) -> pd.DataFrame:
    """Load Parquet file with optional filtering.

    Parameters
    ----------
    path: str
        Path to Parquet file.
    filters: dict, optional
        Dictionary of column names and threshold values for filtering.
    mbr: bool, default=True
        Whether to use MBR-specific filters.
    crap_str: str or list of str, default='cRAP'
        String or list of strings to filter out contaminants.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """

    # check if file exists
    try:
        rp = pd.read_parquet(path)
        print(f"Loaded {len(rp)} precursors from report.parquet")

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")

    # Use default filters if none provided
    if filters is None:
        if mbr:
            filters = {
                "Q.Value": 0.01,  # Default filter for quality
                "PG.Q.Value": 0.01,
                "Lib.Q.Value": 0.01,
                "Lib.PG.Q.Value": 0.01,
            }
        else:
            filters = {
                "Q.Value": 0.01,  # Default filter for quality
                "PG.Q.Value": 0.01,
                "Global.Q.Value": 0.01,
                "Global.PG.Q.Value": 0.01,
            }

    if isinstance(crap_str, str):
        crap_str = [crap_str]

    for crp in crap_str:
        initial_count = len(rp)
        rp = rp[~rp["Protein.Names"].str.contains(crp, na=False)]
        filtered_count = len(rp)
        print(
            f"Filtered out {initial_count - filtered_count} entries containing '{crp}'"
        )

    # Apply filters
    all_filters_bool = [
        rp[col] <= val for col, val in filters.items() if col in rp.columns
    ]
    initial_count = len(rp)
    if all_filters_bool:
        combined_filter = np.logical_and.reduce(all_filters_bool)
        rp = rp[combined_filter]
        filtered_count = len(rp)
        print(
            f"Applied filters: {filters}. Filtered out {initial_count - filtered_count} entries. Remaining: {filtered_count}."
        )
    else:
        print("No valid filter columns found in DataFrame.")

    return rp


def parquet_to_pg(
    path,
    filters: dict = None,
    mbr: bool = True,
    crap_str: str | list[str] = "cRAP",
    index_cols: list[str] = None,
    reset_index: bool = True,
) -> pd.DataFrame:
    """
    Load DIANN precursor data from a Parquet file and convert to protein group-level intensities.
    """
    rp = load_parquet(path, filters=filters, mbr=mbr, crap_str=crap_str)

    if index_cols is None:
        index_cols = ["Protein.Group", "Genes"]

    # Select relevant columns and drop duplicates
    pg = rp[["Run", "PG.MaxLFQ"] + index_cols].drop_duplicates().copy()
    pg["PG.MaxLFQ"] = pg["PG.MaxLFQ"].replace(0, pd.NA)

    pg = pg.pivot_table(
        index=index_cols,
        columns="Run",
        values="PG.MaxLFQ",
        aggfunc="first",  # noqa
    )

    # convert numerical columns to float
    pg = pg.astype(float)

    print(f"Aggregation to protein group level done. Final shape: {pg.shape}")

    if reset_index:
        pg = pg.reset_index()
    return pg


def silac_protein_group_from_diann(df):
    """
    Convert DIANN SILAC precursor data to protein group-level ratios.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing DIANN SILAC precursor-level data with necessary columns. Should be pre-filtered for quality.

    Returns
    -------
    pd.DataFrame
        Protein group-level DataFrame with aggregated SILAC ratios and metadata.
    """

    def pivot_and_calc_ratios(df):
        """Pivot once globally, minimal index"""
        required_cols = {
            "Precursor.Id",
            "Protein.Group",
            "Stripped.Sequence",
            "Channel",
            "Precursor.Quantity",
            "Ms1.Normalised",
        }
        missing = required_cols - set(df.columns)
        if missing:
            raise KeyError(f"Missing columns : {missing}")

        print(
            f"Pivoting DataFrame of shape {df.shape} with channels {', '.join(df['Channel'].unique().tolist())}"
        )

        pivot_df = df.pivot_table(
            index=["Run", "Precursor.Id", "Protein.Group", "Stripped.Sequence"],
            columns="Channel",
            values=[
                "Precursor.Quantity",
                "Ms1.Normalised",
                "Global.PG.Q.Value",
                "Channel.Q.Value",
            ],
            aggfunc="first",
        )
        pivot_df.columns = [
            f"{a}_{b}" for a, b in pivot_df.columns
        ]  # flatten multiindex; adds channel suffixes
        pivot_df = pivot_df.reset_index()

        # calculate ratios between all combinations of channels
        for combination in combinations(sorted(df["Channel"].unique()), 2):
            ch1, ch2 = combination
            pivot_df[f"Ratio_{ch1}_{ch2}_Ms1"] = (
                pivot_df[f"Ms1.Normalised_{ch1}"] / pivot_df[f"Ms1.Normalised_{ch2}"]
            )
            pivot_df[f"Ratio_{ch1}_{ch2}_Precursor"] = (
                pivot_df[f"Precursor.Quantity_{ch1}"]
                / pivot_df[f"Precursor.Quantity_{ch2}"]
            )

        pivot_df.replace([0, np.inf, -np.inf], np.nan, inplace=True)
        return pivot_df

    def sum_per_run(run_name, ratio_df, mapper):
        """Per-run protein aggregation"""
        return (
            ratio_df.groupby("Protein.Group", observed=True)
            .agg(mapper)
            .add_suffix(f"_{run_name}")
        )

    def format_final_output(processed, orig):
        """Create MaxQuant-style output with metadata"""
        # Get first instance metadata
        metadata_cols = ["Protein.Ids", "Protein.Names", "Genes"]
        metadata = orig.groupby("Protein.Group")[metadata_cols].first()

        # Combine with processed data
        final_df = metadata.join(processed, how="right")

        # Sort columns: metadata first, then run-based columns
        col_order = metadata_cols + sorted(
            [c for c in final_df.columns if c not in metadata_cols]
        )

        return final_df[col_order]

    # --- Main Execution ---
    # Step 1: Pivot and calculate ratios for each run
    start_time = time.time()
    ratio_df = pivot_and_calc_ratios(df)

    # set Protein.Group as categorical for memory efficiency
    ratio_df["Run"] = ratio_df["Run"].astype("category")
    ratio_df["Protein.Group"] = ratio_df["Protein.Group"].astype("category")
    print(
        f"Pivoted and calculated ratios. Runtime: {time.time() - start_time:.2f} sec."
    )

    # Step 2: Process runs in parallel
    # construct aggregation mapper
    mapper = {}
    for ch1, ch2 in combinations(sorted(df["Channel"].unique()), 2):
        mapper[f"Ratio_{ch1}_{ch2}_Ms1"] = "median"
        mapper[f"Ratio_{ch1}_{ch2}_Precursor"] = "median"
        mapper[f"Ms1.Normalised_{ch1}"] = "sum"
        mapper[f"Ms1.Normalised_{ch2}"] = "sum"
        mapper[f"Precursor.Quantity_{ch1}"] = "sum"
        mapper[f"Precursor.Quantity_{ch2}"] = "sum"
        mapper[f"Global.PG.Q.Value_{ch1}"] = "min"
        mapper[f"Global.PG.Q.Value_{ch2}"] = "min"
        mapper[f"Channel.Q.Value_{ch1}"] = "min"
        mapper[f"Channel.Q.Value_{ch2}"] = "min"
        mapper[f"Stripped.Sequence"] = lambda x: ",".join(x.unique())

    print(f"Using aggregation mapper\n {mapper}")
    start_time = time.time()
    results = []
    run_groups = ratio_df.groupby("Run", sort=False, observed=True)
    print(
        f"Processing {len(run_groups)} runs in parallel using {multiprocessing.cpu_count()} cores."
    )
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures_list = {
            executor.submit(sum_per_run, run_name, group, mapper): run_name
            for run_name, group in run_groups
        }
        for fut in as_completed(futures_list):
            results.append(fut.result())
    print(f"Processed all runs. Runtime: {time.time() - start_time:.2f} sec.")

    # Step 3: Combine results by merging on index
    start_time = time.time()
    combined = pd.concat(results, axis=1)
    print(f"Combined all runs. Runtime: {time.time() - start_time:.2f} sec.")

    # Step 4: Format output
    start_time = time.time()
    final_df = format_final_output(combined, df)
    print(f"Formatted final output. Runtime: {time.time() - start_time:.2f} sec.")
    print(f"Final protein matrix shape: {final_df.shape}")
    return final_df.sort_values(final_df.columns[3], ascending=False)
