import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

import numpy as np
import pandas as pd


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
        pivot_df.columns = [f"{a}_{b}" for a, b in pivot_df.columns]
        pivot_df = pivot_df.reset_index()

        pivot_df["H_L_Ms1"] = (
            pivot_df["Ms1.Normalised_H"] / pivot_df["Ms1.Normalised_L"]
        )
        pivot_df["H_L_Precursor"] = (
            pivot_df["Precursor.Quantity_H"] / pivot_df["Precursor.Quantity_L"]
        )

        pivot_df.replace([0, np.inf, -np.inf], np.nan, inplace=True)
        return pivot_df

    def sum_per_run(run_name, ratio_df):
        """Per-run protein aggregation"""
        return (
            ratio_df.groupby("Protein.Group", observed=True)
            .agg(
                {
                    "H_L_Ms1": "median",
                    "H_L_Precursor": "median",
                    "Ms1.Normalised_H": "sum",
                    "Ms1.Normalised_L": "sum",
                    "Precursor.Quantity_H": "sum",
                    "Precursor.Quantity_L": "sum",
                    "Global.PG.Q.Value_H": "min",
                    "Channel.Q.Value_H": "min",
                    "Stripped.Sequence": lambda x: ",".join(x.unique()),
                }
            )
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
    start_time = time.time()
    results = []
    run_groups = ratio_df.groupby("Run", sort=False, observed=True)
    print(
        f"Processing {len(run_groups)} runs in parallel using {multiprocessing.cpu_count()} cores."
    )
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures_list = {
            executor.submit(sum_per_run, run_name, group): run_name
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
