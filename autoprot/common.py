from typing import Union
from importlib import metadata
import pandas as pd


def set_default_kwargs(keyword_dict: Union[dict, None], default_dict: dict):
    """
    Compares a default parameter dict with the user-provided and updates the latter if necessary.

    Parameters
    ----------
    keyword_dict: dict or None
        user-supplied kwargs dict
    default_dict: dict
        Standard settings that should be applied if not specified differently by the user.
    """
    if keyword_dict is None:
        return default_dict
    for k, v in default_dict.items():
        if k not in keyword_dict.keys():
            keyword_dict[k] = v

    return keyword_dict


def generate_environment_txt():
    with open("environment.txt", "w") as env_:
        # write header with whitespace padding
        env_.write(f"{'Package':<30} {'Version':<15}\n")
        env_.write(f"{'-' * 30} {'-' * 15}\n")
        # get installed distributions (and sort)
        distributions = sorted(
            metadata.distributions(), key=lambda d: d.metadata["Name"].lower()
        )
        for dist in distributions:
            name = dist.metadata["Name"]
            version = dist.version
            env_.write(f"{name:<30} {version:<15}\n")


def get_uniprot_accession(
    df: pd.DataFrame, gene: str, organism: str
) -> Union[str, None]:
    """
    Finds the matching UniProt ID in a dataset given a gene name and a corresponding organism.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing UniProt IDs and gene names.
    gene: str
        Gene name.
    organism: str
        Organism name.

    Returns
    -------
    str or None
        UniProt ID if found, None otherwise.
    """
    gene = gene.upper()
    try:
        gene_in_gene = (df["GENE"].str.upper() == gene) & (df["ORGANISM"] == organism)
        gene_in_protein = (df["PROTEIN"].str.upper() == gene) & (
            df["ORGANISM"] == organism
        )

        uniprot_acc = df.loc[(gene_in_gene | gene_in_protein), "ACC_ID"].iloc[0]

        return uniprot_acc

    except IndexError:
        return None


def get_uniprot_sequence_locally(
    uniprot_acc: str, organism: str, uniprot: pd.DataFrame
) -> str:
    """
    Get sequence from a locally stored uniprot file by UniProt ID.

    Parameters
    ----------
    uniprot_acc: str
        UniProt ID.
    organism: str
        Organism name.
    uniprot: pd.DataFrame
        DataFrame containing UniProt IDs, gene names and sequences.

    Returns
    -------
    str or False
        Sequence if found, False otherwise.
    """
    if organism == "mouse":
        uniprot_organism = "Mus musculus (Mouse)"
    else:
        uniprot_organism = "Homo sapiens (Human)"

    sequence = uniprot["Sequence"][
        (uniprot["Entry"] == uniprot_acc) & (uniprot["Organism"] == uniprot_organism)
    ]
    try:
        sequence = sequence.values.tolist()[0]
    except IndexError:
        print(f"no match found for {uniprot_acc}")
        sequence = False
    return sequence
