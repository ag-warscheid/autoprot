# -*- coding: utf-8 -*-
"""
Autoprot Preprocessing Functions.

@author: Wignand, Julian, Johannes

@documentation: Julian
"""

import numpy as np
import pandas as pd
from importlib import resources
import re
import os
from subprocess import run, PIPE, STDOUT, CalledProcessError
import requests

from urllib import parse
from ftplib import FTP
import warnings
from typing import Union

@report
def read_csv(file, sep='\t', low_memory=False, **kwargs):
    r"""
    pd.read_csv with modified default args.

    Parameters
    ----------
    file : str
        Path to input file.
    sep : str, optional
        Column separator. The default is '\t'.
    low_memory : bool, optional
        Whether to reduce memory consumption by inferring dtypes from chunks. The default is False.
    **kwargs :
        see https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html.

    Returns
    -------
    pd.DataFrame
        The parsed dataframe.

    """
    return pd.read_csv(file, sep=sep, low_memory=low_memory, **kwargs)


def to_csv(df, file, sep='\t', index=False, **kwargs):
    r"""
    Write to CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to write.
    file : str
        Path to output file.
    sep : str, optional
        Column separator. The default is '\t'.
    index : bool, optional
        Whether to add the dataframe index to the output. The default is False.
    **kwargs :
        see https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html.

    Returns
    -------
    None.

    """
    df.to_csv(file, sep=sep, index=index, **kwargs)


def download_from_ftp(url, save_dir, login_name='anonymous', login_pw=''):
    r"""
    Download a file from FTP.

    Parameters
    ----------
    url : TYPE
        DESCRIPTION.
    save_dir : TYPE
        DESCRIPTION.
    login_name : str
        Login name for the FTP server.
        Default is 'anonymous' working for the PRIDE FTP server.
    login_pw : str
        Password for access to the FTP server.
        Default is ''
    Returns
    -------
    str
        Path to the downloaded file.

    Examples
    --------
    Download all files from a dictionary holding file names and ftp links and
    save the paths to the downloaded files in a list.

    >>> downloadedFiles = []
    >>> for file in ftpDict.keys():
    ...     downloadedFiles.append(pp.download_from_ftp(ftpDict[file], r'C:\Users\jbender\Documents\python_playground'))

    """
    path, file = os.path.split(parse.urlparse(url).path)
    ftp = FTP(parse.urlparse(url).netloc)
    ftp.login(login_name, login_pw)
    ftp.cwd(path)
    ftp.retrbinary("RETR " + file, open(os.path.join(save_dir, file), 'wb').write)
    print(f'Downloaded {file}')
    ftp.quit()
    return os.path.join(save_dir, file)


def fetch_from_pride(accession, term, ignore_caps=True):
    """
    Get download links files belonging to a PRIDE identifier.

    Parameters
    ----------
    accession : str
        PRIDE identifier.
    term : str
        Part of the filename belonging to the project.
        For example 'proteingroups'
    ignore_caps : bool, optional
        Whether to ignore capitalisation during matching of terms.
        The default is True.

    Returns
    -------
    file_locs : dict
        Dict mapping filenames to FTP download links.

    Examples
    --------
    Generate a dict mapping file names to ftp download links.
    Not that only files containing the string proteingroups are retrieved.

    >>> ftpDict = pp.fetch_from_pride("PXD031829", 'proteingroups')

    """
    js_list = requests.get(f'https://www.ebi.ac.uk/pride/ws/archive/v2/files/byProject?accession={accession}',
                           headers={'Accept': 'application/json'}).json()

    file_locs = {}

    for fdict in js_list:
        fname = fdict['fileName']
        if ignore_caps is True:
            fname = fname.lower()
            term = term.lower()
        if term in fname:
            for protocol in fdict['publicFileLocations']:
                if protocol['name'] == 'FTP Protocol':
                    file_locs[fname] = protocol['value']
                    print(f'Found file {fname}')
    return file_locs
