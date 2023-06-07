# autoprot

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![made-with-r](https://img.shields.io/badge/Made%20with-R-1f425f.svg)](https://www.r-project.org/)
[![made-with-sphinx-doc](https://img.shields.io/badge/Made%20with-Sphinx-1f425f.svg)](https://www.sphinx-doc.org/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Website shields.io](https://img.shields.io/website-up-down-green-red/http/ag-warscheid.github.io/autoprot/)](https://ag-warscheid.github.io/autoprot/)

![Mastodon Follow](https://img.shields.io/mastodon/follow/109993892962152197?domain=https%3A%2F%2Fmstdn.science&style=social)
![Twitter Follow](https://img.shields.io/twitter/follow/warscheidalb?style=social)

## Description

autoprot streamlines and simplifies proteomics data analysis from preprocessing to visualisation.

Its main features are:
- Works with [Pandas dataframes](https://pandas.pydata.org/)
- Is modularised so that only a required submodule can be loaded for a certain task
- Connects with established [R](rhttps://r-project.org) functions for advances bio-statistical analysis
- Supports interactive visualisations made with [Plotly](https://plotly.com/)

![logo.png](logo.png)

## Installation

- Generate a new python environment using [anaconda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
or [pip](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).
- Install required Python packages (see [requirements.txt](requirements.txt)) in the environment
- Download or clone autoprot
  - If you clone the repository, make sure to include the dependencies as submodules (example below)

```
git clone --recurse-submodules  git://github.com/foo/bar.git
```

- Next you need to install R. Please follow the instructions at the [R manual](https://cran.r-project.org/index.html) and install R to a custom location
- Start autoprot by importing it from any Python console you like. It will generate an autoprot.conf file that you need to edit.
  - Insert the path to your Rscript executable that you just installed as value for the R variable
  - The RFunctions variable should point the RFunctions.R file from autoprot.
- You can now either try to start using autoprot (it will automatically install required R packages) or manually trigger the install (recommended).
  - For this open your R console and start Functions.R with Rscript

```
C:\Users\USer\Documents\R\R-4.1.3\bin\Rscript.exe RFunctions.R
```

- You can now start with e.g. with the example notebook [01_ap-ms.ipynb](examples%2F01_ap-ms.ipynb) provided with autoprot.
- A more detailed description of the installation can be found in [the documentation](https://ag-warscheid.github.io/autoprot/installation.html).

## Documentation
Please find the full documentation including function references at https://ag-warscheid.github.io/autoprot/installation.html.

## Contribution
If you want to contribute to the code or found a bug, please feel free to submit an issue or a pull request to https://github.com/ag-warscheid/autoprot. 
