---
myst:
  html_meta:
    "description lang=en": |
      Top-level documentation for autoprot, with links to the rest
      of the site..
html_theme.sidebar_secondary.remove: true
---

# autoprot
Welcome to the documentation of the autoprot package.

Autoprot is a Python module for simplified analysis of quantitative mass spectrometry-based proteomics experiments
processed with the [MaxQuant](https://www.maxquant.org/) software.
It provides access to established functions written in both Python and R for statistical
testing and data transformation.
Moreover, it generates JavaScript-based interactive plots (based on the [plotly](https://plotly.com/python/) library)
that can be integrated into interactive web applications.
Thereby, autoprot offers standardised, fast and reliable proteomics data
analysis while maintaining the high customisability required to tailor the analysis pipeline to specific experiment

## How to install
For more information on how to install the package.
```{toctree}
:maxdepth: 1
installation
```
## The main package
The autoprot package is divided into three main parts: preprocessing, analysis and visualization.

### Preprocessing module
The preprocessing module is used to preprocess the protein sequences.

```{toctree}
:maxdepth: 2
preprocessing/index
```

### Analysis module
The analysis module is used to analyze the protein sequences.

```{toctree}
:maxdepth: 2
analysis/index

```

### Visualization module
The visualization module is used to visualize the results of the analysis.

```{toctree}
:maxdepth: 2
visualization/index
```