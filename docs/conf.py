# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath('../.'))
# noinspection PyUnresolvedReferences
import autoprot

# -- Project information -----------------------------------------------------

project = 'autoprot'
project_copyright = '2024, The autoprot contributors'
author = autoprot.__author__
version = autoprot.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    'sphinx.ext.autosectionlabel',
    'matplotlib.sphinxext.plot_directive',
    'numpydoc',
    "myst_parser",
]

# dont rely on a manually generated class members list
numpydoc_show_class_members = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

autosummary_generate = True

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Plotting Directives -----------------------------------------------------
plot_pre_code = """
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from autoprot import analysis as ana
from autoprot import preprocessing as pp
from autoprot import visualization as vis
"""

plot_include_source = True  # include source code in the generated plots

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'pydata_sphinx_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

version = autoprot.__version__
html_theme_options = {'navigation_with_keys': False,
                      'logo': {
                          'image_light': '_static/logo.svg',
                          'image_dark': '_static/logo.svg',
                          'text': 'autoprot',
                      },
                      "navbar_start": ["navbar-logo", "version-switcher"],
                      "switcher": {
                          "json_url": "https://raw.githubusercontent.com/ag-warscheid/autoprot/dev/docs/_static/switcher.json",
                          "version_match": version,
                      }
                      }
