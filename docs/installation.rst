==============
Installation
==============

The easiest way to install Python on Windows and Linux is the `Anaconda Python Distribution <https://www.anaconda.com/products/individual>`_.
Download the installer, follow the installation process and open an anaconda prompt.
On Windows this should be possible by searching for "Anaconda Prompt" in the start menu.
In Linux Anaconda usually comes with an install script that integrates anaconda into the standard shell (visible by the term "(base)" at the beginning of the shell prompt.

Downloading autoprot
====================

Download the current version of autoprot from the `autoprot repository <https://salzgitter.biologie.uni-freiburg.de/FunctionalProteomics/autoprot>`_ and save locally.
Alternatively you may download the files via git to easily maintain an updated version of autoprot.
For this, `install git <https://git-scm.com/downloads>`_ and clone the autoprot repo by::

   git clone https://salzgitter.biologie.uni-freiburg.de:443/FunctionalProteomics/autoprot.git

You may then keep your repo up to date using::

    git pull

in the autoprot folder.

Installing R
============

Autoprot relies on a few statistical analyses implemented and maintained in the R language.
To execute these script, you need a local copy of R running that python can access.
Head over to the `R repo <https://cran.r-project.org/bin/>`_ and download a version of R suitable for your operating system.
Alternatively, you may `install R using anaconda <https://docs.anaconda.com/anaconda/user-guide/tasks/using-r-language/>`_.
Locate the directory in which R was installed (usually something like "C:\Program Files\R\R-4.1.1" or within the anaconda folder).

Once you run autoprot, it will generate an autoprot.conf file in which you have to define the paths to your Rscript executable and to the RFunctions.R file inside the autoprot directory.
Run autoprot again and it will use the paths from the edited file.
Autoprot uses several R libraries that it will attempt to automatically install if they are not present.
However, a manual timeout is set to 10 min to prevent the script being caught in infinite loops.
If the automatic installation is not finished within 10 min, try increasing the timeout (in sec) in RHelper.py.
Moreover, automatic install using anaconda requires additional conda packages to be installed:

conda install -c conda-forge r-gmp
conda install -c conda-forge r-rmpfr

The autoprot virtual environment
================================

Autoprot comes with an "environment.yml" file containing the information which packages are required for autoprot.
You can directly install these packages into a new environment using::

    conda env create -f environment.yml

If the installation is successful,congratulations.
If not, read the troubleshooting_ section.
In any case, you may proceed by activating the environment::

    conda activate autoprot

If the installation was successful you can start exploring autoprot by starting an interactive jupyter notebook within the autoprot environment::

    jupyter notebook

Loading autoprot in jupyter
---------------------------

To use autoprot in your jupyter notebooks, you have to add the autoprot folder to the path variable::

    import sys
    sys.path.append(r"Z:\0_people\jbender\02_software\")
    import autoprot.preprocessing as pp

The path you add corresponds to the parent folder of your local autoprot installation.

Troubleshooting
===============
.. _:troubleshooting:

If the installation using environment.yml did not work, usually incompatible package versions between Windows and Linux or between different Anaconda installations are the problem.
Try locating which package did not install properly (e.g. by running autoprot and see where it breaks) and then install the missing packages using conda or pip::

    conda activate autoprot
    conda install PACKAGENAME

or using pip::

    conda activate autoprot
	pip install PACKAGENAME
