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
Install R in a separate directory in your home folder (i.e. under C:\\Users\USer\Documents\R\R-4.1.3).
Note that you do not need admin privileges for installing R.
Alternatively, you may `install R using anaconda <https://docs.anaconda.com/anaconda/user-guide/tasks/using-r-language/>`_,
however, we think that a clean R install outside anaconda is the better way to go.
Locate the directory in which R was installed and find the path to the Rscript.exe (or

Downloading pre-build binaries on Windows
-----------------------------------------
You can find pre-compiled binaries for all R versions at https://cran.r-project.org/bin/windows/base/old/.
It is usually a good idea to choose one version behind the most recent as most likely not all packages required for autoprot
will have updated to the newest R version.

Installation from source on Linux
---------------------------------

If you use R on Linux, you can either install R system-wide using the standard package installer or compile a local version in your home dir.
This has the advantage that you can easily switch between R versions and even have multiple R installations with different
libraries next to each other. For this, have the following libs ready (on a recent Ubuntu 22.04 installation)::

      sudo apt install build-essential gfortran libreadline8 libreadline-dev libxt-dev zlib1g zlib1g-dev bzip2
      libbz2-dev liblzma-dev openjdk-18-jre-headless openjdk-18-jre libssl-dev libcurl4-openssl-dev libcurl4-gnutls-dev
      libfontconfig1-dev libharfbuzz-dev libfribidi-dev libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev libgmp3-dev
      cmake libmpfr-dev

During compilation, R looks for the ICU library and may end up finding it in a close anaconda installation (leading to errors in the compilation).
If this happens to you, compile R without the ICU libraries using::

    ./configure --without-ICU
    make

Linking R with autoprot
-----------------------

Once you run autoprot, it will generate an autoprot.conf file in which you have to define the paths to your Rscript executable and to the RFunctions.R file inside the autoprot directory.
Run autoprot again and it will use the paths from the edited file.
Autoprot uses several R libraries that it will attempt to automatically install if they are not present.

Setting up the Python environment
=================================

Autoprot comes with an "autoprot.yml" file containing the information which packages are required for autoprot.
It is recommended to manage environment with the anaconda package manager (see https://www.anaconda.com/).
For increased performance during install or update of packages (i.e. if you are actively changing autoprot) it is recommended
to also install `mamba <https://mamba.readthedocs.io/en/latest/>`_::

    conda install mamba -c conda-forge

Using either conda or mamba (substitute mamba for conda) you can install from the yml file by::

    mamba env create -f autoprot.yml

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

Installation of R using anaconda requires additional conda packages to be installed::

    conda install -c conda-forge r-gmp
    conda install -c conda-forge r-rmpfr

