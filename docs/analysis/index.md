# Analysis

The analysis submodule contains functions for basic statistical tests, clustering and dimension reduction algorithms.
TIt is generally imported as follows and the functions are called as direct methods of the imported object.

```python
from autoprot import analysis as ana
# e.g. for the t-test
ana.ttest(args)
```

Nevertheless, the functions are organised in distinct files and the general structure of this documentation follows this organisation.

```{toctree}
:maxdepth: 2

clustering
functional
pca
qc
stat_test
stats
```
