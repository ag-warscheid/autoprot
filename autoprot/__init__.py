import sys

from . import common
from . import decorators
from . import r_helper

__version__ = "dev"
__author__ = "The autoprot contributors"
__license__ = "BSD-3-Clause"

# this is a pointer to the module object instance itself.
module_pointer = sys.modules[__name__]
# we can explicitly make assignments on it
module_pointer.check_r_install = False
# write the environment.txt file
common.generate_environment_txt()

from . import preprocessing
from . import analysis
from . import visualization
