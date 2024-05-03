import sys
from . import common

# this is a pointer to the module object instance itself.
module_pointer = sys.modules[__name__]
# we can explicitly make assignments on it
module_pointer.check_r_install = False

# write the environment.txt file
common.generate_environment_txt()
