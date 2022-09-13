# This exports the environment without version numbers and with only the main packages
# it further removes the prefix path
conda env export | cut -f-2 -d '=' | grep -v ^prefix
