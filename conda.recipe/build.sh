#!/bin/bash

# Exit on error
set -e

# Install the package using pip
$PYTHON -m pip install . -vv --no-deps --no-build-isolation
