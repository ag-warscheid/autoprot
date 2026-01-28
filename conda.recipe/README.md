# Conda Recipe for autoprot

This directory contains the conda recipe for building and distributing autoprot as a conda package.

## Files

- `meta.yaml`: The main recipe file containing package metadata, dependencies, and build instructions
- `build.sh`: Build script for Unix-based systems (Linux, macOS)
- `bld.bat`: Build script for Windows systems

## Building the Package

To build the conda package locally:

```bash
# Install conda-build if not already installed
conda install conda-build

# Build the package from the recipe directory
conda build conda.recipe

# Or build from the parent directory
cd ..
conda build conda.recipe
```

## Installing the Package

After building, you can install the package locally:

```bash
conda install --use-local autoprot
```

## Publishing to conda-forge

To publish autoprot to conda-forge:

1. Fork the [staged-recipes](https://github.com/conda-forge/staged-recipes) repository
2. Copy the contents of this `conda.recipe` directory to `staged-recipes/recipes/autoprot/`
3. Update the `source.sha256` field in `meta.yaml` with the SHA256 checksum of the source tarball
4. Submit a pull request to conda-forge/staged-recipes
5. Once merged, a feedstock repository will be created automatically

## Publishing to Anaconda Cloud

Alternatively, you can publish to Anaconda Cloud:

```bash
# Install anaconda-client
conda install anaconda-client

# Login to Anaconda Cloud
anaconda login

# Upload the built package
anaconda upload /path/to/conda-bld/noarch/autoprot-0.3.0-*.tar.bz2
```

## Notes

- The recipe is set to `noarch: python` since autoprot is a pure Python package
- R dependencies are included as runtime requirements
- The package requires Python 3.8 or higher
- Make sure to update the version number in `meta.yaml` when releasing new versions
- Remember to calculate and add the SHA256 checksum for source tarballs when publishing
