# ROCprofiler Documentation

## Build Instructions

1. Install conda
    - `wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh`
    - `bash miniconda.sh -b -p /opt/conda`
    - `export PATH=${PATH}:/opt/conda`
2. Install conda environment
    - `source activate`
    - `conda env create -n rocprofiler-docs -f environment.yml`
    - `conda activate rocprofiler-docs`
3. Build the docs
    - `../scripts/update-docs.sh`
    - HTML docs will be located in `_build/html`

## Developer Information

If you create a new page, add the name of the new markdown file (without extension) to the [index.md](index.md) file.
