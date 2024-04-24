# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import subprocess

from rocm_docs import ROCmDocs

get_version_year = r'sed -n -e "s/^#define\ AMDSMI_LIB_VERSION_YEAR\ //p" ../include/amd_smi/amdsmi.h'
get_version_major = r'sed -n -e "s/^#define\ AMDSMI_LIB_VERSION_MAJOR\ //p" ../include/amd_smi/amdsmi.h'
get_version_minor = r'sed -n -e "s/^#define\ AMDSMI_LIB_VERSION_MINOR\ //p" ../include/amd_smi/amdsmi.h'
get_version_release = r'sed -n -e "s/^#define\ AMDSMI_LIB_VERSION_RELEASE\ //p" ../include/amd_smi/amdsmi.h'
version_year = subprocess.getoutput(get_version_year)
version_major = subprocess.getoutput(get_version_major)
version_minor = subprocess.getoutput(get_version_minor)
version_release = subprocess.getoutput(get_version_release)
name = f"AMD SMI {version_year}.{version_major}.{version_minor}.{version_release}"

external_toc_path = "./sphinx/_toc.yml"

docs_core = ROCmDocs(f"{name} Documentation")
docs_core.run_doxygen(doxygen_root="doxygen", doxygen_path="doxygen/docBin/xml")
docs_core.enable_api_reference()
docs_core.setup()
docs_core.html_theme_options = {
    "repository_url": "https://github.com/RadeonOpenCompute/amdsmi"
}

for sphinx_var in ROCmDocs.SPHINX_VARS:
    globals()[sphinx_var] = getattr(docs_core, sphinx_var)
