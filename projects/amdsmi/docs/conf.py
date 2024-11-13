# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import re


# get version number to print in docs
def get_version_info(filepath):
    with open(filepath, "r") as f:
        content = f.read()

    version_pattern = (
        r"^#define\s+AMDSMI_LIB_VERSION_YEAR\s+(\d+)\s*$|"
        r"^#define\s+AMDSMI_LIB_VERSION_MAJOR\s+(\d+)\s*$|"
        r"^#define\s+AMDSMI_LIB_VERSION_MINOR\s+(\d+)\s*$|"
        r"^#define\s+AMDSMI_LIB_VERSION_RELEASE\s+(\d+)\s*$"
    )

    matches = re.findall(version_pattern, content, re.MULTILINE)

    if len(matches) == 4:
        version_year, version_major, version_minor, version_release = [
            match for match in matches if any(match)
        ]
        return version_year[0], version_major[1], version_minor[2], version_release[3]
    else:
        raise ValueError("Couldn't find all VERSION numbers.")


version_year, version_major, version_minor, version_release = get_version_info(
    "../include/amd_smi/amdsmi.h"
)
version_number = f"{version_year}.{version_major}.{version_minor}.{version_release}"

# project info
project = "AMD SMI"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved."
version = version_number
release = version_number

html_theme = "rocm_docs_theme"
html_theme_options = {"flavor": "rocm"}
html_title = f"AMD SMI {version_number} documentation"
exclude_patterns = ["rocm-smi-lib"]
suppress_warnings = ["etoc.toctree"]
external_toc_path = "./sphinx/_toc.yml"

external_projects_current_project = "amdsmi"
extensions = ["rocm_docs", "rocm_docs.doxygen"]

doxygen_root = "doxygen"
doxysphinx_enabled = True
doxygen_project = {
    "name": "AMD SMI C++ API reference",
    "path": "doxygen/docBin/xml",
}
