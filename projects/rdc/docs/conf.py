# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import re

# for PDF output on Read the Docs
project = "ROCm Data Center tool"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved."

with open('../CMakeLists.txt', encoding='utf-8') as f:
    match = re.search(r'.*\bget_version_from_tag\(\"([0-9.]+)\"', f.read())
    if not match:
        raise ValueError("VERSION not found!")
    version_number = match[1]

html_theme = "rocm_docs_theme"
html_theme_options = {
    "flavor": "rocm",
    "repository_url": "https://github.com/ROCm/rocm-systems",
    "path_to_docs": "projects/rdc/docs",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_download_button": True,
}
html_title = f"RDC {version_number} documentation"
external_toc_path = "./sphinx/_toc.yml"

external_projects_current_project = "rdc"
extensions = ["rocm_docs", "rocm_docs.doxygen"]

doxygen_root = "doxygen"
doxysphinx_enabled = True
doxygen_project = {
    "name": "ROCm Data Center Tool API reference",
    "path": "doxygen/xml",
}

# Generate and publish llms.txt
rocm_docs_generate_llms = True
