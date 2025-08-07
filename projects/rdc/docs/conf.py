# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# for PDF output on Read the Docs
project = "ROCm Data Center tool"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved."

html_theme = "rocm_docs_theme"
html_theme_options = {"flavor": "rocm"}
html_title = f"RDC documentation"
external_toc_path = "./sphinx/_toc.yml"

external_projects_current_project = "rdc"
extensions = ["rocm_docs", "rocm_docs.doxygen"]

doxygen_root = "doxygen"
doxysphinx_enabled = True
doxygen_project = {
    "name": "ROCm Data Center Tool API reference",
    "path": "doxygen/xml",
}
