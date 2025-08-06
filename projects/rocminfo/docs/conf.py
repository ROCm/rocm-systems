# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import re


html_theme = "rocm_docs_theme"
html_theme_options = {"flavor": "rocm"}

extensions = ["rocm_docs"]
external_toc_path = "./sphinx/_toc.yml"

with open('../CMakeLists.txt', encoding='utf-8') as f:
    match = re.search(r'get_package_version_number\(\"?([0-9.]+)[^0-9.]+', f.read())
    if not match:
        raise ValueError("VERSION not found!")
    version_number = match[1]

version = version_number
release = version_number
html_title = f"rocminfo {version} Documentation"
project = "rocminfo"
author = "Advanced Micro Devices, Inc."
copyright = (
    "Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved."
)
