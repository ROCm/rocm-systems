# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import re
import sys
import subprocess
from pathlib import Path
from typing import Any, Dict, List


version_numbers = []
version_file = open("../VERSION", "r")
lines = version_file.readlines()
for line in lines:
    if line[0] == '#':
        continue
    version_numbers.append(line.strip())
version_number = ".".join(version_numbers)
left_nav_title = f"HIP {version_number} Documentation"

# for PDF output on Read the Docs
project = "HIP Documentation"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved."
version = version_number
release = version_number

external_toc_path = "./sphinx/_toc.yml"
external_projects_current_project = "hip"

# Add the _extensions directory to Python's search path
sys.path.append(str(Path(__file__).parent / 'extension'))

extensions = ["rocm_docs", "rocm_docs.doxygen", "sphinxcontrib.doxylink", "custom_directive"]

cpp_id_attributes = ["__global__", "__device__", "__host__", "__forceinline__", "static"]
cpp_paren_attributes = ["__declspec"]

suppress_warnings = ["etoc.toctree"]

numfig = False

html_static_path = ["sphinx/static/css"]
html_css_files = ["rocm_custom.css"]

exclude_patterns = [
    "./doxygen/mainpage.md",
    "./understand/glossary.md",
    './how-to/debugging_env.rst',
    "./reference/env_variables"
]

doxygen_root = "doxygen"
doxysphinx_enabled = True
doxygen_project = {
    "name": "doxygen",
    "path": "doxygen/xml",
}

html_theme = "rocm_docs_theme"
html_theme_options = {
    "announcement": f"This is ROCm 7.13.0 technology preview release documentation. For the latest production stream release, refer to <a id='rocm-banner' href='https://rocm.docs.amd.com/en/latest/'>ROCm documentation</a>.",
    "flavor": "generic",
    "header_title": f"ROCm™ 7.13.0 Preview",
    "header_link": f"https://rocm.docs.amd.com/en/7.13.0-preview/index.html",
    "version_list_link": "",
    "nav_secondary_items": {
        "GitHub": "https://github.com/ROCm/ROCm",
        "Community": "https://github.com/ROCm/ROCm/discussions",
        "Blogs": "https://rocm.blogs.amd.com/",
        "System and Infra Docs": "https://instinct.docs.amd.com/",
        "Support": "https://github.com/ROCm/ROCm/issues/new/choose",
    },
    "link_main_doc": False,
}
