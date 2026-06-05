# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import re


with open('../CMakeLists.txt', encoding='utf-8') as f:
    match = re.search(r'set\(VERSION_STRING\s+([0-9.]+)\)', f.read())
    if not match:
        raise ValueError("VERSION_STRING not found in CMakeLists.txt")
    version_number = match[1]
left_nav_title = f"rocSHMEM {version_number} documentation"

# for PDF output on Read the Docs
project = "rocSHMEM"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved."
version = version_number
release = version_number

external_toc_path = "./sphinx/_toc.yml"
external_projects_current_project = "rocshmem"
cpp_id_attributes = ["__host__", "__global__", "__device__"]
exclude_patterns = ["README.md"]

doxygen_root = "doxygen"
doxygen_project = {
    "name": "doxygen",
    "path": "doxygen/xml",
}
extensions = [
    "rocm_docs",
    "rocm_docs.doxygen"
]
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