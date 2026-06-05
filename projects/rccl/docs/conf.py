# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import subprocess


name = "RCCL"
get_major = r'sed -n -e "s/^NCCL_MAJOR.*\([0-9]\+\).*/\1/p" ../makefiles/version.mk'
get_minor = r'sed -n -e "s/^NCCL_MINOR.*\([0-9]\{2,\}\).*/\1/p" ../makefiles/version.mk'
get_patch = r'sed -n -e "s/^NCCL_PATCH.*\([0-9]\+\).*/\1/p" ../makefiles/version.mk'
major = subprocess.getoutput(get_major)
minor = subprocess.getoutput(get_minor)
patch = subprocess.getoutput(get_patch)
version_number = f"{major}.{minor}.{patch}"

# for PDF output on Read the Docs
project = f"{name} Documentation"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved."
version = version_number
release = version_number

external_toc_path = "./sphinx/_toc.yml"
external_projects_current_project = "rccl"

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
