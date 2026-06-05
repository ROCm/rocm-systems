# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import re


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

extensions = ["rocm_docs"]
external_toc_path = "./sphinx/_toc.yml"

external_projects_current_project = "rocr_debug_agent"

with open('../CMakeLists.txt', encoding='utf-8') as f:
    match = re.search(r'project\(rocm-debug-agent VERSION\s+\"?([0-9.]+)[^0-9.]+', f.read())
    if not match:
        raise ValueError("VERSION not found!")
    version_number = match[1]

version = version_number
release = version_number
html_title = f"ROCR Debug Agent {version} Documentation"
project = "rocr_debug_agent"
author = "Advanced Micro Devices, Inc."
copyright = (
    "Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved."
)
