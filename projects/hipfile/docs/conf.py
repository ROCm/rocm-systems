# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import re

from rocm_docs import ROCmDocs

with open("../CMakeLists.txt", encoding="utf-8") as f:
    cmake = f.read()
parts = [
    re.search(rf"set\(AIS_LIBRARY_{level}\s+([0-9]+)", cmake).group(1)
    for level in ("MAJOR", "MINOR", "PATCH")
]
version_number = ".".join(parts)
left_nav_title = f"hipFile {version_number} Documentation"

# for PDF output on Read the Docs
project = "hipFile Documentation"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved."
version = version_number
release = version_number

external_toc_path = "./sphinx/_toc.yml"

docs_core = ROCmDocs(left_nav_title)
docs_core.run_doxygen(doxygen_root="doxygen", doxygen_path="doxygen/xml")
docs_core.enable_api_reference()
docs_core.setup()

external_projects_current_project = "hipfile"

for sphinx_var in ROCmDocs.SPHINX_VARS:
    globals()[sphinx_var] = getattr(docs_core, sphinx_var)

# Let the C/C++ domain ignore the API export macro so declarations like
# `HIPFILE_API int foo(void)` parse cleanly.
c_id_attributes = ["HIPFILE_API"]
cpp_id_attributes = ["HIPFILE_API"]
