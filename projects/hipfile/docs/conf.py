# Configuration file for the Sphinx documentation builder.
#
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import re

# Single source of truth for the version is the library's CMakeLists.txt.
with open("../CMakeLists.txt", encoding="utf-8") as f:
    _cmake = f.read()


def _cmake_version_part(part):
    match = re.search(rf"set\(AIS_LIBRARY_{part}\s+(\d+)\)", _cmake)
    if not match:
        raise ValueError(f"AIS_LIBRARY_{part} not found in CMakeLists.txt")
    return match.group(1)


version_number = (
    f"{_cmake_version_part('MAJOR')}."
    f"{_cmake_version_part('MINOR')}."
    f"{_cmake_version_part('PATCH')}"
)

project = "hipFile"
author = "Advanced Micro Devices, Inc."
copyright = "2024-2026, Advanced Micro Devices, Inc."
version = version_number
release = version_number
html_title = f"hipFile {version_number} documentation"

extensions = [
    "rocm_docs",
    "sphinxcontrib.mermaid",
    "breathe",
]

# breathe renders the doxygen XML referenced by the reference/api-*.rst pages
breathe_projects = {"hipfile": "doxygen/xml"}
breathe_default_project = "hipfile"

# let the C/C++ domain ignore the export macro so `HIPFILE_API int foo(void)` parses cleanly
c_id_attributes = ["HIPFILE_API"]
cpp_id_attributes = ["HIPFILE_API"]

exclude_patterns = ["_build", "_diffs", "Thumbs.db", ".DS_Store"]

html_theme = "rocm_docs_theme"
html_theme_options = {"flavor": "rocm"}

external_toc_path = "sphinx/_toc.yml"
external_projects_current_project = "hipfile"
