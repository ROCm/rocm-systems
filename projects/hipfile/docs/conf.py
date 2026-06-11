# Configuration file for the Sphinx documentation builder.
#
# For a full list of Sphinx configuration options, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Pylint is NOT happy with the naming scheme ROCm chose in ROCmDocs
# pylint: disable=invalid-name

"""This file is for configuring hipFile documentation"""

import re

from pathlib import Path
from sphinx.errors import ConfigError

# -- Version information -----------------------------------------------------


def get_version_info(filepath):
    """Get the version from the header"""

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    version_pattern = (
        r"^#define\s+HIPFILE_VERSION_MAJOR\s+(\d+)\s*$|"
        r"^#define\s+HIPFILE_VERSION_MINOR\s+(\d+)\s*$|"
        r"^#define\s+HIPFILE_VERSION_PATCH\s+(\d+)\s*$"
    )

    matches = re.findall(version_pattern, content, re.MULTILINE)

    if len(matches) != 3:
        raise ValueError("Couldn't find all VERSION numbers in " + filepath)

    major, minor, patch = [match for match in matches if any(match)]

    return major[0], minor[1], patch[2]


version_major, version_minor, version_patch = get_version_info("../include/hipfile.h")
version_number = f"{version_major}.{version_minor}.{version_patch}"
left_nav_title = f"hipFile {version_number} documentation"

# -- Project information -----------------------------------------------------

project = "hipFile"
author = "Advanced Micro Devices, Inc."
# pylint: disable=redefined-builtin
copyright = "Copyright (c) Advanced Micro Devices, Inc. All rights reserved."
# pylint: enable=redefined-builtin
version = version_number
release = version_number

# -- General configuration ---------------------------------------------------

html_theme = "rocm_docs_theme"
html_theme_options = {"flavor": "rocm"}
html_title = f"hipFile {version_number} documentation"
suppress_warnings = ["etoc.toctree"]
external_toc_path = "./sphinx/_toc.yml"
external_projects_current_project = "hipfile"
extensions = ["rocm_docs", "rocm_docs.doxygen", "sphinx_design"]
exclude_patterns = ["README.md"]

# -- Doxygen configuration ---------------------------------------------------

doxygen_root = "doxygen"
doxysphinx_enabled = True
doxygen_project = {
    "name": "hipFile C API reference",
    "path": "doxygen/xml",
}


def generate_doxyfile(app, _):
    """Process Doxyfile (normally done with CMake)"""

    doxyfile_in = Path(app.confdir) / doxygen_root / "Doxyfile.in"
    doxyfile_out = Path(app.confdir) / doxygen_root / "Doxyfile"

    if not doxyfile_in.exists():
        raise ConfigError(f"Missing Doxyfile.in at {doxyfile_in}")

    with open(doxyfile_in, encoding="utf-8") as f:
        content = f.read()

    # Fix up version
    content = content.replace("@AIS_LIBRARY_VERSION@", version_number)

    # Fix up Doxygen markup location
    content = content.replace("@AIS_DOXYFILE_INPUT@", "../../include/hipfile.h")

    with open(doxyfile_out, "w", encoding="utf-8") as f:
        f.write(content)


# -- Sphinx Setup ------------------------------------------------------------


def setup(app):
    """Sphinx setup"""
    app.connect("config-inited", generate_doxyfile, priority=100)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
