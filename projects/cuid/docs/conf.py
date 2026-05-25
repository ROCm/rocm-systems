# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import re
from pathlib import Path

def get_version_info(filepath):
    with open(filepath, "r") as f:
        content = f.read()

    version_pattern = (
        r"^#define\s+AMDCUID_LIB_VERSION_MAJOR\s+(\d+)\s*$|"
        r"^#define\s+AMDCUID_LIB_VERSION_MINOR\s+(\d+)\s*$|"
        r"^#define\s+AMDCUID_LIB_VERSION_PATCH\s+(\d+)\s*$"
    )

    matches = re.findall(version_pattern, content, re.MULTILINE)

    if len(matches) == 3:
        version_major, version_minor, version_patch = [
            match for match in matches if any(match)
        ]
        return version_major[0], version_minor[1], version_patch[2]
    else:
        raise ValueError("Couldn't find all VERSION numbers in amd_cuid.h.")


version_major, version_minor, version_patch = get_version_info(
    "../lib/include/amd_cuid.h"
)
version_number = f"{version_major}.{version_minor}.{version_patch}"

# -- Project information -----------------------------------------------------
project = "CUID"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved."
version = version_number
release = version_number

# -- General configuration ---------------------------------------------------
html_theme = "rocm_docs_theme"
html_theme_options = {"flavor": "rocm"}
html_title = f"CUID {version_number} documentation"
suppress_warnings = ["etoc.toctree"]
external_toc_path = "./sphinx/_toc.yml"

external_projects_current_project = "amdcuid"
extensions = ["rocm_docs", "rocm_docs.doxygen"]

doxygen_root = "doxygen"
doxysphinx_enabled = True
doxygen_project = {
    "name": "AMD CUID C API reference",
    "path": "doxygen/docBin/xml",
}


def generate_doxyfile(app, _):
    doxyfile_in = Path(app.confdir) / doxygen_root / "Doxyfile.in"
    doxyfile_out = Path(app.confdir) / doxygen_root / "Doxyfile"

    if not doxyfile_in.exists():
        from sphinx.errors import ConfigError

        raise ConfigError(f"Missing Doxyfile.in at {doxyfile_in}")

    with open(doxyfile_in) as f:
        content = f.read()

    content = content.replace("@PROJECT_NUMBER@", version_number)

    with open(doxyfile_out, "w") as f:
        f.write(content)


def setup(app):
    app.connect("config-inited", generate_doxyfile, priority=100)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
