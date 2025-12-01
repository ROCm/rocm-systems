# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import re
import sys
from pathlib import Path

sys.path.append(str(Path("_extension").resolve()))


# get version number to print in docs
def get_version_info(filepath):
    with open(filepath, "r") as f:
        content = f.read()

    version_pattern = (
        r"^#define\s+AMDSMI_LIB_VERSION_MAJOR\s+(\d+)\s*$|"
        r"^#define\s+AMDSMI_LIB_VERSION_MINOR\s+(\d+)\s*$|"
        r"^#define\s+AMDSMI_LIB_VERSION_RELEASE\s+(\d+)\s*$"
    )

    matches = re.findall(version_pattern, content, re.MULTILINE)

    if len(matches) == 3:
        version_major, version_minor, version_release = [
            match for match in matches if any(match)
        ]
        return version_major[0], version_minor[1], version_release[2]
    else:
        raise ValueError("Couldn't find all VERSION numbers.")


version_major, version_minor, version_release = get_version_info(
    "../include/amd_smi/amdsmi.h"
)
version_number = f"{version_major}.{version_minor}.{version_release}"

# project info
project = "AMD SMI"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved."
version = version_number
release = version_number

html_theme = "rocm_docs_theme"
html_theme_options = {"flavor": "rocm"}
html_title = f"AMD SMI {version_number} documentation"
suppress_warnings = ["etoc.toctree"]
external_toc_path = "./sphinx/_toc.yml"

external_projects_current_project = "amdsmi"
extensions = ["rocm_docs", "rocm_docs.doxygen", "go_api_ref"]

doxygen_root = "doxygen"
doxysphinx_enabled = True
doxygen_project = {
    "name": "AMD SMI C++ API reference",
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
