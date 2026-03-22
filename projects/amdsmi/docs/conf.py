# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import re
import sys
from pathlib import Path

from sphinx.errors import ConfigError

DOCS_DIR = Path(__file__).parent.resolve()
DOXYGEN_DIR = DOCS_DIR / "doxygen"
AMDSMI_DIR = DOCS_DIR.parent
AMDSMI_H = AMDSMI_DIR / "include" / "amd_smi" / "amdsmi.h"


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


version_major, version_minor, version_release = get_version_info(AMDSMI_H)

# project info
project = "AMD SMI"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) %Y Advanced Micro Devices, Inc. All rights reserved."
version = f"{version_major}.{version_minor}.{version_release}"
release = version

# theme and html output
html_theme = "rocm_docs_theme"
html_theme_options = {"flavor": "rocm", "show_toc_level": 2}
html_title = f"AMD SMI {version}"
html_static_path = ["sphinx/static"]
html_css_files = ["amdsmi_docs.css"]

# Sphinx build-related settings
# suppress_warnings = ["etoc.toctree"]

# Extension-related settings
sys.path.append(str(DOCS_DIR / "_extension"))

extensions = [
    "rocm_docs",
    "rocm_docs.doxygen",
    "amdsmi_docs.doxygen",
    "amdsmi_docs.go_api_ref",
    "sphinxcontrib.mermaid",
]
external_projects_current_project = "amdsmi"
external_toc_path = DOCS_DIR / "sphinx" / "_toc.yml"

myst_fence_as_directive = ["mermaid"]
myst_html_meta = {
    "keywords": "amd, smi, amdsmi, system, management, interface, rocm",
    "property=og:locale": "en_US",
}
# Use Jinja-style {{ key }} syntax to substitute values in .md files
myst_substitutions = {
    "AMDSMI_VERSION": version,
}

# doxygen-related settings
doxygen_root = DOXYGEN_DIR
breathe_projects = {"amdsmi": doxygen_root / "xml"}
breathe_default_project = "amdsmi"
breathe_domain_by_extension = {"h": "c"}
breathe_order_parameters_first = True
amdsmi_doxygen_tagfile = DOXYGEN_DIR / "tagfile.xml"


def generate_doxyfile():
    doxyfile_in = doxygen_root / "Doxyfile.in"
    doxyfile_out = doxygen_root / "Doxyfile"

    if not doxyfile_in.exists():
        raise ConfigError(f"Missing Doxyfile.in at {doxyfile_in}")

    with open(doxyfile_in) as f:
        content = f.read()

    content = content.replace("@PROJECT_NUMBER@", version)

    with open(doxyfile_out, "w") as f:
        f.write(content)


generate_doxyfile()

external_projects_remote_repository=""
