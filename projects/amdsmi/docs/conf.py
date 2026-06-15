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


# Get version string to print in docs
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

# Project info
project = "AMD SMI"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) %Y Advanced Micro Devices, Inc. All rights reserved."
version =  f"{version_major}.{version_minor}.{version_release}"
release = version

# Theme-related settings
html_theme = "rocm_docs_theme"
html_theme_options = {"flavor": "rocm"}
html_title = f"AMD SMI {version}"
html_static_path = ["static"]
html_css_files = ["amdsmi_docs.css"]

# Extension-related settings
sys.path.append(str(DOCS_DIR / "extension"))
extensions = [
    "rocm_docs",
    "rocm_docs.doxygen",
    "amdsmi_docs.doxygen",
    "amdsmi_docs.go_api_ref",
    "sphinxcontrib.mermaid",
]

external_toc_path = "sphinx/_toc.yml"
external_projects_current_project = "amdsmi"
myst_fence_as_directive = ["mermaid"]
# For substitutions in MyST Markdown files using Jinja-style
# double-curly-braces.
# Usage:
#   ```md
#   {{ AMDSMI_VERSION }}
#   ```
myst_substitutions = {"AMDSMI_VERSION": version}

# Builder-related settings
exclude_patterns = [
    "extension/**",
    "CLAUDE.md",
    "AGENTS.md",
]

# Doxygen-related settings
doxygen_root = DOCS_DIR / "doxygen"
breathe_projects = {"amdsmi": doxygen_root / "_out" / "xml"}
breathe_default_project = "amdsmi"
breathe_domain_by_extension = {"h": "c"}
amdsmi_doxygen_tagfile = doxygen_root / "_out" / "tagfile.xml"
doxysphinx_enabled = False


# Make Doxyfile consistent with this Sphinx config
def generate_doxyfile(_app, _config):
    doxyfile_in = doxygen_root / "Doxyfile.in"
    doxyfile_out = doxygen_root / "Doxyfile"

    if not doxyfile_in.exists():
        raise ConfigError(f"Missing Doxyfile.in at {doxyfile_in}")

    replacements = {
        "@PROJECT_NUMBER@": version,
        "@OUTPUT_DIRECTORY@": str(doxygen_root / "_out"),
        "@GENERATE_TAGFILE@": str(amdsmi_doxygen_tagfile),
    }

    def _replace(m):
        key = m.group(0)
        if key not in replacements:
            raise ConfigError(f"Unknown template variable {key} in Doxyfile.in")
        return replacements[key]

    content = re.sub(r"@\w+@", _replace, doxyfile_in.read_text())
    doxyfile_out.write_text(content)


def setup(app):
    app.connect("config-inited", generate_doxyfile, priority=100)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
