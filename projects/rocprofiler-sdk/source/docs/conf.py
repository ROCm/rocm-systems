# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
import subprocess as sp

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath(".."))


def install(package):
    sp.call([sys.executable, "-m", "pip", "install", package])


# Check if we're running on Read the Docs' servers
read_the_docs_build = os.environ.get("READTHEDOCS", None) == "True"

_srcdir = os.path.realpath(os.path.join(os.getcwd(), "../.."))


def build_doxyfile():
    sp.run(
        [
            "cmake",
            f"-DSOURCE_DIR={_srcdir}",
            "-DPROJECT_NAME='ROCprofiler-SDK'",
            f"-P {_srcdir}/source/docs/generate-doxyfile.cmake",
        ]
    )


build_doxyfile()

# -- Project information -----------------------------------------------------
project = "Rocprofiler SDK"
copyright = "2023-2025, Advanced Micro Devices, Inc."
author = "Advanced Micro Devices, Inc."

project_root = os.path.normpath(os.path.join(os.getcwd(), "..", ".."))
version = open(os.path.join(project_root, "VERSION")).read().strip()
# The full version, including alpha/beta/rc tags
release = version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "rocm_docs",
    "rocm_docs.doxygen",
]

doxygen_root = "."
doxysphinx_enabled = True

breathe_projects = {
    "rocprofiler-sdk": "_doxygen/rocprofiler-sdk/xml",
    "roctx": "_doxygen/roctx/xml",
}
breathe_default_project = "rocprofiler-sdk"

doxyfile = "rocprofiler-sdk.dox"

external_projects_current_project = "rocprofiler-sdk"
external_projects = []

master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]
external_toc_path = "./_toc.yml"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
suppress_warnings = ["etoc.toctree"]
nitpick_ignore = [
    ("cpp:identifier", "uint32_t"),
    ("cpp:identifier", "uint64_t"),
    ("cpp:identifier", "hsa_agent_s"),
    ("cpp:identifier", "ihipStream_t"),
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = "rocm_docs_theme"
html_theme_options = {"flavor": "rocm"}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_title = f"ROCprofiler-SDK {version} Documentation"
