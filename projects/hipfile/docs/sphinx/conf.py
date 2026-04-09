# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Pylint is NOT happy with the naming scheme ROCm chose
# pylint: disable=invalid-name

"""This file is for configuring hipFile documentation"""

import os
from rocm_docs import ROCmDocs

version_number = "0.2.0"
left_nav_title = f"hipFile {version_number} documentation"

# for PDF output on Read the Docs
project = "rocSHMEM"
author = "Advanced Micro Devices, Inc."
# pylint: disable=redefined-builtin
copyright = "Copyright (c) Advanced Micro Devices, Inc. All rights reserved."
# pylint: enable=redefined-builtin
version = version_number
release = version_number
external_projects_current_project = "rocshmem"
extensions = ["sphinx_design"]
external_toc_path = "./sphinx/_toc.yml"

docs_core = ROCmDocs(left_nav_title)

doxygen_root = os.environ.get("DOXYGEN_ROOT")
doxygen_xml_dir = os.environ.get("DOXYGEN_XML_DIR")

if not doxygen_root or not doxygen_xml_dir:
    raise RuntimeError(
        "DOXYGEN_ROOT and DOXYGEN_XML_DIR must be set. "
        "Build this documentation via CMake, not directly with sphinx-build."
    )

docs_core.run_doxygen(doxygen_root=doxygen_root, doxygen_path=doxygen_xml_dir)

docs_core.setup()

# Transfer all Sphinx config variables into this module's global scope
for sphinx_var in ROCmDocs.SPHINX_VARS:
    globals()[sphinx_var] = getattr(docs_core, sphinx_var)
