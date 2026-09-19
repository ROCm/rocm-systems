# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Browsable HTML view of one or more OpenSpec directories.

Run as ``python3 -m openspec_viewer`` or through the ``openspec_viewer.py``
wrapper next to this package.
"""

from .model import Project, Site, check, load_project, load_site
from .render import render

__all__ = ["Project", "Site", "check", "load_project", "load_site", "render"]
