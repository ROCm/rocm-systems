# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# sys.path.insert(0, os.path.abspath('.'))

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


# -- Project information -----------------------------------------------------
project = "omnitrace"
copyright = "2022, Advanced Micro Devices, Inc."
author = "Audacious Software Group"

project_root = os.path.normpath(os.path.join(os.getcwd(), "..", ".."))
version = open(os.path.join(project_root, "VERSION")).read().strip()
# The full version, including alpha/beta/rc tags
release = version

_docdir = os.path.realpath(os.getcwd())
_srcdir = os.path.realpath(os.path.join(os.getcwd(), ".."))
_sitedir = os.path.realpath(os.path.join(os.getcwd(), "..", "site"))
_staticdir = os.path.realpath(os.path.join(_docdir, "_static"))
_templatedir = os.path.realpath(os.path.join(_docdir, "_templates"))

if not os.path.exists(_staticdir):
    os.makedirs(_staticdir)

if not os.path.exists(_templatedir):
    os.makedirs(_templatedir)


# -- General configuration ---------------------------------------------------

install("sphinx_rtd_theme")

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_markdown_tables",
    "recommonmark",
    "breathe",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

from recommonmark.parser import CommonMarkParser

source_parsers = {".md": CommonMarkParser}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The master toctree document.
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

default_role = None

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_theme_options = {
    'analytics_id': 'G-1HLBBRSTT9',  #  Provided by Google in your dashboard
    'analytics_anonymize_ip': False,
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    # 'style_nav_header_background': 'white',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Breathe Configuration
breathe_projects = {"omnitrace": "_doxygen/xml"}
breathe_default_project = "omnitrace"
breathe_default_members = ('members', )
breathe_projects_source = {
    "auto": (
        os.path.join(project_root, "source"),
        [
            "lib/omnitrace-user/omnitrace/user.h",
        ],
    )
}

from pygments.styles import get_all_styles

# The name of the Pygments (syntax highlighting) style to use.
styles = list(get_all_styles())
preferences = ("emacs", "pastie", "colorful")
for pref in preferences:
    if pref in styles:
        pygments_style = pref
        break

from recommonmark.transform import AutoStructify

# app setup hook
def setup(app):
    app.add_config_value(
        "recommonmark_config",
        {
            "auto_toc_tree_section": "Contents",
            "enable_eval_rst": True,
            "enable_auto_doc_ref": False,
        },
        True,
    )
    app.add_transform(AutoStructify)
