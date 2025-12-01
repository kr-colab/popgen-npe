# Configuration file for the Sphinx documentation builder.
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# Project information
project = "popgen-npe"
copyright = "2025"
author = "Andrew Kern"
release = "0.1.0"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx_rtd_theme",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx.ext.autosectionlabel",
    "myst_parser",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Several workflow modules depend on scientific packages that may not be
# available in the documentation environment. Mock them so autodoc still
# renders API references.
autodoc_mock_imports = [
    "tskit",
    "msprime",
    "demes",
    "torch",
    "numpy",
    "stdpopsim",
    "sbi",
    "dinf",
    "dinf.misc",
    "allel",
    "pandas",
    "tsinfer",
    "matplotlib",
    "seaborn",
    "lightning",
    "zarr",
]

# HTML output options
html_theme = "sphinx_rtd_theme"

# html_static_path = ["_static"]
