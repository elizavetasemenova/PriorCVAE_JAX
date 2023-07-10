# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import os

import priorCVAE
import sphinx_book_theme

sys.path.insert(0, os.path.abspath(os.sep.join((os.curdir, '..'))))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PriorCVAE'
copyright = ''
author = 'The PriorCVAE Authors'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_nb',
    # 'codediff',
    # "sphinx.ext.githubpages",
    'sphinx_design',
]


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = ['.rst', '.ipynb', '.md']

autosummary_generate = True

master_doc = 'index'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"


# title of the website
# html_title = 'PriorCVAE'

html_static_path = ['_static']

html_theme_options = {
    'repository_url': 'https://github.com/elizavetasemenova/PriorCVAE_JAX',
    'use_repository_button': True,     # add a 'link to repository' button
    'use_issues_button': False,        # add an 'Open an Issue' button
    'show_navbar_depth': 1,
}

always_document_param_types = True

# -- Options for myst ----------------------------------------------
# uncomment line below to avoid running notebooks during development
nb_execution_mode = 'off'
# Notebook cell execution timeout; defaults to 30.
nb_execution_timeout = 100
# List of patterns, relative to source directory, that match notebook
# files that will not be executed.
myst_enable_extensions = ['dollarmath']
# nb_execution_excludepatterns = [
#   'getting_started.ipynb', 
# ]
