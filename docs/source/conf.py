# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from datetime import datetime
from sphinx.ext.autodoc import ClassDocumenter, _


sys.path.insert(0, os.path.abspath('../..'))
# sys.path.insert(0, os.path.abspath('../../openwind'))
# sys.path.insert(0, os.path.abspath('../../examples'))

# -- Project information -----------------------------------------------------

project = 'Openwind Documentation'
copyright = '2019 - %s, Makutu' %datetime.now().year
author = 'Makutu'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.napoleon',
              'sphinx.ext.autodoc',
              'recommonmark',
              'sphinx.ext.viewcode',
              # 'sphinx_jekyll_builder',
              # 'sphinx.ext.autosummary',
              'sphinx.ext.autosectionlabel',
              # 'sphinx.ext.intersphinx',
              # 'matplotlib.sphinxext.plot_directive',
              ]

source_suffix = ['.rst', '.md']

master_doc = 'index'

# Make sure the target is unique
autosectionlabel_prefix_document = True

# autosummary_generate=True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['']
# exclude_patterns = ['build/*']

# Keep the source order of class and method
autodoc_member_order = 'bysource'

# -- Options for HTML output -------------------------------------------------

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# html_favicon='_static/logo_fav.png'
# html_logo='_static/logo.png'

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
# html_theme_options = {
#     'navigation_depth': 3,
# }


# Avoid having object in inheritance with automodule
# from : https://stackoverflow.com/questions/46279030/how-can-i-prevent-sphinx-from-listing-object-as-a-base-class

add_line = ClassDocumenter.add_line
line_to_delete = _(u'Bases: %s') % u':class:`object`'

def add_line_no_object_base(self, text, *args, **kwargs):
    if text.strip() == line_to_delete:
        return
    add_line(self, text, *args, **kwargs)

add_directive_header = ClassDocumenter.add_directive_header

def add_directive_header_no_object_base(self, *args, **kwargs):
    self.add_line = add_line_no_object_base.__get__(self)
    result = add_directive_header(self, *args, **kwargs)
    del self.add_line
    return result

ClassDocumenter.add_directive_header = add_directive_header_no_object_base
