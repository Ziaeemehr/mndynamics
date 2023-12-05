import sys
import os
from unittest.mock import MagicMock as Mock
from setuptools_scm import get_version

# sys.path.insert(0,os.path.abspath("../examples"))
sys.path.insert(0,os.path.abspath("../mndynamics"))

needs_sphinx = '1.3'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'numpydoc',
    'sphinx.ext.graphviz'
]

source_suffix = '.rst'
master_doc = 'index'
project = u'mndynamics'
copyright = u'2023, Abolfazl Ziaeemehr'

release = version = get_version(root='..', relative_to=__file__)

default_role = "any"
add_function_parentheses = True
add_module_names = False
html_theme = 'nature'
pygments_style = 'colorful'
# htmlhelp_basename = 'JiTCODEdoc'

numpydoc_show_class_members = False
autodoc_member_order = 'bysource'
graphviz_output_format = "svg"
toc_object_entries_show_parents = 'hide'

def on_missing_reference(app, env, node, contnode):
	if node['reftype'] == 'any':
		return contnode
	else:
		return None

def setup(app):
	app.connect('missing-reference', on_missing_reference)
