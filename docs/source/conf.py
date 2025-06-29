# pylint: disable=invalid-name
# pylint: disable=redefined-builtin
import warnings

project = "l2gx"
copyright = "2024"
author = ""

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx_markdown_builder", "sphinx.ext.autodoc", "sphinx.ext.autosummary"]

autosummary_generate = True
# autodoc_default_options = {
#     "members": True,
#     "undoc-members": True,
#     "inherited-members": True,
#     "show-inheritance": True,
# }

# diable warnings
warnings.filterwarnings("ignore", category=UserWarning)

templates_path = ["_templates"]
exclude_patterns = []

# autodoc_mock_imports = ["torch", "torch.nn", "torch.optim", "torch_geometric"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
