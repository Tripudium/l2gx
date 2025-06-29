Introduction
============

Setup
-----

**Supported Python Versions**: 3.10, 3.11, 3.12

**Supported Operating Systems**: macOS, Linux

Clone the repository on your machine

.. code-block:: bash

    git clone https://github.com/OxfordRSE/L2GX.git

Setup the virtual environment

1. Create and activate a virtual environment:

   .. code-block:: bash

       python3 -m venv .venv
       source .venv/bin/activate

2. Install the dependencies:

   .. code-block:: bash

       pip install . --find-links https://data.pyg.org/whl/torch-{version}%2Bcpu.html

    For the above, select 2.5.1 for macOS and 2.4.1 for Linux. Note that this
    installs CPU versions of the dependencies. To install GPU versions, consult the
    pytorch-geometric_ documentation for the appropriate repository links, or visit
    https://data.pyg.org/whl/ to see all possible torch/GPU supported versions.


3. To build docs

   .. code-block:: bash

      pip install '.[docs]'

4. To install the development dependencies

   .. code-block:: bash

      pip install '.[dev]'

.. _pytorch-geometric: https://pypi.org/project/torch-geometric/

GitHub actions & pre-commit integration
------------------------------------------

This project uses `pylint` for code quality and linting checks, integrated with both GitHub Actions for continuous integration and `pre-commit` (`pre-commit.com <href https://pre-commit.com>`_) for local development checks.

The pylint GitHub workflow file is located at ``.github/workflows/pylint.yml``

The `pre-commit` hook is set up to automatically run `pylint` on files within the ``l2gv2`` folder whenever code is committed locally (``.pre-commit-config.yaml``). This helps catch and resolve linting issues before they are pushed to the repository.

To set up the pre-commit hook

1. Install `pre-commit` if it’s not already installed:

   .. code-block:: bash

      pip install pre-commit

2. Install the pre-commit hooks defined in the configuration file:

   .. code-block:: bash

      pre-commit install

3. Run the `pre-commit` hooks manually to check all files:

   .. code-block:: bash

      pre-commit run --all-files


Documentation
-------------

The project is setup to generate documentation with `Sphinx <https://www.sphinx-doc.org/en/master/index.html>`_).

Documentation is available at `l2gx.readthedocs.io <https://l2gx.readthedocs.io>`_

Generate the package documentation

.. code-block:: bash

   sphinx-apidoc -o docs/source/reference -H "Code Reference" l2gv2

Generate `html` or `markdown` documentation locally

.. code-block:: bash

   sphinx-build html docs/source/ docs/build/
   sphinx-build markdown docs/source/ docs/build/

Automatically refresh and serve the html documentation locally at `http://127.0.0.1:8000 <http://127.0.0.1:8000>`_ upon file updates during development

.. code-block:: bash

   sphinx-autobuild docs/source docs/build/html
