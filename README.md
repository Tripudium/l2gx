
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)
[![Pylint](https://github.com/lotzma/L2GX/actions/workflows/pylint.yml/badge.svg)](https://github.com/lotzma/L2GX/actions/workflows/pylint.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

# l2gv2 - Local2Global

## Overview

## Documentation

Full documentation available [here](https://l2gx.readthedocs.io/en/latest/)

## Setup

**Supported Python Versions**: 3.10, 3.11, 3.12  
**Supported Operating Systems**: macOS, Linux

**Clone the repository** on your machine

```shell
git clone https://github.com/OxfordRSE/L2GX.git
```

We use [`uv`](https://docs.astral.sh/uv/) for Python package management. You
can install it on macOS or Linux using `brew install uv`. Alternatively, you
can use uv's [installation script](https://docs.astral.sh/uv/#installation).

[nox](https://nox.thea.codes) simplifies Python testing, particularly across
multiple Python versions. We provide a [noxfile.py](noxfile.py), which allows you
to run tests and perform linting with one command. You'll first need
to install nox:

```shell
brew install nox      # macOS
pipx install nox      # with pipx
sudo apt install nox  # debian
sudo dnf install nox  # fedora
uv tool install nox   # with uv
```

To run the tests and linting with
[pylint](https://pylint.readthedocs.io/en/stable/) and
[ruff](https://docs.astral.sh/ruff/):

```shell
nox
```

To display a list of tasks:

```shell
nox --list
```

To run only a task, such as `lint`, run `nox -s lint`.

If you are only using this library as a dependency, use:

```shell
pip install git+https://github.com/OxfordRSE/L2GX
```

For development, we highly recommend **installing the pre-commit hook** that
helps lint and autoformat on every commit:

```shell
brew install pre-commit     # macOS
pipx install pre-commit     # with pipx
sudo apt install pre-commit # debian
sudo dnf install pre-commit # fedora
uv tool install pre-commit  # with uv
```

To setup pre-commit hooks, run `pre-commit install` once in the repository;
this will ensure that checks run before every commit.

## License

This project is licensed under the [MIT](LICENSE) license.

## Contributors

The following people contributed to this project ([emoji key](https://allcontributors.org/docs/en/emoji-key)).


This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification.
