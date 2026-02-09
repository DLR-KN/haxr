[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18759623.svg)](https://doi.org/10.5281/zenodo.18759623)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyPI](https://img.shields.io/pypi/v/haxr)](https://pypi.org/project/haxr/)
[![CI](https://img.shields.io/github/actions/workflow/status/DLR-KN/haxr/ci.yml?branch=main)](https://github.com/DLR-KN/haxr/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[[Documentation 📚]][doc]

# haxr

``haxr`` is the companion library for the [**HAXR**][zenodo] dataset.
It is an easy-to-use, local-cache-based access layer for the dataset files and a collection of handy utilities for working with the data.

``haxr`` releases are available as wheel packages for macOS, Windows and Linux on [PyPI][pypi].
Install it using pip:

```bash
pip install haxr
```

and find the documentation of the API [here][doc].

As a teaser, here is an example that plots the 1000th cycle of `amerikahoeft_08-UTC.hdf`:
```python
import haxr
import matplotlib.pyplot as plt


def plot_cycle(store, station, split_hour_utc, cycle, file_name="demo.png"):
    chunk = store.get_chunk(station=station, split_hour_utc=split_hour_utc)
    with store.open(chunk.radar_file) as f:
        df = haxr.utilities.load_cycle(f, cycle)
        h, az_edges, r_edges = haxr.utilities.fill_histogram(
            az=df["az (degree)"], r=df["r (meter)"], weights=df["amp"]
        )
        x, y, v = haxr.utilities.histogram_to_cartesian_meshgrid(h, az_edges, r_edges)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.pcolormesh(x, y, v, cmap="plasma")
    ax.set_aspect("equal")
    ax.set_axis_off()

    fig.savefig(file_name, dpi=300)


with haxr.Store(base_url=haxr.DOI.latest) as store:
    plot_cycle(store, "amerikahoeft", split_hour_utc=8, cycle=1000)
```

<img src="https://zenodo.org/records/18759623/files/demo.png" alt="demo.png" width="500">

# Contributing 👷

Contributions are welcome!
If you find any issues or have suggestions for improvements, please feel free to submit a pull request.

As a rule of thumb:

- For bugs and feature requests, please open a GitHub issue.
- For changes, please open a pull request.
- Small, focused PRs are easiest to review.
- Improvements for tests and doc are highly appreciated.

## Development Setup

Clone the repository and set up a development environment.
The project defines dependency groups (see `pyproject.toml`), including a `dev` group for development tools and a `docs` group for building the documentation.

### Option A: uv (recommended)

```bash
git clone https://github.com/DLR-KN/haxr
cd haxr
uv sync --all-groups
```

If you only need the documentation toolchain:

```bash
uv sync --group docs
```

### Option B: pip + venv

```bash
git clone https://github.com/DLR-KN/haxr
cd haxr
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e . --group dev --group docs
```

## Pre-commit Hooks

To maintain code quality and avoid pushing invalid commits, we recommend using [pre-commit hooks][pre-commit].
These hooks perform automated checks before commits are made.
To set up pre-commit hooks (part of the `dev` group), follow these steps:

```bash
pre-commit install
```

Optionally, run the hooks once on all files:

```bash
pre-commit run --all-files
```

## Tests

We use [pytest] to orchestrate our tests.
Run all tests via

```bash
uv run pytest
```

or, in a plain `venv`,

```bash
python -m pytest
```

Make sure to run tests before submitting a pull request to ensure that everything is functioning as expected.

## Generate Documentation (optional)

We use [Sphinx][sphinx] to automatically build the project documentation from the source tree and the docstrings in the codebase.
All documentation-specific dependencies are bundled in the [uv] dependency group `docs`, so you can easily install them via

```bash
uv sync --group docs
```

Then, build the HTML docs:

```bash
uv run --group docs make -C docs html
```

and open `docs/build/html/index.html` in your browser.

[doc]: https://dlr-kn.github.io/haxr/
[github]: https://github.com/DLR-KN/haxr
[pre-commit]: https://pre-commit.com
[pytest]: https://docs.pytest.org/en/stable/
[pypi]: https://pypi.org/project/haxr/
[sphinx]: https://www.sphinx-doc.org/
[uv]: https://docs.astral.sh/uv/
[zenodo]: https://doi.org/10.5281/zenodo.18759623
