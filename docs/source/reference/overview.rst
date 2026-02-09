Overview
========

``haxr`` is the companion library for the `HAXR <https://doi.org/10.5281/zenodo.18759623>`_ dataset.
It provides:

- an easy, local-cache-based access layer for the dataset files (radar + AIS + station
  metadata), and
- a handful of pragmatic utilities for working with the radar stream (cycles/frames)
  and for building quick visualizations (histogram helpers).

The dataset is distributed as a set of per-station, per-hour :class:`~haxr.store.Chunk`. Each chunk
contains one radar HDF5 file and one AIS CSV file.

Installation
------------

Install the latest stable release from PyPI:

.. code-block:: bash

    pip install haxr

If you want to work with/on the most recent version, find further instructions in the ``README.md`` of our `GitHub repository <https://github.com/DLR-KN/haxr>`_.
Contributions are welcome! Please feel free to open an issue or pull request.

Quickstart
----------

The main entry point is :class:`~haxr.store.Store`. It manages a local cache directory
and (optionally) downloads missing files from a dataset release endpoint.

.. code-block:: python

    from haxr import DOI, Store
    from haxr.utilities import load_cycle, iter_frames

    # Pick a release endpoint. For reproducible workflows prefer a versioned DOI.
    with Store(base_url=DOI.latest) as store:
        # Discover what’s available
        chunks = store.list_chunks(station="altona")

        # Select one chunk (station + UTC hour split)
        chunk = store.get_chunk(station="altona", split_hour_utc=9)

        # Open the radar HDF5 file (download into cache if needed)
        with store.open(chunk.radar_file) as radar:
            df0 = load_cycle(radar, k=0)

            # Iterate a sparse subset of cycles ("frames")
            for k in iter_frames(radar, k=0, n=5):
                df = load_cycle(radar, k=k)
                ...

        # Load AIS data as a DataFrame
        ais = store.load_ais_data(chunk.ais_file)

.. _cycle-vs-frame:

Cycle vs Frame
--------------

The radar data is a time-ordered stream of reflection measurements ("cells"), each associated with an azimuth angle (antenna bearing) and a range (distance).
The antenna rotates clockwise; one full rotation takes a few seconds.

In the HDF5 radar files, the raw measurements are stored as a single, growing sequence of cells.
The group ``cycle`` provides two 1D datasets, ``cycle/first`` and ``cycle/last``, that define inclusive index ranges into this cell stream.
A *cycle* is intended to represent *one full antenna rotation*, but there is no canonical choice for where a rotation *starts* (and thus ends).
Instead, the dataset defines cycles as a **sliding window**: for (almost) every azimuth step, it gives you the window of cells you need to accumulate to cover approximately one full turn.

As a consequence, **consecutive cycles overlap heavily**: two adjacent entries in ``cycle/first`` typically differ by roughly one azimuth step, so almost all cells are shared between adjacent cycles (except for the small part that enters/leaves the sliding window).

For many tasks (annotation, *one image per rotation*, downsampling in time) this dense, overlapping representation is inconvenient.
Therefore, ``haxr`` introduces the notion of a **frame**:

- A *frame* is still "one full rotation", but chosen such that adjacent frames do **not** overlap in the cell stream.
- Given a cycle index ``k``, the next frame is defined as the first cycle index ``k_next`` with ``cycle/first[k_next] > cycle/last[k]`` (strictly forward in time).

In other words: **cycles** and **frames** both cover about one full rotation, but cycles advance by ~one azimuth step (almost complete overlap), whereas frames advance by ~one full rotation (no overlap).

Utilities provided for this:

- :func:`~haxr.utilities.iter_frames` yields the cycle indices of successive frames.
- :func:`~haxr.utilities.load_frames` loads several frames and adds a ``frame`` column.

API at a glance
---------------

Dataset release endpoints
~~~~~~~~~~~~~~~~~~~~~~~~~

- :class:`~haxr.doi.DOI` is an :class:`enum.StrEnum` of known dataset release endpoints
  (usable as ``Store(base_url=...)``).

Cache + access layer
~~~~~~~~~~~~~~~~~~~~

- :class:`~haxr.store.Store` manages the local cache and provides:

  - :meth:`~haxr.store.Store.list_chunks` / :meth:`~haxr.store.Store.get_chunk`
  - :meth:`~haxr.store.Store.ensure` / :meth:`~haxr.store.Store.ensure_file`
  - :meth:`~haxr.store.Store.open` (yields :class:`h5py.File` for ``.hdf5``)
  - :meth:`~haxr.store.Store.load_ais_data` (loads CSV data in a :class:`pandas.DataFrame`)
  - :attr:`~haxr.store.Store.stations` (station metadata table)

Radar stream utilities
~~~~~~~~~~~~~~~~~~~~~~

- :func:`~haxr.utilities.load_cycle` loads one cycle into a :class:`pandas.DataFrame`
  (including derived mid-point columns for azimuth/range).
- :func:`~haxr.utilities.iter_frames`, :func:`~haxr.utilities.load_frames` help you
  work with non-overlapping "frames".

Visualization helpers
~~~~~~~~~~~~~~~~~~~~~

- :func:`~haxr.utilities.fill_histogram` builds a 2D histogram in polar coordinates
  (azimuth, range), with optional weights (e.g., amplitude values).
- :func:`~haxr.utilities.histogram_to_cartesian_meshgrid` converts that polar histogram
  to Cartesian meshgrids suitable for plotting (e.g., with :meth:`matplotlib.axes.Axes.pcolormesh`).

Notes on reproducibility
------------------------

:class:`~haxr.store.Store` is a convenience wrapper around cached files on disk; it is
**not** a dataset version manager. If you care about reproducibility, pin the release
endpoint (e.g. a versioned :class:`~haxr.doi.DOI` member) and avoid mixing different
releases in the same cache directory.
