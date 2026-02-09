import enum


class DOI(enum.StrEnum):
    """Named dataset release endpoints usable as a ``Store`` base URL.

    ``DOI`` is a :class:`enum.StrEnum`, so its members behave like strings and can be
    passed directly where a URL is expected (e.g., as ``base_url`` for
    :class:`~haxr.store.Store`).

    The values are URL prefixes pointing at a release's file endpoint. The store
    appends file names like ``manifest.json``, ``stations.csv``, and per-chunk files to
    this prefix when downloading.

    Prefer a versioned member (e.g. ``DOI.v1``) for reproducible workflows. Use
    ``DOI.latest`` to follow the most recent supported release.

    Attributes:
        v1: Base URL prefix for dataset release v1.
        latest: Alias for the most recent supported release.
    """

    v1 = "https://zenodo.org/records/18759623/files"
    latest = v1
