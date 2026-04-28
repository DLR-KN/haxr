import enum


class DOI(enum.StrEnum):
    """Named dataset release endpoints usable as a ``Store`` base URL.

    ``DOI`` is a :class:`enum.StrEnum`, so its members behave like strings and can be
    passed directly where a URL is expected (e.g., as ``base_url`` for
    :class:`~haxr.store.Store`).

    The values are URL prefixes pointing at a release's file endpoint. The store
    appends file names like ``manifest.json``, ``stations.csv``, and per-chunk files to
    this prefix when downloading.

    The version numbers in member names refer to HAXR dataset releases published on
    Zenodo, not to versions of the ``haxr`` Python package. For example,
    ``DOI.v1_1_0`` points to the Zenodo file endpoint for dataset release v1.1.0,
    regardless of the installed package version. When in doubt, trim the ``/files``
    suffix and print the URL to see the corresponding Zenodo record:

    .. code-block:: python

        from haxr import DOI

        print(str(DOI.v1_1_0).removesuffix("/files"))
        # https://zenodo.org/records/19824555

    Prefer a versioned member (e.g. ``DOI.v1_1_0``) for reproducible workflows. Use
    ``DOI.latest`` to follow the most recent supported release.

    Attributes:
        v1_0_0: Base URL prefix for dataset release v1.0.0.
        v1_1_0: Base URL prefix for dataset release v1.1.0.
        latest: Alias for the most recent supported release.
    """

    v1_0_0 = "https://zenodo.org/records/18759623/files"
    v1_1_0 = "https://zenodo.org/records/19824555/files"
    latest = v1_1_0
