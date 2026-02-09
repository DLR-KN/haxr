# Keep type hints unevaluated so Sphinx renders/link public aliases (needed for
# h5py.File) instead of private impl paths.
from __future__ import annotations

import dataclasses
import hashlib
import json
import numbers
import tempfile
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import IO, Any, Self, TypeAlias

import h5py
import pandas as pd
from tqdm import tqdm

from .doi import DOI

H5File: TypeAlias = h5py.File


def _download(
    base_url: str | None,
    file: str,
    *,
    dest: Path,
    sha256: str | None = None,
    show_progress: bool = False,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    try:
        if base_url is None or base_url == "":
            raise ValueError(
                f"Cannot download '{file}' into '{str(dest)}'. No base URL was "
                "provided (base_url is `None`/empty). Provide `base_url` or use a "
                " populated `cache_dir` / disable downloads via `allow_download=False`."
            )

        url = f"{base_url}/{file}"
        with urllib.request.urlopen(url) as r, part.open("wb") as f:
            total: int | None = None
            content_length = r.headers.get("Content-Length")
            if content_length is not None:
                with suppress(ValueError):
                    n = int(content_length)
                    if n > 0:
                        total = n

            with tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"Downloading {url} to {dest.parent}",
                disable=not show_progress,
            ) as pbar:
                for chunk in iter(lambda: r.read(1024 * 1024), b""):
                    f.write(chunk)
                    pbar.update(len(chunk))

        part.replace(dest)
    finally:
        if part.exists():
            with suppress(OSError):
                part.unlink()

    if sha256:
        _verify_sha256(dest, sha256)


def _verify_sha256(path: Path, expected_hex: str) -> None:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    got = h.hexdigest()
    if got.lower() != expected_hex.lower():
        raise OSError(
            f"SHA256 mismatch for {path.name}: expected {expected_hex}, got {got}"
        )


@dataclasses.dataclass(frozen=True, slots=True)
class File:
    """A reference to a file managed by the store.

    This is a small value object that identifies a file via its cache path and
    (optionally) an integrity checksum. It may refer to an already-cached file or a file
    that can be fetched into the cache by the store.

    Attributes:
        path: Absolute path to the file location in the store's cache.
        sha256: Optional expected SHA256 hex digest for the file contents. If provided,
            the store can use it to verify the file's integrity.
    """

    path: Path
    sha256: str | None


@dataclasses.dataclass(frozen=True, slots=True)
class Chunk:
    """A logical unit of data for a station and a specific UTC hour split.

    A ``Chunk`` bundles the files that belong together for one station and one
    hour-of-day partition. The store returns ``Chunk`` objects so callers can locate,
    fetch, and open the underlying data files.

    Attributes:
        station: Station identifier.
        split_hour_utc: Hour of day in UTC (0..23) used to partition the data.
        radar_file: Radar data file for this chunk.
        ais_file: AIS data file for this chunk.
    """

    station: str
    split_hour_utc: int
    radar_file: File
    ais_file: File


class Manifest:
    def __init__(
        self, *, base_url: str | None, cache_dir: Path, allow_download: bool
    ) -> None:
        manifest_path = cache_dir / "manifest.json"
        if not manifest_path.is_file():
            if not allow_download:
                raise FileNotFoundError(
                    "Manifest not in cache and downloads disabled: "
                    f"{manifest_path.name}"
                )
            _download(
                base_url, "manifest.json", dest=manifest_path, show_progress=False
            )

        self._manifest = json.load(manifest_path.open())

        self._meta = self._parse_meta_data(self._manifest, cache_dir=cache_dir)
        self._chunks = [
            self._parse_chunk(chunk, cache_dir=cache_dir)
            for chunk in self._manifest.get("chunks", [])
        ]

    @staticmethod
    def _parse_meta_data(raw, *, cache_dir: Path) -> File:
        if not isinstance(raw, dict) or "stations" not in raw:
            raise ValueError("manifest.json must contain 'stations' as an object.")

        path = raw["stations"].get("name")
        sha256 = raw["stations"].get("sha256")

        if not isinstance(path, str) or not path:
            raise ValueError(
                "'stations' entry must contain non-empty string field 'name'."
            )

        return File(path=cache_dir / path, sha256=sha256)

    @staticmethod
    def _parse_chunk(raw, *, cache_dir: Path) -> Chunk:
        if not isinstance(raw, dict):
            raise ValueError("Invalid chunk")

        station = raw.get("station")
        hour = raw.get("split_hour_utc")
        radar = raw.get("radar")
        ais = raw.get("ais")

        if not isinstance(station, str) or not station:
            raise ValueError("Each chunk must have non-empty string field 'station'.")

        if not isinstance(hour, int) or not (0 <= hour <= 23):
            raise ValueError(
                "Each chunk must have integer field 'split_hour_utc' in 0..23."
            )

        if not isinstance(radar, dict) or not radar:
            raise ValueError("Each chunk must contain 'radar' as an object.")

        if not isinstance(ais, dict) or not ais:
            raise ValueError("Each chunk must contain 'ais' as an object.")

        radar_path = radar.get("name")
        radar_sha256 = radar.get("sha256")
        ais_path = ais.get("name")
        ais_sha256 = ais.get("sha256")

        if not isinstance(radar_path, str) or not radar_path:
            raise ValueError(
                "Each radar object must have a non-empty string field 'name'."
            )

        if radar_sha256 is not None and (
            not isinstance(radar_sha256, str) or not radar_sha256
        ):
            raise ValueError(
                "Each radar object must have a non-empty string field 'sha256'."
            )

        if not isinstance(ais_path, str) or not ais_path:
            raise ValueError(
                "Each AIS object must have a non-empty string field 'name'."
            )

        if ais_sha256 is not None and (
            not isinstance(ais_sha256, str) or not ais_sha256
        ):
            raise ValueError(
                "Each AIS object must have a non-empty string field 'sha256'."
            )

        return Chunk(
            station=station,
            split_hour_utc=hour,
            radar_file=File(path=cache_dir / radar_path, sha256=radar_sha256),
            ais_file=File(path=cache_dir / ais_path, sha256=ais_sha256),
        )

    @property
    def meta_data(self) -> File:
        return self._meta

    @property
    def chunks(self) -> list[Chunk]:
        return self._chunks


class Store:
    """Local cache and access layer for radar/AIS data and station metadata.

    The store manages an on-disk cache directory and can populate it by downloading
    missing files from ``base_url`` (if enabled). Callers typically discover available
    data via :meth:`list_chunks` / :meth:`get_chunk`, then ensure files are present via
    :meth:`ensure` or access content using :meth:`open`, :meth:`load_ais_data`, and
    :attr:`stations`.

    If ``cache_dir`` is not provided, the store creates a temporary directory and
    deletes it when :meth:`close` is called or when exiting a context manager.

    Warning:
        ``Store`` is a convenience layer around data files on disk. It helps you locate,
        open, and (optionally) lazily download files into a local cache directory. It is
        **not** a dataset version manager.

        Choosing an appropriate ``base_url`` (e.g., a specific :class:`~haxr.doi.DOI`
        release) is entirely the caller's responsibility. The store does not track
        which release a cache directory belongs to, and it does not prevent you from
        mixing files from different releases on disk.

        If you reuse a ``cache_dir`` from an incompatible release (assume releases are
        pairwise incompatible), you might get lucky and notice the problem via SHA256
        checksum mismatches during :meth:`ensure`/download. However, it is also possible
        that no verification is triggered (because files are already present), and you
        end up silently working with a different cached release than the one implied by
        ``base_url``. Note that passing ``base_url=None`` does not make the cache
        version-safe; it only prevents downloads. You must still avoid mixing
        incompatible releases in one cache.

        If you need reproducibility, you must pin ``base_url`` to a versioned release
        and ensure that each release uses its own cache directory (or that old caches
        are cleaned before switching).

    Args:
        base_url: Base URL used to download files into the cache. This may be a plain
            string URL, a :class:`~haxr.doi.DOI` value, or ``None``.

            Passing ``None`` enables an "offline" mode: the store will only use files
            that already exist in ``cache_dir``. If a download is required (e.g.,
            ``manifest.json`` or a chunk file is missing, or a cached file fails SHA256
            verification and a re-download would be needed), the store raises
            :exc:`ValueError`.

            Note that ``manifest.json`` must already be present in ``cache_dir`` when
            using ``base_url=None`` with ``allow_download=True``; otherwise
            initialization will fail when the manifest needs to be fetched.
        cache_dir: Directory used for the local cache. If ``None``, a temporary cache
            directory is created automatically.
        allow_download: If False, operations that require uncached files fail with
            :exc:`FileNotFoundError` instead of downloading.

    Example:
        .. code-block:: python

            from haxr import DOI, Store, load_cycle

            with Store(base_url=DOI.latest) as store:
                chunk = store.get_chunk(station="altona", split_hour_utc=9)
                with store.open(chunk.radar_file) as f:
                    ...

    Notes:
        - :meth:`open` yields an :class:`h5py.File` for ``.hdf5`` files and a normal
            file object for other paths.
        - :attr:`stations` loads and caches station metadata as a
          :class:`pandas:pandas.DataFrame`.
    """

    def __init__(
        self,
        *,
        base_url: str | DOI | None,
        cache_dir: Path | None = None,
        allow_download: bool = True,
    ) -> None:
        self._base_url = None if base_url is None else str(base_url).strip("/")
        self._allow_download = allow_download

        self._tmp = None
        if cache_dir is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="haxr_")
            cache_dir = Path(self._tmp.name)
        else:
            cache_dir.mkdir(parents=True, exist_ok=True)

        self._manifest = Manifest(
            base_url=self._base_url, cache_dir=cache_dir, allow_download=allow_download
        )

        self._stations: pd.DataFrame | None = None
        self._chunks: dict[tuple[str, int], Chunk] = {
            (chunk.station, chunk.split_hour_utc): chunk
            for chunk in self._manifest.chunks
        }

    def close(self) -> None:
        """Close the store and release any owned resources.

        If the store created a temporary cache directory (because ``cache_dir`` was not
        provided), this method deletes that directory and all cached files within it.
        If a user-supplied ``cache_dir`` is used, the directory is left untouched.

        This method is idempotent and may be called multiple times.

        Returns:
            ``None``
        """

        if self._tmp is not None:
            self._tmp.cleanup()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def list_chunks(self, *, station: str | None = None) -> list[Chunk]:
        """List available chunks known to this store.

        The returned list is sorted deterministically by ``(station, split_hour_utc)``
        in ascending order. This method only consults the store's in-memory chunk index
        created during initialization; it does not touch the network and does not
        download data files.

        Note that the returned :class:`Chunk` objects may reference files that are not
        yet present in the local cache directory. Use :meth:`ensure` (or
        :meth:`ensure_file`) to populate missing files.

        Args:
            station: Optional station identifier. If provided, only chunks whose
                ``chunk.station == station`` are returned. If no chunks match, an empty
                list is returned.

        Returns:
            A list of chunks, sorted by ``(station, split_hour_utc)``.
        """
        keys = sorted(
            [(s, t) for (s, t) in self._chunks if not station or s == station]
        )

        return [self._chunks[k] for k in keys]

    def get_chunk(self, *, station: str, split_hour_utc: int | str) -> Chunk:
        """Return the chunk identified by station and UTC split hour.

        This performs a lookup in the store's in-memory chunk index populated during
        initialization. It neither access the network nor it ensures that the chunk's
        files are present in the local cache. Use :meth:`ensure` or :meth:`ensure_file`
        to populate missing files.

        Args:
            station: Station identifier.
            split_hour_utc: Hour of day in UTC (typically 0..23) used to partition the
                dataset. May be given as an ``int`` or as a decimal string convertible
                to ``int`` (e.g., ``"9"`` or ``"09"``)

        Returns:
            The chunk for the given ``(station, split_hour_utc)`` key.

        Raises:
            TypeError: If ``split_hour_utc`` is neither an ``int``-like value nor a
                decimal string.
            ValueError: If ``split_hour_utc`` is a string but cannot be converted to
                ``int``.
            KeyError: If no chunk exists for the given ``(station, split_hour_utc)``.
                The error message includes the requested key.
        """
        split_hour_raw = split_hour_utc
        if isinstance(split_hour_utc, str):
            try:
                split_hour_utc = int(split_hour_utc, 10)
            except ValueError as e:
                raise ValueError(
                    "split_hour_utc must be a decimal string convertible to int; "
                    f"got {split_hour_raw!r}"
                ) from e
        elif isinstance(split_hour_utc, bool) or not isinstance(
            split_hour_utc, numbers.Integral
        ):
            raise TypeError(
                "split_hour_utc must be an int or a decimal string; "
                f"got {type(split_hour_raw).__name__}"
            )
        else:
            split_hour_utc = int(split_hour_utc)

        try:
            return self._chunks[(station, split_hour_utc)]
        except KeyError as e:
            raise KeyError(
                f"No chunk for station={station}, split_hour_utc={split_hour_utc}"
            ) from e

    def ensure_file(self, file: File, show_progress: bool = True) -> Path:
        """Ensure that ``file`` is present in the local cache.

        If the file is already present, it is reused. If ``file.sha256`` is provided,
        the cached file is verified.

        If the file is missing (or verification fails) and downloads are allowed, the
        store downloads the file into the cache.

        Args:
            file: File reference returned by the store.
            show_progress: Whether to show a download progress bar.

        Returns:
            Path to the cached file.

        Raises:
            FileNotFoundError: If the file is not cached and downloads are disabled.
            ValueError: If a download is required but ``base_url`` is ``None``/empty.
            OSError: If the cached file fails SHA256 verification and recovery is not
                possible (e.g., downloads disabled).
        """
        if file.path.is_file():
            if file.sha256 is None:
                return file.path

            try:
                _verify_sha256(file.path, file.sha256)
                return file.path
            except OSError:
                if not self._allow_download:
                    raise

        if not self._allow_download:
            raise FileNotFoundError(
                f"File not in cache and downloads disabled: {file.path}"
            )

        _download(
            self._base_url,
            file.path.name,
            dest=file.path,
            sha256=file.sha256,
            show_progress=show_progress,
        )
        return file.path

    def ensure(self, chunk: Chunk, show_progress: bool = True) -> Chunk:
        """Ensure that all files for a chunk are available in the local cache.

        This is a convenience wrapper around :meth:`ensure_file`. The given `chunk` is
        first normalized by resolving it through :meth:`get_chunk`, ensuring that the
        store's canonical metadata (paths, optional checksums) is used.

        The chunk's radar and AIS files are then ensured to exist on disk. If a file is
        already cached, it is reused. If a SHA256 checksum is available, the cached
        file may be verified; a mismatch can trigger a re-download when downloads are
        enabled.

        Args:
            chunk: Chunk identifying the data to ensure. The
                ``(station, split_hour_utc)`` fields are used to resolve the canonical
                chunk via :meth:`get_chunk`.
            show_progress: Whether to show a progress bar while downloading.

        Returns:
            The canonical chunk instance as returned by :meth:`get_chunk`.

        Raises:
            KeyError: If the chunk's ``(station, split_hour_utc)`` does not exist in
                this store.
            FileNotFoundError: If a required file is missing from the cache and
                downloads are disabled.
            ValueError: If a download is required but no ``base_url`` was provided.
            OSError: If a cached file fails SHA256 verification and cannot be recovered
                (e.g., downloads disabled).
        """
        chunk = self.get_chunk(
            station=chunk.station, split_hour_utc=chunk.split_hour_utc
        )

        for f in [
            f for f in [chunk.radar_file, chunk.ais_file] if not f.path.is_file()
        ]:
            self.ensure_file(f, show_progress=show_progress)

        return chunk

    @contextmanager
    def open(
        self, file: File, mode: str = "r", show_progress: bool = False, **kwargs
    ) -> Iterator[H5File | IO[Any]]:
        """Open a cached file and yield a file handle.

        This is a context manager. It first ensures that ``file`` is present in the
        local cache via :meth:`ensure_file` (downloading if necessary, subject to
        ``allow_download`` and ``base_url``). The file is then opened and automatically
        closed when leaving the context.

        For files ending in ``.hdf5``, this uses :class:`h5py.File`. For all other
        paths, it uses Python's built-in :func:`open`.

        Args:
            file: File reference to open.
            mode: File mode passed to :class:`h5py.File` or :func:`open`.
            show_progress: Whether to show a progress bar while downloading.
            **kwargs: Additional keyword arguments forwarded to :class:`h5py.File` or
                :func:`open` (depending on the file type).

        Yields:
            An open file handle. This is an :class:`h5py.File` for ``.hdf5`` files,
            otherwise a regular Python file object.

        Notes:
            Sphinx currently renders the return type in the generated HTML signature
            incorrectly as ``TypeAliasForwardRef('h5py.File')`` instead of
            :class:`h5py.File`. This is a known upstream issue in Sphinx. See
            https://github.com/sphinx-doc/sphinx/issues/14003

        Raises:
            FileNotFoundError: If the file is missing from the cache and downloads are
                disabled.
            ValueError: If a download is required but no ``base_url`` was provided.
            OSError: If a cached file fails SHA256 verification, or if opening/reading
                the file fails (including errors raised by :class:`h5py.File`).
        """
        path = self.ensure_file(file, show_progress=show_progress)

        f = (
            h5py.File(path, mode, **kwargs)
            if path.suffix == ".hdf5"
            else open(path, mode, **kwargs)  # noqa: SIM115
        )
        try:
            yield f
        finally:
            f.close()

    def load_ais_data(
        self,
        file: File,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Load AIS data from a cached CSV file.

        This is a convenience wrapper around :meth:`ensure_file` and
        :func:`pandas:pandas.read_csv`. The file is ensured to exist in the local cache
        (downloading if enabled) and is then read as a CSV. Keyword arguments are
        forwarded to :func:`pandas:pandas.read_csv` to customize parsing.

        Args:
            file: File reference to an AIS CSV (see :class:`Chunk`, attribute
                ``ais_file``).
            show_progress: Whether to show a progress bar while downloading.
            **kwargs: Keyword arguments forwarded to :func:`pandas:pandas.read_csv`.
                Use these to override pandas defaults.

        Returns:
            A :class:`pandas:pandas.DataFrame` containing the AIS records from the CSV.

        Raises:
            FileNotFoundError: If the file is not cached and downloads are disabled.
            ValueError: If a download is required but no ``base_url`` was provided.
            OSError: If SHA256 verification fails (when provided) or I/O fails.
            TypeError: If invalid keyword arguments are passed to
                :func:`pandas:pandas.read_csv`.
            pandas.errors.ParserError: If pandas fails to parse the CSV.
        """
        path = self.ensure_file(file, show_progress=show_progress)
        return pd.read_csv(path, **kwargs)

    @property
    def stations(
        self,
    ) -> pd.DataFrame:
        """Station metadata table.

        Lazily loads the dataset's station metadata (`stations.csv`), caches it on the
        :class:`Store` instance, and returns a copy of the cached
        :class:`pandas:pandas.DataFrame` on each access. Callers may freely mutate the
        returned DataFrame without affecting the cache.

        Returns:
            A copy of the cached :class:`pandas:pandas.DataFrame` containing station
            metadata.

        Raises:
            FileNotFoundError: If the metadata file is not cached and downloads are
                disabled.
            ValueError: If a download is required but no ``base_url`` was provided.
            OSError: If SHA256 verification fails (when provided) or I/O fails.
            pandas.errors.ParserError: If :func:`pandas:pandas.read_csv` fails to parse
                the CSV.
        """
        if self._stations is None:
            path = self.ensure_file(self._manifest.meta_data, show_progress=False)
            self._stations = pd.read_csv(path)

        return self._stations.copy()
