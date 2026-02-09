# Keep type hints unevaluated so Sphinx renders/link public aliases (needed for
# h5py.File) instead of private impl paths.
from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TypeAlias

import h5py
import numpy as np
import pandas as pd

H5File: TypeAlias = h5py.File

Float1DLike: TypeAlias = (
    Sequence[float]
    | np.ndarray[tuple[int], np.dtype[np.float32]]
    | np.ndarray[tuple[int], np.dtype[np.float32]]
    | np.ndarray[tuple[int], np.dtype[np.float64]]
    | np.ndarray[tuple[int], np.dtype[np.float64]]
)
Float1D: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float64]]
Float2D: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.float64]]


def load_cycle(data: H5File, k: int) -> pd.DataFrame:
    """Load one radar cycle from an open HDF5 radar file.

    The cycle boundaries are stored as inclusive cell-index ranges in ``cycle/first``
    and ``cycle/last``. This function slices the per-cell datasets (``tod``, ``az1``,
    ``az2``, ``r1``, ``r2``, ``amp``) for cycle ``k`` and returns them as a DataFrame
    indexed by the original cell indices.

    Column names are annotated with units if the underlying dataset defines an HDF5
    attribute ``unit`` (e.g. ``tod (second)``). Two derived columns are added: ``az`` is
    the circular mean of ``az1`` and ``az2`` (degrees, rounded to 3 decimals) and ``r``
    is the mid-range between ``r1`` and ``r2`` (rounded to 2 decimals). The derived
    column names use the units from ``az1`` and ``r1``. For the distinction between
    *cycles* and *frames*, see :ref:`Cycle vs frame <cycle-vs-frame>`. To iterate
    frames, use :func:`~haxr.utilities.iter_frames` and
    :func:`~haxr.utilities.load_frames`.

    Args:
        data: Open :class:`h5py.File` containing ``cycle/first``, ``cycle/last``,
            ``tod``, ``az1``, ``az2``, ``r1``, ``r2``, and ``amp``.
        k: Cycle index into ``cycle/first`` and ``cycle/last``.

    Returns:
        A :class:`pandas:pandas.DataFrame` containing the raw per-cell fields plus the
        derived columns ``az`` and ``r`` for cycle ``k``.

    Raises:
        IndexError: If ``k`` is out of bounds for ``cycle/first`` or ``cycle/last``.
        KeyError: If required datasets are missing, or if ``az1`` or ``r1`` lack the
            ``unit`` attribute needed to name the derived columns.

    Example:
        .. code-block:: python

            from haxr import Store
            from haxr.utilities import load_cycle

            with Store(base_url=..., cache_dir=...) as store:
                chunk = store.get_chunk(...)
                with store.open(chunk.radar_file) as f:
                    df = load_cycle(f, 0)
    """
    i = data["cycle/first"][k]
    j = data["cycle/last"][k] + 1

    cols = dict()
    for v in ["tod", "az1", "az2", "r1", "r2", "amp"]:
        unit = data[v].attrs.get("unit")
        cols[f"{v} ({unit})" if unit else v] = data[v][i:j]

    az1 = np.deg2rad(data["az1"][i:j])
    az2 = np.deg2rad(data["az2"][i:j])
    az = np.atan2(np.sin(az1) + np.sin(az2), np.cos(az1) + np.cos(az2))
    az = np.rad2deg(az)
    az = np.where(az < 0, az + 360, az)
    cols["az"] = np.round(az, 3)

    cols["r"] = np.round((data["r1"][i:j] + data["r2"][i:j]) / 2, 2)

    mapper = {
        v: f"{v} ({unit})"
        for v in ["az1", "az2", "r1", "r2", "amp"]
        if data[v].attrs.get("unit")
    }
    mapper["az"] = f"az ({data['az1'].attrs['unit']})"
    mapper["r"] = f"r ({data['r1'].attrs['unit']})"

    return pd.DataFrame(cols, index=range(i, j)).rename(columns=mapper)


def load_frames(data: H5File, k: int, *, n: int = 1) -> tuple[pd.DataFrame, int]:
    """Load up to ``n`` frames starting at cycle index ``k``.

    A frame is a sparse, non-overlapping subset of cycles as produced by
    :func:`~haxr.utilities.iter_frames` (see ref:`Cycle vs frame <cycle-vs-frame>`).
    This function loads the frames via :func:`~haxr.utilities.load_cycle`, concatenates
    them into a single DataFrame, and adds a 1-based ``frame`` column (``1..m``)
    identifying which loaded frame each row belongs to.

    Args:
        data: Open :class:`h5py.File` containing the cycle datasets required by
            :func:`~haxr.utilities.load_cycle`.
        k: Start cycle index (interpreted as the first frame).
        n: Maximum number of frames to load (must be ``>= 1``).

    Returns:
        A tuple ``(df, m)`` where ``df`` is the concatenated DataFrame and ``m`` is the
        number of frames actually loaded (``m <= n``). The returned DataFrame contains
        all columns from :func:`~haxr.utilities.load_cycle` plus the ``frame`` column.

    Raises:
        ValueError: If no frames are loaded (e.g. ``k`` is out of range), because
            :func:`pandas.concat` has no objects to concatenate.
        IndexError: If ``k`` (or a subsequent frame index) is out of bounds.
        KeyError: If required datasets are missing from ``data``.

    Example:
        .. code-block:: python

            from haxr import Store
            from haxr.utilities import load_frames

            with Store(base_url=..., cache_dir=...) as store:
                chunk = store.get_chunk(...)
                with store.open(chunk.radar_file) as f:

                    # load 5 frames starting at cycle `k = 100`
                    df, m = load_frames(f, k=100, n=5)

                    first_frame = df[df["frame"] == 1]
                    last_frame  = df[df["frame"] == m]
                    ...
    """
    dfs = [load_cycle(data, i) for i in iter_frames(data, k, n=n)]

    for i, df in enumerate(dfs, start=1):
        df["frame"] = i

    return pd.concat(dfs), len(dfs)


def arg_next_frame(data: H5File, k: int) -> int | None:
    """Return the index of the next frame after frame ``k``.

    The radar stream is stored as overlapping cycles with inclusive cell-index bounds
    in ``cycle/first`` and ``cycle/last``. A frame is defined here as the next cycle
    that starts strictly after the current cycle ends. This helper therefore returns
    the first cycle index whose start index is greater than ``cycle/last[k]``.

    See :ref:`Cycle vs Frame <cycle-vs-frame>` for the underlying definition. This is a
    low-level primitive used by :func:`~haxr.utilities.iter_frames` and
    :func:`~haxr.utilities.load_frames`.

    Args:
        data: Open :class:`h5py.File` containing the datasets ``cycle/first`` and
            ``cycle/last``.
        k: Current cycle index (interpreted as the current frame).

    Returns:
        The integer index of the next frame, or ``None`` if no cycle starts after
        ``cycle/last[k]``.

    Raises:
        IndexError: If ``k`` is out of bounds for ``cycle/last``.
        KeyError: If ``cycle/first`` or ``cycle/last`` is missing from ``data``.
    """
    first = data["cycle/first"]
    last = data["cycle/last"]

    idx = np.searchsorted(first, last[k], side="right").item()

    return idx if idx < first.size else None


def iter_frames(data: H5File, k: int, n: int | None = None) -> Iterator[int]:
    """Iterate cycle indices starting at ``k``, jumping by frames.

    Consecutive radar cycles overlap heavily. This iterator yields a sparse subset of
    cycle indices by repeatedly applying :func:`~haxr.utilities.arg_next_frame`, i.e.,
    it advances to the first cycle whose start index is strictly greater than the end
    index of the previously yielded cycle.

    See :ref:`Cycle vs Frame <cycle-vs-frame>` for why adjacent cycles overlap and how
    frames are selected.

    Args:
        data: Open :class:`h5py.File` containing ``cycle/first`` and ``cycle/last``.
        k: Start cycle index.
        n: Maximum number of indices to yield (including ``k``). If ``None`` (default),
            iterate until no next frame exists.

    Yields:
        Cycle indices ``k, k2, k3, ...`` corresponding to successive frames.

    Raises:
        KeyError: If ``cycle/first`` or ``cycle/last`` is missing from ``data``.
        IndexError: If ``k`` (or a subsequent index) is out of bounds.

    Example:
        .. code-block:: python

            from haxr import Store
            from haxr.utilities import iter_frames, load_cycle

            with Store(base_url=..., cache_dir=...) as store:
                chunk = store.get_chunk(...)
                with store.open(chunk.radar_file) as f:

                    # iterate 5 frames starting at cycle `k = 100`
                    for k in iter_frames(f, k=100, n=5):
                        df = load_cycle(f, k)
                        ...

        For loading several frames into one DataFrame, use
        :func:`~haxr.utilities.load_frames`.
    """
    if k < data["cycle/first"].size:
        yield k

    if n is None:
        n = data["cycle/last"].size - k

    maybe_k = k
    for _ in range(n - 1):
        maybe_k = arg_next_frame(data, maybe_k)
        if maybe_k is not None:
            yield maybe_k
        else:
            break


def fill_histogram(
    *,
    az: Float1DLike,
    r: Float1DLike,
    weights: Float1DLike | None = None,
    az_edges: Float1DLike | None = None,
    r_edges: Float1DLike | None = None,
) -> tuple[Float2D, Float1D, Float1D]:
    """Fill a 2D histogram in polar coordinates (azimuth, range).

    This is a thin wrapper around :func:`numpy.histogram2d`. If ``weights`` is provided,
    it is passed through as the per-sample weight array; a canonical choice is the per
    cell amplitude value. If ``weights`` is ``None``, the histogram contains counts.

    If ``az_edges`` or ``r_edges`` is ``None``, bin edges are inferred from the data.
    The helper :func:`~haxr.utilities.infer_az_edges` returns a schema compatible with
    :func:`numpy.linspace` (keys ``start``, ``stop``, ``num``), and
    :func:`~haxr.utilities.infer_r_edges` returns a schema compatible with
    :func:`numpy.arange` (keys ``start``, ``step``). These schemas are used to construct
    ``az_edges`` and ``r_edges`` for this function.

    Args:
        az: Azimuth samples (degrees).
        r: Range samples.
        weights: Optional per-sample weights (same length as ``az`` and ``r``).
        az_edges: Optional azimuth bin edges. If ``None``, inferred via
            :func:`~haxr.utilities.infer_az_edges`.
        r_edges: Optional range bin edges. If ``None``, inferred via
            :func:`~haxr.utilities.infer_r_edges`.

    Returns:
        A tuple ``(hist, az_edges, r_edges)`` as returned by :func:`numpy.histogram2d`,
        where ``hist.shape == (len(az_edges) - 1, len(r_edges) - 1)``.

    Raises:
        ValueError: If the provided or inferred bins are invalid for
            :func:`numpy.histogram2d`.
        KeyError: If inferred edge schemas are empty or missing required keys.
    """
    if az_edges is None:
        pattern = infer_az_edges(az=az)

        start: float = pattern["start"]
        num: int = int(pattern["num"])
        stop: float = pattern["stop"]
        az_edges = np.linspace(start=start, stop=stop, num=num, dtype=np.float64)

    if r_edges is None:
        pattern = infer_r_edges(r=r)

        start: float = pattern["start"]
        step: float = pattern["step"]
        stop: float = step + float(np.max(r))
        r_edges = np.arange(start, stop=stop, step=step, dtype=np.float64)

    az_flat = np.asarray(az, dtype=np.float64).ravel()
    r_flat = np.asarray(r, dtype=np.float64).ravel()
    weights_flat = (
        None if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    )

    az_edges_flat = np.asarray(az_edges, dtype=np.float64).ravel()
    r_edges_flat = np.asarray(r_edges, dtype=np.float64).ravel()

    return np.histogram2d(
        az_flat, r_flat, weights=weights_flat, bins=(az_edges_flat, r_edges_flat)
    )


def infer_r_edges(*, r: Float1DLike) -> dict[str, float]:
    """Infer a range-bin schema for histogramming.

    The returned schema is meant to be used with :func:`numpy.arange` to construct
    range bin edges. It contains ``start`` and ``step`` only (no ``stop``), because
    the appropriate upper bound depends on the use case. A common choice is
    ``stop = float(np.max(r)) + step`` (as done in
    :func:`~haxr.utilities.fill_histogram`).

    The step size is estimated as the median of ``np.diff(r)`` and the first edge is
    placed half a step before the minimum value.

    Warning:
        This function uses heuristics; the inferred edges may be incorrect for
        irregularly sampled or noisy inputs. If the inferred edges look unreasonable,
        prefer computing them explicitly.

    Args:
        r: Range samples with approximately constant spacing.

    Returns:
        A dict with keys ``start`` and ``step`` suitable for::

            schema = infer_r_edges(r=r)
            stop = float(np.max(r)) + schema["step"]
            r_edges = numpy.arange(stop=stop, **schema)

    Raises:
        ValueError: If ``r`` is empty or cannot be reduced (e.g. ``min`` / ``diff``
            fails).
    """
    r = np.asarray(r)

    r0 = r.min()
    dr = np.median(np.diff(r)).item()

    return {"start": r0 - dr / 2, "step": dr}


def infer_az_edges(*, az: Float1DLike) -> dict[str, float | int]:
    """Infer an azimuth-bin schema for histogramming.

    The returned schema is meant to be used with :func:`numpy.linspace` to construct
    azimuth bin edges that cover one full turn. The edges are chosen such that the
    interval ``[0, 360]`` is fully covered and a wrap-around bin spanning the
    discontinuity at 360/0 is represented naturally (e.g., a bin ``355 .. 5`` degrees).

    This helper is primarily intended for :func:`~haxr.utilities.fill_histogram`.
    If inference fails (e.g., the azimuth step size cannot be determined reliably),
    an empty dict is returned.

    Warning:
        This function uses heuristics; the inferred edges may be incorrect for
        irregularly sampled or noisy inputs. If the inferred edges look unreasonable,
        prefer computing them explicitly.

    Args:
        az: 1D azimuth samples in degrees, typically in acquisition order.

    Returns:
        A schema dict with keys ``start``, ``stop``, and ``num`` suitable for::

            schema = infer_az_edges(az=az)
            az_edges = numpy.linspace(**schema)

        The returned ``start`` may be negative and ``stop`` may be greater than 360 to
        ensure wrap-around coverage.

    Raises:
        ValueError: If ``az`` is empty or cannot be reduced (e.g. ``max`` / ``diff``
            fails).
    """
    az = np.asarray(az)

    az0 = az[np.argmax(az)]
    daz = np.diff(az)

    daz = daz[(daz > 0.01) & (daz < 10.0)]
    daz, c = np.unique(daz, return_counts=True)
    if daz.size < 1:
        return dict()

    frac = daz / daz[np.argmax(c)]
    sel = (frac > 0.9) & (frac < 1.1)
    daz = np.average(daz[sel], weights=c[sel])

    az_num = round(360.0 / daz)
    daz = 360.0 / az_num

    az_start = (az0 + np.sign(180 - az0) * daz / 2) % daz - daz
    az_end = az_start + np.ceil((360 - az_start) / daz) * daz

    if az_end - daz > 360:
        az_end -= daz

    assert az_start <= 0, az_start
    assert az_end >= 360, az_end

    return {
        "start": az_start.item(),
        "num": round((az_end - az_start) / daz) + 1,
        "stop": az_end.item(),
    }


def histogram_to_cartesian_meshgrid(
    hist: Float2D,
    az_edges: Float1DLike,
    r_edges: Float1DLike,
) -> tuple[Float2D, Float2D, Float2D]:
    """Convert a polar 2D histogram into a Cartesian meshgrid for plotting.

    Given histogram values binned in polar coordinates (azimuth, range) together with
    their bin edges, this function computes Cartesian coordinate grids ``x`` and ``y``
    for the bin corners. The result is suitable for visualizing the polar histogram in
    a Cartesian plot (e.g., via :meth:`matplotlib.axes.Axes.pcolormesh`).

    Azimuth edges are interpreted as degrees and converted to radians. The Cartesian
    mapping follows the usual radar convention (azimuth measured from North, increasing
    clockwise): ``x = r * sin(az)`` (East) and ``y = r * cos(az)`` (North).

    Typically, ``hist``, ``az_edges`` and ``r_edges`` come from
    :func:`~haxr.utilities.fill_histogram`.

    Args:
        hist: 2D histogram values binned over azimuth and range.
        az_edges: 1D azimuth bin edges in degrees.
        r_edges: 1D range bin edges.

    Returns:
        A tuple ``(x, y, values)`` where ``x`` and ``y`` are 2D arrays of bin-corner
        coordinates with shape ``(len(az_edges), len(r_edges))`` and ``values`` are the
        corresponding per-bin histogram values suitable for plotting against these
        edges.

    Example:
        .. code-block:: python

            hist, az_edges, r_edges = fill_histogram(az=az, r=r, weights=amp)
            x, y, values = histogram_to_cartesian_meshgrid(hist, az_edges, r_edges)

            fig, ax = plt.subplots()
            ax.pcolormesh(x, y, values, shading="auto")
            ax.set_aspect("equal", adjustable="box")
    """
    az_edges_flat = np.asarray(az_edges, dtype=np.float64).ravel()
    r_edges_flat = np.asarray(r_edges, dtype=np.float64).ravel()
    r, az = np.meshgrid(r_edges_flat, np.deg2rad(az_edges_flat), indexing="xy")

    return r * np.sin(az), r * np.cos(az), hist
