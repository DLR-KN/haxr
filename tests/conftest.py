import contextlib
import dataclasses
import hashlib
import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pytest

from haxr.store import Chunk, File


@contextlib.contextmanager
def _serve_directory(root: Path):
    handler = partial(SimpleHTTPRequestHandler, directory=str(root))

    host = "127.0.0.1"
    httpd = ThreadingHTTPServer((host, 0), handler)
    port = httpd.server_port

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


def _circular_mean(x, y):
    x = np.deg2rad(x)
    y = np.deg2rad(y)
    m = np.arctan2(np.sin(x) + np.sin(y), np.cos(x) + np.cos(y))
    m = np.rad2deg(m)

    return np.where(m < 0, m + 360.0, m)


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


@dataclasses.dataclass(slots=True)
class DummyDataset:
    chunks: list[Chunk]
    data_dir: Path
    truth: dict[str, Any]


@pytest.fixture
def dummy_dataset(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)

    pd.DataFrame(dict(station=["station1", "station2"], x=[1, 2], y=[3, 4])).to_csv(
        data_dir / "stations.csv", index=False
    )

    cycle_first = np.array([0, 2], dtype=np.uint32)
    cycle_last = np.array([5, 6], dtype=np.uint32)

    tod = np.array(
        [1000.0, 1001.0, 1002.0, 1003.0, 1004.0, 1005.0, 1006.0],
        dtype=np.float32,
    )

    az1 = np.array([10.0, 170.0, 355.0, 80.0, 200.0, 300.0, 40.0], dtype=np.float32)
    az2 = np.array([20.0, 190.0, 5.0, 100.0, 220.0, 320.0, 60.0], dtype=np.float32)

    r1 = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=np.float32)
    r2 = np.array([5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0], dtype=np.float32)

    amp = np.array([11, 12, 13, 21, 22, 23, 24], dtype=np.uint8)

    truth = {
        "cycle_first": cycle_first,
        "cycle_last": cycle_last,
        "arrays": {
            "tod": tod,
            "az1": az1,
            "az2": az2,
            "r1": r1,
            "r2": r2,
            "amp": amp,
        },
        "derived": {
            "az": _circular_mean(az1, az2),
            "r": (r1 + r2) / 2.0,
        },
        "column_names": {
            "tod": "tod (s)",
            "az1": "az1 (deg)",
            "az2": "az2 (deg)",
            "r1": "r1 (m)",
            "r2": "r2 (m)",
            "amp": "amp",
            "az": "az (deg)",
            "r": "r (m)",
        },
    }

    chunks: list[Chunk] = []
    for i, (station, hour) in enumerate(
        zip(["station1", "station1", "station2"], ["08", "09", "08"], strict=True)
    ):
        radar_file = data_dir / f"{station}_{hour}-UTC.hdf5"
        with h5py.File(radar_file, "w") as f:
            cycles = f.create_group("cycle")
            cycles.create_dataset("first", data=cycle_first)
            cycles.create_dataset("last", data=cycle_last)

            f.create_dataset("tod", data=tod)
            f.create_dataset("az1", data=az1)
            f.create_dataset("az2", data=az2)
            f.create_dataset("r1", data=r1)
            f.create_dataset("r2", data=r2)
            f.create_dataset("amp", data=amp)

            f["tod"].attrs["unit"] = "s"
            f["az1"].attrs["unit"] = "deg"
            f["az2"].attrs["unit"] = "deg"
            f["r1"].attrs["unit"] = "m"
            f["r2"].attrs["unit"] = "m"

        ais_file = data_dir / f"{station}_{hour}-UTC.csv"
        pd.DataFrame(
            dict(
                alias=["foo", "bar"],
                r=[100 + 10 * i, 200 + 10 * i],
                az=[10 + 5 * i, 20 + 5 * i],
            )
        ).to_csv(ais_file, index=False)

        chunks.append(
            Chunk(
                station=station,
                split_hour_utc=int(hour),
                radar_file=File(path=radar_file, sha256=_compute_sha256(radar_file)),
                ais_file=File(path=ais_file, sha256=_compute_sha256(ais_file)),
            )
        )

    manifest = {
        "stations": dict(name="stations.csv"),
        "chunks": [
            {
                "station": chunk.station,
                "split_hour_utc": chunk.split_hour_utc,
                "radar": dict(
                    name=chunk.radar_file.path.name, sha256=chunk.radar_file.sha256
                ),
                "ais": dict(
                    name=chunk.ais_file.path.name, sha256=chunk.ais_file.sha256
                ),
            }
            for chunk in chunks
        ],
    }
    with open(data_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)

    chunks = sorted(chunks, key=lambda c: (c.station, c.split_hour_utc))

    return DummyDataset(chunks=chunks, data_dir=data_dir, truth=truth)


@pytest.fixture
def dataset_server(dummy_dataset: DummyDataset):
    with _serve_directory(dummy_dataset.data_dir) as base_url:
        yield base_url, dummy_dataset.chunks
