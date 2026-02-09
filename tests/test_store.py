import pandas as pd

from haxr.store import Chunk, File, Store


def test_chunk_lookup(dataset_server, tmp_path):
    cache_dir = tmp_path / "cache"

    base_url, chunks = dataset_server
    chunks = [
        Chunk(
            station=chunk.station,
            split_hour_utc=chunk.split_hour_utc,
            radar_file=File(
                path=cache_dir / chunk.radar_file.path.name,
                sha256=chunk.radar_file.sha256,
            ),
            ais_file=File(
                path=cache_dir / chunk.ais_file.path.name, sha256=chunk.ais_file.sha256
            ),
        )
        for chunk in chunks
    ]

    stations = [chunk.station for chunk in chunks]

    with Store(base_url=base_url, cache_dir=cache_dir) as store:
        assert store.list_chunks() == chunks

        for station in stations:
            expected_chunks = [c for c in chunks if c.station == station]
            assert store.list_chunks(station=station) == expected_chunks

        for chunk in chunks:
            expected_chunks = [
                c
                for c in chunks
                if c.station == chunk.station
                and c.split_hour_utc == chunk.split_hour_utc
            ]
            assert len(expected_chunks) == 1

            assert (
                store.get_chunk(
                    station=chunk.station, split_hour_utc=chunk.split_hour_utc
                )
                == expected_chunks[0]
            )


def test_caching(dataset_server, tmp_path):
    cache_dir = tmp_path / "cache"

    base_url, _ = dataset_server
    with Store(base_url=base_url, cache_dir=cache_dir) as store:
        for chunk in store.list_chunks():
            assert not chunk.radar_file.path.is_file()
            assert not chunk.ais_file.path.is_file()

            assert store.ensure(chunk) == chunk
            assert chunk.radar_file.path.is_file()
            assert chunk.ais_file.path.is_file()


def test_stations_lookup(dataset_server, tmp_path):
    cache_dir = tmp_path / "cache"

    base_url, _ = dataset_server
    with Store(base_url=base_url, cache_dir=cache_dir) as store:
        stations = store.stations

        assert isinstance(stations, pd.DataFrame)
        assert not stations.empty


def test_ais_loading(dataset_server, tmp_path):
    cache_dir = tmp_path / "cache"

    base_url, _ = dataset_server
    with Store(base_url=base_url, cache_dir=cache_dir) as store:
        ais_file = store.list_chunks()[0].ais_file
        ais_data = store.load_ais_data(ais_file)

        assert isinstance(ais_data, pd.DataFrame)
        assert not ais_data.empty
