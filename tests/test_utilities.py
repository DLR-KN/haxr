import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose

from haxr.utilities import load_cycle


@pytest.mark.parametrize("cycle", [0, 1])
def test_load_cycle(dummy_dataset, cycle):
    chunk = dummy_dataset.chunks[0]
    truth = dummy_dataset.truth

    with h5py.File(chunk.radar_file.path, "r") as f:
        df = load_cycle(f, cycle)

    cols = truth["column_names"]
    assert sorted(df.columns) == sorted(cols.values())

    first = truth["cycle_first"]
    last = truth["cycle_last"]

    i = int(first[cycle])
    j = int(last[cycle]) + 1

    assert list(df.index) == list(range(i, j))

    arrays = truth["arrays"]
    assert_allclose(df[cols["tod"]].to_numpy(), arrays["tod"][i:j])
    assert_allclose(df[cols["az1"]].to_numpy(), arrays["az1"][i:j])
    assert_allclose(df[cols["az2"]].to_numpy(), arrays["az2"][i:j])
    assert_allclose(df[cols["r1"]].to_numpy(), arrays["r1"][i:j])
    assert_allclose(df[cols["r2"]].to_numpy(), arrays["r2"][i:j])

    assert df[cols["amp"]].dtype == np.uint8
    assert (df[cols["amp"]].to_numpy() == arrays["amp"][i:j]).all()

    derived = truth["derived"]
    assert_allclose(
        df[cols["az"]].to_numpy(), np.round(derived["az"][i:j], 3), rtol=0.0, atol=1e-3
    )
    assert_allclose(
        df[cols["r"]].to_numpy(), np.round(derived["r"][i:j], 2), rtol=0.0, atol=1e-6
    )
