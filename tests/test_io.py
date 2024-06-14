import pytest
from anguilla import IML
from anguilla.nnsearch import *

import numpy as np

arrtypes = [
    (list, lambda x: x.tolist()),
    (np.ndarray, np.array)
    ]

try:
    import torch
    arrtypes.append((torch.Tensor, torch.tensor))
except ImportError:
    print("warning: torch not installed, not testing with torch types")

@pytest.mark.parametrize('index', [IndexBrute, IndexFast])
@pytest.mark.parametrize('arrtype', arrtypes)
@pytest.mark.parametrize('dtype', [float, np.float16, np.float32, np.float64])
def test_saveload(tmp_path, index, arrtype, dtype):
    arrtype, arrcons = arrtype

    iml = IML(index=index)
    
    rng = np.random.default_rng(0)

    xs = rng.normal(size=(99,30)).astype(dtype)
    ys = rng.normal(size=(99,8)).astype(dtype)

    xs, ys = arrcons(xs), arrcons(ys)

    iml.add_batch(xs, ys)

    iml.save(tmp_path / "test.json")

    iml2 = iml.load(tmp_path / "test.json")

    # TODO test equality
    assert len(iml.pairs) == len(iml2.pairs)

    for k in iml.pairs:
        assert k in iml2.pairs
        assert isinstance(iml2.pairs[k].input, arrtype)
        assert isinstance(iml2.pairs[k].output, arrtype)
        assert all(np.equal(iml.pairs[k].input, iml2.pairs[k].input))
        assert all(np.equal(iml.pairs[k].output, iml2.pairs[k].output))
        assert np.allclose(iml.map([0]*30), iml2.map([0]*30)) 
        # iml.map([0.0]*30)
