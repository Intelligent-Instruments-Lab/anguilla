import pytest
from anguilla import IML
from anguilla.embed import *
from anguilla.nnsearch import *
from anguilla.interpolate import *

import numpy as np

@pytest.mark.parametrize('k', [1,2,100])
@pytest.mark.parametrize('index', [IndexBrute, IndexFast])
@pytest.mark.parametrize('embed_input', [Identity, ProjectAndSort, Random])
@pytest.mark.parametrize('interp', [Nearest, Mean, Softmax, Smooth, Ripple])
def test_batching(index, embed_input, interp, k):
    iml = IML(embed_input=embed_input, index=index, interpolate=interp)
    
    rng = np.random.default_rng(0)

    xs = rng.normal(size=(2,99,30))
    ys = rng.normal(size=(99,8))

    if embed_input==ProjectAndSort:
        xs = xs.reshape(2,99,5,6)
    elif embed_input==Random:
        xs = tuple(tuple(tuple(row) for row in x) for x in xs)

    # compute in batch mode
    iml.add_batch(xs[0], ys)
    zs_batch = iml.map_batch(xs[1], k=k)

    iml.reset()

    # compute in serial mode
    for x, y in zip(xs[0], ys):
        iml.add(x, y)
    zs = []
    for x in xs[1]:
        zs.append(iml.map(x, k=k))
    zs = np.stack(zs, 0)

    # print(zs.shape, zs_batch.shape)

    # print(zs_batch[0], zs[0])
    err = np.max(np.abs((zs_batch - zs)))
    print('max error is', err)
    assert err < 1e-2
    assert np.allclose(zs_batch, zs)