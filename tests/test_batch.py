import pytest
from anguilla import IML
from anguilla.embed import *
from anguilla.nnsearch import *
from anguilla.interpolate import *

import numpy as np

@pytest.mark.parametrize('index', [IndexBrute, IndexFast])
@pytest.mark.parametrize('emb', [Identity, ProjectAndSort, Random])
@pytest.mark.parametrize('interp', [Nearest, Mean, Softmax, Smooth, Ripple])
def test_batching(index, emb, interp):
    iml = IML(emb=emb, index=index, interp=interp)
    
    rng = np.random.default_rng(0)

    xs = rng.normal(size=(2,99,30))
    ys = rng.normal(size=(99,8))

    if emb==ProjectAndSort:
        xs = xs.reshape(2,99,5,6)
    elif emb==Random:
        xs = tuple(tuple(tuple(row) for row in x) for x in xs)

    iml.add_batch(xs[0], ys)
    zs_batch = iml.map_batch(xs[1])

    iml.reset()
    for x, y in zip(xs[0], ys):
        iml.add(x, y)

    zs = []
    for x in xs[1]:
        zs.append(iml.map(x))
    zs = np.stack(zs, 0)

    print(zs_batch[0], zs[0])
    assert np.allclose(zs_batch, zs)