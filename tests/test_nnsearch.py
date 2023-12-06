import pytest
import numpy as np
from anguilla.nnsearch import *

@pytest.mark.parametrize('index', [IndexBrute, IndexFast])
@pytest.mark.parametrize('metric', [sqL2])
@pytest.mark.parametrize('d', [None,1,2,99])
@pytest.mark.parametrize('k', [1,2,5,100])
def test_nnsearch(d, k, index, metric):
    nn = NNSearch(index(d, metric()))
    # should print a warning but not raise an Exception
    assert nn.get(0) is None

    # use deterministic, but different rng per param
    rng = np.random.default_rng(abs(hash((d,k,str(index),str(metric)))))

    # in d=None case, pick an actual d now that the NNSearch object is made:
    if d is None:
        d = rng.integers(1, 100)
    # fewer than k points case
    pts = []
    ids = []
    for _ in range(k-1):
        pt = rng.normal(size=d)
        pts.append(pt)
        ids.append(nn.add(pt))

    # test `get`
    for pt, id in zip(pts, ids):
        assert np.allclose(nn.get(id), pt)

    # test `search` (excluding k=1 cases for now)
    if k>1:
        i = rng.choice(ids)
        pt = nn.get(i)
        nbrs, scores = nn(pt, k=k)
        assert i in nbrs, "query point should be in neighbors"
        assert len(nbrs)==k-1, "should be total number of points, which is <k"
        for nbr, s in zip(nbrs, scores):
            if nbr == i:
                assert min(scores) == s, "query point should have min score"

    # more than k points case:
    for _ in range(k+1):
        pt = rng.normal(size=d)
        pts.append(pt)
        ids.append(nn.add(pt))

    # test `search` 
    i = rng.choice(ids)
    pt = nn.get(i)
    nbrs, scores = nn(pt, k=k)
    assert i in nbrs, "query point should be in neighbors"
    assert len(nbrs)==k, "should be k neighbors"
    for nbr, s in zip(nbrs, scores):
        if nbr == i:
            assert min(scores) == s, "query point should have min score"

    # use the nn as iterator over ids
    # remove random points
    for _ in range(k-1):
        i = rng.choice(list(nn))
        nn.remove(i)

    # use nn.items()
    # check that ids and points still correspond
    pairs = zip(ids, pts)
    for i,p in nn.items():
        assert any(i==j and np.allclose(p,q) for j,q in pairs)

    # remove_near
    i = rng.choice(list(nn))
    assert i in nn, "query point should be present"

    pt = nn.get(i)
    nn.remove_near(pt, k=1)
    assert i not in nn, "query point should have been removed"
