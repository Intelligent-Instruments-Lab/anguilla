import pytest
import numpy as np
from anguilla.nnsearch import *

@pytest.mark.parametrize('index', [IndexNumpy, IndexBrute, IndexFast])
@pytest.mark.parametrize('metric', [sqL2])
@pytest.mark.parametrize('d', [None,(1,1),(2,2),(98,99)])
@pytest.mark.parametrize('k', [1,2,5,100])
def test_nnsearch(d, k, index, metric):
    index = index(d, metric())
    # should print a warning but not raise an Exception
    assert index.get(0) == None

    # use deterministic, but different rng per param
    rng = np.random.default_rng(abs(hash((d,k,str(index),str(metric)))))

    # in d=None case, pick an actual d now that the NNSearch object is made:
    if d is None:
        d_in, d_out = rng.integers(1, 100), rng.integers(1, 100)
    else:
        d_in, d_out = d

    # fewer than k points case
    pts = (rng.normal(size=(k-1, d_in)), rng.normal(size=(k-1, d_out)))
    ids = index.add(*pts)
    ids = list(ids)

    # test `get`
    for z, w, i in zip(*pts, ids):
        z_, w_ = index.get(i)
        assert np.allclose(z, z_)
        assert np.allclose(w, w_)

    # test `search` (excluding k=1 cases for now)
    if k>1:
        i = rng.choice(ids)
        z, _ = index.get(i)
        _, _, nbrs, scores = index.search([z], k=k)
        nbrs = nbrs[0]
        scores = scores[0]
        assert i in nbrs, "query point should be in neighbors"
        assert len(nbrs)==k-1, "should be total number of points, which is <k"
        for nbr, s in zip(nbrs, scores):
            if nbr == i:
                assert min(scores) == s, "query point should have min score"

    # more than k points case:
    new_pts = (rng.normal(size=(k+1, d_in)), rng.normal(size=(k+1, d_out)))
    ids.extend(index.add(*new_pts))
    pts = (
        np.concatenate((pts[0], new_pts[0]), 0), 
        np.concatenate((pts[1], new_pts[1]), 0))

    # test `search` 
    i = rng.choice(ids)
    z, _ = index.get(i)
    _, _, nbrs, scores = index.search([z], k=k)
    nbrs = nbrs[0]
    scores = scores[0]
    assert i in nbrs, "query point should be in neighbors"
    assert len(nbrs)==k, "should be k neighbors"
    for nbr, s in zip(nbrs, scores):
        if nbr == i:
            assert min(scores) == s, "query point should have min score"

    # use index.items()
    # print(f'{ids=}')
    for i,(pz,pw) in index.items():
        # print(i)
        assert any(
            i==j 
            # and np.allclose(pw,qw)
            # and np.allclose(pz,qz) 
            for j,qz,qw in zip(ids, *pts))
        
    # use the index as iterator over ids
    # remove random points
    for _ in range(k-1):
        i = rng.choice(list(index), size=1)
        index.remove(i)

    # check that ids and points still correspond
    for i,(pz,pw) in index.items():
        assert any(
            i==j and np.allclose(pw,qw) and np.allclose(pz,qz) 
            for j,qz,qw in zip(ids, *pts))

    # remove_near
    i = rng.choice(list(index))
    assert i in index, "query point should be present"

    z, _ = index.get(i)
    index.remove_near([z], k=1)
    assert i not in index, "query point should have been removed"

    # replace all points
    n = len(index)
    pts = (rng.normal(size=(n, d_in)), rng.normal(size=(n, d_out)))
    ids = list(index)
    print(ids)
    index.add(*pts, ids=ids)

