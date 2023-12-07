import pytest
from anguilla.embed import *

@pytest.mark.parametrize('x', [0, [0], [0,1], [[0]], [[0,1],[2,3]], [[[0]]]])
def test_id_chain_cat(x):
    emb = Identity()
    z = emb(x)
    if hasattr(x, '__len__'):
        assert len(x) == z.shape[0]
    else:
        assert z.shape == tuple()

    emb = Chain(Identity(), Identity())
    assert np.allclose(emb(x), z)

    emb = Cat(Identity(), Identity())
    if hasattr(x, '__len__'):
        assert np.allclose(np.concatenate([x,x],-1), emb([x,x]))
    else:
        assert np.allclose(np.stack([x,x],-1), emb([x,x]))

@pytest.mark.parametrize('x', [('a',0,object()), [(0,1), frozenset(('a','b'))]])
@pytest.mark.parametrize('d', [1,2,99])
def test_cat_random(x, d):
    r = Random(d)
    emb = Cat(*[r]*len(x))
    assert np.allclose(emb(x), emb(x))
