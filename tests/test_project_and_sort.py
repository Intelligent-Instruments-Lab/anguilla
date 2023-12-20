import pytest
import numpy as np

from anguilla import IML
from anguilla.embed import ProjectAndSort, Identity
from anguilla.nnsearch import IndexBrute, IndexFast

@pytest.mark.parametrize('index', [IndexBrute, IndexFast])
@pytest.mark.parametrize('lazy', [False, True])
def test_project_and_sort(index, lazy):
    src_x=16
    src_y=2
    tgt_size=4

    rng = np.random.default_rng(0)
    d_src = (src_x,src_y)
    ctrl = rng.random(d_src)

    if lazy:
        iml = IML(embed_input=ProjectAndSort, index=index)
    else:
        iml = IML(
            embed_input=ProjectAndSort(d_src),
            embed_output=Identity(tgt_size),
            index=index)

    def iml_map():
        while(len(iml.pairs) < 32):
            src = rng.random(d_src)
            tgt = rng.random(tgt_size)*2
            iml.add(src, tgt)
    iml_map()

    _z = np.zeros(tgt_size)
    _z[:] = iml.map(ctrl, k=5)
    indices = rng.permutation(ctrl.shape[0])

    # test invariance to order along batch dimension
    z = np.zeros(tgt_size)
    def update_pos():
        indices[:] = rng.permutation(ctrl.shape[0])
        ctrl[:] = ctrl[indices]
        z[:] = iml.map(ctrl, k=5)

    for _ in range(32):
        update_pos()
        assert np.allclose(z, _z), f"Expected z to remain constant, but found difference: {z-_z}"