import pytest
from anguilla import IML
from anguilla.embed import ProjectAndSort
from anguilla.nnsearch import IndexBrute, IndexFast
import torch

@pytest.mark.parametrize('index', [IndexBrute, IndexFast])
@pytest.mark.parametrize('lazy', [False, True])
def test_project_and_sort(index, lazy):
    src_x=16
    src_y=2
    tgt_size=4

    d_src = (src_x,src_y)
    ctrl = torch.rand(d_src)
    z = torch.zeros(tgt_size)

    if lazy:
        iml = IML(emb=ProjectAndSort, index=index)
    else:
        iml = IML(emb=ProjectAndSort(d_src), index=index)

    def iml_map():
        while(len(iml.pairs) < 32):
            src = torch.rand(d_src)
            tgt = z + torch.randn(tgt_size)*2
            iml.add(src, tgt)
    iml_map()

    _z = torch.zeros(tgt_size)
    _z[:] = torch.from_numpy(iml.map(ctrl, k=5))
    indices = torch.randperm(ctrl.shape[0])

    # test invariance to order along batch dimension
    def update_pos():
        indices[:] = torch.randperm(ctrl.shape[0])
        ctrl[:] = ctrl[indices]
        z[:] = torch.from_numpy(iml.map(ctrl, k=5))

    for _ in range(32):
        update_pos()
        assert torch.equal(z, _z), f"Expected z to remain constant, but found difference: {z-_z}"