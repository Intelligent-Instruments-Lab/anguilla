# TODO: import types from iipyper?
import numpy as np
from typing import Any, Optional, List, Tuple, Dict, Union, Callable, Generator
from typing import NamedTuple
from numpy.typing import ArrayLike

torch_present = False
try:
    import torch
    torch_present = True
except ImportError:
    pass

Input = Any # thing that goes into a mapping
Output = Any # thing that comes out of a mapping
Feature = ArrayLike # Inputs are mapped to Features
Scores = ArrayLike # Scores describe distance between inputs in feature space
PairID = int # PairIDs associate Inputs (via Features) with Outputs
PairIDs = ArrayLike

class IOPair(NamedTuple):
    input:Input
    output:Output
class SearchResult(NamedTuple):
    inputs:List[Input]
    outputs:List[Output]
    ids:PairIDs
    scores:Scores

def _np_coerce(x):
    if x is None:
        return None
    if hasattr(x, 'numpy'):
        # torch tensor, etc
        return x.numpy()
    else:
        try:
            # nested iterables case
            return np.stack([_np_coerce(item) for item in x])
            # assert hasattr(x[0], 'numpy')
            # return np.stack(tuple(item.numpy() for item in x))
        except Exception:
            return np.array(x)

def np_coerce(*a):
    return (_np_coerce(x) for x in a)