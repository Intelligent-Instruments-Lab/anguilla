import numpy as np

from .types import *
from .serialize import JSONSerializable

class Interpolate(JSONSerializable):
    """
    `Interpolate` combines a set of `Outputs` weighted by dissimilarity scores.

    The scores depend on the `Metric` used by the `NNSearch`. 
    They may be, for example, distances or negative cosine similarities.
    """
    def __init__(self, **kw):
        super().__init__(**kw)

    def __call__(self, targets: List[Output], scores: Scores) -> Output:
        raise NotImplementedError

class Nearest(Interpolate):
    """return nearest neighbor (voronoi cell mapping)"""
    def __init__(self):
        super().__init__()

    def __call__(self, targets, scores):
        return targets[np.argmin(scores)]

class Mean(Interpolate):
    """mean of neighbors (piecewise constant mapping)"""
    def __init__(self):
        super().__init__()

    def __call__(self, targets, scores):
        return sum(targets) / len(targets)

class Softmax(Interpolate):
    """
    Like `Mean`, but weighted toward the nearer neighbors.
    
    when `k` is small, has discontinuities
    when temp is large, acts more like `Mean`.
    -> tends to get 'washed out' for larger `k` / larger temp

    when temp is small, acts more like `Nearest` (voronoi cells).
    """
    def __init__(self):
        super().__init__()

    def __call__(self, targets:List[Output], scores:List[float], temp:float=0.5):
        """
        Args:
            targets: size [K x ...output_dims...] list or ndarray
            scores: size [K] list or ndarray
            temp: temperature of softmax
        """
        targets, scores = np_coerce(targets, scores)
        # print(targets.shape, scores.shape)

        if temp==0:
            result = targets[np.argmin(scores)]
        else:
            centered = scores - np.mean(scores) # for numerical precision
            logits = np.maximum(-centered/temp, -20)
            # print(f'{logits=}')
            if np.max(np.abs(logits)) > 80:
                result = targets[np.argmin(scores)]
            else:
                weights = np.exp(logits)
                # print(f'{weights=}')
                weights /= weights.sum()
                # print(f'{weights=}')
                result = (np.moveaxis(targets,0,-1)*weights).sum(-1)
        # print(f'{result=}')
        return result

class Smooth(Interpolate):
    """
    Interpolate which is non-discontinuous (for `k` > 2)
    
    tries to prevent discontinuities while preserving the input-output mapping
    exactly where close to data points.

    works well with larger `k`.
    out-of-domain input areas tend to be averages of many outputs.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, targets:List[Output], scores:List[float], eps:float=1e-9):
        """
        Args:
            targets: size [K x ...output_dims...] list or ndarray
            scores: size [K] list or ndarray
            eps: small value preventing division by zero
        """
        targets, scores = np_coerce(targets, scores)

        scores = scores + eps
        assert np.min(scores) > 0

        # largest scores -> 0 weight
        # zero score -> inf weight
        # zero first/second derivative at largest score
        mx = np.max(scores)
        weights = 1/scores + (-3*mx*mx + 3*mx*scores - scores*scores)/(mx**3)

        weights = weights + eps
        weights = weights / weights.sum()

        result = (np.moveaxis(targets,0,-1)*weights).sum(-1)

        return result
    
class Ripple(Interpolate):
    """
    like `Smooth` but with high-frequency ripples outside the input domain.
    
    useful for making random mappings in high dimensional spaces / bootstrapping expressive mappings from a few points.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, 
            targets:List[Output], scores:List[float], 
            ripple:float=1, ripple_depth:float=1, eps:float=1e-9):
        """
        Args:
            targets: size [K x ...output_dims...] list or ndarray
            scores: size [K] list or ndarray
            ripple: frequency of ripples
            ripple_depth: amplitude of ripples
            eps: small value preventing division by zero
        """
        targets, scores = np_coerce(targets, scores)

        scores = scores + eps
        assert np.min(scores) > 0

        # largest scores -> 0 weight
        # zero score -> inf weight
        mx = np.max(scores)
        weights = 1/scores + (-3*mx*mx + 3*mx*scores - scores*scores)/(mx**3)
        weights = weights * 2**(
            ripple_depth * 
            (1+np.cos(np.pi*scores/mx)*np.sin(scores*np.pi*ripple))
            )

        weights = weights + eps
        weights = weights / weights.sum()

        result = (np.moveaxis(targets,0,-1)*weights).sum(-1)

        return result