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
        """
        Args:
            targets: [k x ...batch dims... x output ]
                first dimension is neighbor dimension
                trailing dimensions are feature dimensions
                remaining dimensions are batch dimensions
            scores: [k x ...batch dims...]

        Returns:
            output: [<batch dims> x output]
        """
        raise NotImplementedError

class Nearest(Interpolate):
    """
    return nearest neighbor (voronoi cell mapping)
    
    NOTE: preferable to use `Mean` with `k=1` in most cases.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, targets, scores):
        # print(f'{scores.shape=}')
        idx = np.argmin(scores, 0)[None,...,None]
        # print(f'{idx.shape=} {targets.shape=}')
        r = np.take_along_axis(targets, idx, 0)[0]
        # print(f'{r.shape=}')
        return r

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

    Scale-equivariant; the scale of inputs interacts with the `temp` parameter. 
    If input data is scaled by a factor `s`, scale `temp` by `s` to compensate.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, targets:List[Output], scores:List[float], temp:float=0.25):
        """
        Args:
            targets: size [K x ...batch dims... x ...output_dims...] 
            scores: size [K x ...batch dims...] 
            temp: temperature of softmax
        """
        scores = scores**0.5

        targets, scores = np_coerce(targets, scores)
        # print(targets.shape, scores.shape)

        if temp==0:
            result = Nearest()(targets, scores)
        else:
            centered = scores - np.min(scores, 0) # for numerical precision
            logits = np.maximum(-centered/temp, -30)
            # print(f'{logits=}')
            weights = np.exp(logits)
            # print(f'{weights=}')
            weights /= weights.sum(0)
            # print(f'{weights=}')
            # result = (np.moveaxis(targets,0,-1)*weights).sum(-1)
            result = np.transpose((
                np.transpose(targets)*np.transpose(weights)).sum(-1))
            
            # 
            # print(f'{logits.shape=}')
            # do_nearest = np.max(np.abs(logits), 0, keepdims=True) > 80
            # result = np.where(do_nearest, Nearest()(targets, scores), result)
        # print(f'{result=}')
        return result

class Smooth(Interpolate):
    """
    Interpolate which tries to prevent discontinuities while preserving the 
    input-output mapping exactly where close to data points.

    Only works for `k > 2`, and generally works well with larger `k`.
    out-of-domain input areas tend to be averages of many outputs.

    Equivalent to `Nearest` if used with `k < 3`.

    Scale-invariant, should work the same if inputs are rescaled.
    (still recommended to keep inputs roughly normalized for numerical stability)
    """
    def __init__(self):
        super().__init__()

    def __call__(self, targets:List[Output], scores:List[float], eps:float=1e-6):
        """
        Args:
            targets: size [K, ...batch dims..., output dim] 
            scores: size [K, ...batch dims...] 
            eps: small value preventing division by zero
        """
        if len(scores) < 3:
            return Nearest()(targets, scores)
        
        targets, scores = np_coerce(targets, scores)

        scores = scores**0.5

        scores = scores + eps
        assert np.min(scores) > 0

        # largest scores -> 0 weight
        # zero score -> inf weight
        # zero first/second derivative at largest score
        mx = np.max(scores, 0)
        # weights = 1/scores + (-3*mx*mx + 3*mx*scores - scores*scores)/(mx**3)
        # weights = 1/scores + (-3*mx*mx + (3*mx - scores)*scores)/(mx**3)
        s_m = scores/mx
        weights = 1/scores + ((3-s_m)*s_m - 3)/mx 

        weights = weights + eps/mx
        weights = weights / weights.sum(0)

        result = np.transpose((
            np.transpose(targets)*np.transpose(weights)).sum(-1))

        return result
    
class Ripple(Interpolate):
    """
    like `Smooth` but with high-frequency ripples outside the input domain.
    
    useful for making random mappings in high dimensional spaces / bootstrapping expressive mappings from a few points.

    Equivalent to `Nearest` if used with `k < 3`

    Scale-equivariant; the scale of inputs interacts with the `ripple` frequency. 
    If input data is scaled by a factor `s`, scale `ripple` by `1/s` to compensate.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, 
            targets:List[Output], scores:List[float], 
            ripple:float=1, ripple_depth:float=1, eps:float=1e-6):
        """
        Args:
            targets: size [K x ...output_dims...] list or ndarray
            scores: size [K] list or ndarray
            ripple: frequency of ripples
            ripple_depth: amplitude of ripples
            eps: small value preventing division by zero
        """
        if len(scores) < 3:
            return Nearest()(targets, scores)
        
        targets, scores = np_coerce(targets, scores)

        scores = scores**0.5

        scores = scores + eps
        assert np.min(scores) > 0

        # largest scores -> 0 weight
        # zero score -> inf weight
        mx = np.max(scores, 0)
        # weights = 1/scores + (-3*mx*mx + 3*mx*scores - scores*scores)/(mx**3)
        s_m = scores/mx
        weights = 1/scores + ((3-s_m)*s_m - 3)/mx 

        weights = weights * 2**(
            ripple_depth * 
            (1+np.cos(2*np.pi/mx*scores)*np.sin(2*np.pi*ripple*scores))
            )

        weights = weights + eps/mx
        weights = weights / weights.sum(0)

        # result = (np.moveaxis(targets,0,-1)*weights).sum(-1)
        result = np.transpose((
            np.transpose(targets)*np.transpose(weights)).sum(-1))

        return result