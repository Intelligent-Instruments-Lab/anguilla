from .types import *
from .serialize import JSONSerializable

import numpy as np

class Embedding(JSONSerializable):
    def __init__(self, **kw):
        super().__init__(**kw)
        # set to True if ArrayLike inputs with
        # leading batch dimensions are supported
        self.is_batched = False

    def __call__(self, source: Input) -> Feature:
        raise NotImplementedError
    
    def inv(self, target: Feature) -> Input:
        raise NotImplementedError


class Identity(Embedding):
    """
    The most basic Embedding. 
    Optionally checks size of input vector and ensures it is a numpy array, 
    but otherwise leaves it unchanged.
    """
    def __init__(self, size:Optional[int]=None):
        """
        Args:
            size: both the Input and the Feature size.
                if supplied, inputs will be validated to be that size.
                otherwise, they will just be converted to ndarrays.
        """
        super().__init__(size=size)
        self.size = self.input_size = size
        self.is_batched = True
        # self.in_type = lambda x:x

    def __call__(self, source) -> ArrayLike:
        """
        embed the input.

        Args:
            source: an input sequence
        
        Returns:
            feature: the input sequence as a numpy array.
        """
        # self.in_type = type(source)

        source, = np_coerce(source)

        if self.size is not None:
            if isinstance(self.size, int):
                assert source.shape[-1] == self.size, (source.shape, self.size)
            else:
                assert source.shape[-len(self.size):] == self.size, (source.shape, self.size)
        return source
    
    def inv(self, w):
        # print(self.in_type, type(w), w.shape)
        # w = self.in_type(w)
        return w
    
# Random, RandomNormal embedding,
# item is hashed and used to sample a distribution
class Random(Embedding):
    """
    Pseudo-random uniform embedding in [-1, 1] for any hashable inputs.
    """
    def __init__(self, size:int=1):
        """
        Args:
            size: the Feature size.
        """
        super().__init__(size=size)
        self.input_size = None
        self.size = size

    def __call__(self, source) -> ArrayLike:
        """
        embed the input.

        Args:
            source: a hashable input. cannot be batched.
        
        Returns:
            feature: uniformly distributed pseudo-random vector
        """
        # TODO: treat lists as batches?

        # get 32 bits from the input
        h = hash(source) & 8589934591
        rng = np.random.default_rng(h)
        return rng.random(size=self.size)*2 - 1
    
class RandomNormal(Embedding):
    """
    Pseudo-random Gaussian embedding for any hashable inputs.
    """
    def __init__(self, size:int=1):
        """
        Args:
            size: the Feature size.
        """
        super().__init__(size=size)
        self.input_size = None
        self.size = size

    def __call__(self, source) -> ArrayLike:
        """
        embed the input.

        Args:
            source: an input sequence
        
        Returns:
            feature: normally distributed pseudo-random vector
        """
        # get 32 bits from the input
        h = hash(source) & 8589934591
        rng = np.random.default_rng(h)
        return rng.normal(size=self.size)

class ProjectAndSort(Embedding):
    """
    Embedding for point cloud-like data.
    use with an L2 distance `Metric` to compute sliced optimal transport.

    if an Input is a 2D array [B x C],
    B being the point dimension (order not meaningful)
    and C being the coordinate dimension (order meaningful)

    e.g.
    [
    [x0,y0,z0],
    [x1,y1,z1],
    [x2,y2,z2],
    [x3,y3,z3],
    ]
    would be a cloud of B=4 points in C=3 dimensional space

    This computes `n` pseudo-random projections down to 1-dimensional spaces,
    sorts along those lines,
    and then concatenates to make one feature vector.
    the L2 distance between feature vectors is the sliced OT distance between point clouds.
    """
    def __init__(self, input_size:Tuple[int,int]=None, n:int=16, seed:int=0):
        """
        Args:
            input_size: input shape [B,C]; if None, lazy init on first __call__
            n: number of random projections.
            seed: random seed.
        """
        super().__init__(input_size=input_size, n=n)
        self.n = n
        self.seed = seed
        self.is_batched = True

        if input_size is not None:
            self.init(input_size)
        else:
            self.input_size = None
            self.size = None

    def init(self, input_size):
        assert len(input_size)>=2, "ProjectAndSort expects at least 2D array data"

        self.rng = np.random.default_rng(self.seed)

        self.input_size = tuple(input_size)

        self.size = input_size[0] * self.n

        proj = self.rng.normal(size=(input_size[1], self.n))
        proj = proj / np.linalg.norm(proj, axis=0, keepdims=True)
        self.proj = proj

    def __call__(self, source) -> ArrayLike:
        """
        embed the input.

        Args:
            source: an 2d input sequence of shape [batch x coordinates]     
                representing a set of points.
        
        Returns:
            feature: 1d array of concatenated projections of the input points.
        """
        source, = np_coerce(source)
        if self.input_size is None:
            # lazy init
            self.init(source.shape[-2:])
        else:
            assert source.shape[-2:] == self.input_size, (source.shape, self.input_size)

        # project coordinate dimension to n lines
        feat = source @ self.proj
        # sort along the lines
        feat = np.sort(feat, axis=-2)
        # flatten
        # feat = feat.T
        feat = feat.reshape((*feat.shape[:-2], -1))

        return feat / np.sqrt(self.size)

### Embeddings for combining other Embeddings

class Cat(Embedding):
    """
    Embedding which applies multiple other embeddings to elements of a sequence,
    and concatenates the results
    """
    def __init__(self, *embs):
        self.embs = embs
    def __call__(self, sources):
        assert len(sources)==len(self.embs)
        parts = [emb(source) for source, emb in zip(sources, self.embs)]
        # convert 0d scalars to 1d before cat
        parts = [p[None] if p.ndim==0 else p for p in parts]
        return np.concatenate(parts, -1)
    
class Chain(Embedding):
    """
    Embedding which applies multiple embeddings in series to the same input
    """
    def __init__(self, *embs):
        self.embs = embs
    def __call__(self, source):
        for emb in self.embs:
            source = emb(source)
        return source