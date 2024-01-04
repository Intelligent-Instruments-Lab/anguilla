#### TODO: decide API for Index.
#### batched everywhere? 
##### what about remove_near?
##### handle batched/unbatched inputs in IML, or base Index?


import itertools as it
import numpy as np
from .types import *    
from .serialize import JSONSerializable
from scipy.spatial.distance import cdist

class Metric(JSONSerializable):
    """
    define a distance between two points. 
    Relative distances will be used to find nearest neighbors,
    and the distances to neighbors will be passed to `Interpolate`.
    """
    def __call__(self, a, b):
        raise NotImplementedError

class sqL2(Metric):
    def __call__(self, a, b):
        return np.sum((a-b)**2, axis=-1)

class Index(JSONSerializable):
    """
    base Index class.
    currently no function besides typing, warning of unimplemented features.

    Subclasses of Index implement nearest neighbor search with different
    cababilities and performance tradeoffs.
    """
    def __init__(self, k=10, **kw):
        super().__init__(k=k, **kw)
        self.default_k = k
        self.is_batched = False

    def add(self, 
            zs:List[Feature], ws:List[Feature], ids:Optional[List[PairID]]=None
        ) -> List[PairID]:
        raise NotImplementedError
    def remove(self, ids:List[PairID]) -> List[PairID]:
        raise NotImplementedError
    def get(self, ids:PairID) -> IOPair:
        """not batched (mainly for testing purposes)"""
        raise NotImplementedError
    def search(self, zs:List[Feature], k:int) -> SearchResult:
        raise NotImplementedError
    def reset(self):
        raise NotImplementedError
    @property
    def ids(self):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.ids)
    
    def __iter__(self):
        """iterate over IDs in the index"""
        return iter(self.ids)
    
    def items(self) -> Generator[Tuple[PairID, IOPair], None, None]:
        """iterate over (ID, (Input, Output))"""
        def iterator():
            for i in self.ids:
                yield i, self.get(i)
        return iterator()
    
    def distance(self, a:Feature, b:Feature) -> float:
        """compute distance between two features"""
        return self.metric(a, b)
    
    def remove_near(self, zs:List[Feature], k:int=None) -> PairIDs:
        """
        Remove point(s) from the index by proximity.
        Use k=1 to remove a single point.

        Args:
            zs: batch of query input features
            k: number of nearest neighbors to each query point to remove
                (defaults to index default k)
        """
        zs, = np_coerce(zs)
        assert zs.ndim==2, zs.shape

        k = k or self.default_k

        _, _, ids, _ = self.search(
            zs, k, return_inputs=False, return_outputs=False, return_ids=True)
        
        ids = set(ids.ravel())
        return self.remove(ids)


class IndexBrute(Index):
    """
    Optimized for simplicity and flexibility,
    may not scale to large datasets.
    """
    def __init__(self, d:Tuple[int, int]=None, metric:Callable=None):
        """
        Args:
            d: optional, dimension of (input, output) features
            metric: distance metric, default to squared euclidean
        """
        if metric is None:
            metric = sqL2()
        
        super().__init__(d=d, metric=metric)
        self.d = d
        self.metric = metric

        self.reset()

    def add(self, 
            zs:List[Feature], 
            ws:List[Feature], 
            ids:Optional[List[PairID]]=None
            ) -> PairID:
        """add a new feature, return its ID.
        Args:
            feature: the feature to add
            id: if not supplied, generate a new ID;
                otherwise, use the supplied id.
                supply an existing id to replace.
        """
        if ids is None:
            ids = it.repeat(None)
        added_ids = []
        for z,w,i in zip(zs, ws, ids):
            if i is None:
                i = max(self.z_data, default=-1) + 1
            self.z_data[i] = z
            self.w_data[i] = w
            added_ids.append(i)
        return added_ids
    
    def remove(self, ids:List[PairID]):
        """remove points by ID"""
        removed_ids = []
        for i in ids:
            del self.z_data[i]
            if i in (self.w_data):
                del self.w_data[i]
                removed_ids.append(i)
            else:
                print(f'WARNING: anguilla: no point {i} in index')
        return removed_ids
                
    def get(self, i:List[PairID]) -> Tuple[List[Feature], List[Feature]]:
        """get a pair by ID"""
        if i in self.z_data:
            return self.z_data[i], self.w_data[i]
        else:
           return None

    def search(self, 
            zs:Feature, k:int=None,
            return_inputs=True,
            return_outputs=True, 
            return_ids=True
        ) -> SearchResult:
        """get feature(s) and IDs by proximity"""
        k = k or self.default_k

        zs_batch = [] if return_inputs else None
        ws_batch = [] if return_outputs else None
        ids_batch = [] if return_ids else None
        scores_batch = []

        for z in zs:
            dist_id = sorted((self.metric(z, v),k) for k,v in self.z_data.items())

            scores, ids = zip(*dist_id[:k])

            if return_inputs:
                zs_batch.append([self.z_data[i] for i in ids])
            if return_outputs:
                ws_batch.append([self.w_data[i] for i in ids])
            if return_ids:
                ids_batch.append(ids)

            scores_batch.append(scores)

        return SearchResult(
            np.array(zs_batch),
            np.array(ws_batch), 
            np.array(ids_batch), 
            np.array(scores_batch))
    
    def reset(self):
        self.z_data:Dict[PairID, Feature] = {}
        self.w_data:Dict[PairID, Feature] = {}

    @property
    def ids(self):
        return self.z_data.keys()
    
class IndexNumpy(Index):
    """
    """
    def __init__(self, 
            d:Optional[Tuple[int,int]]=None, 
            metric:Callable=None, 
            k=10, 
            max_n:int=10000, 
            **kw):
        if metric is None:
            metric = sqL2()
        super().__init__(d=d, metric=metric, max_n=max_n, k=k, **kw)
        self.max_n = max_n
        self.metric = metric
        self.needs_init = True
        if d is not None:
            self.init(d)

    def init(self, d):
        d_in, d_out = d
        # index maps id to internal index
        self.id_to_idx = np.empty(self.max_n, dtype=np.int32)
        self.idx_to_id = np.empty(self.max_n, dtype=np.int32)
        self.z_data = np.empty((self.max_n, d_in), dtype=np.float32)
        self.w_data = np.empty((self.max_n, d_out), dtype=np.float32)
        self.needs_init = False
        self.reset()

    def add(self, 
            zs:List[Feature], 
            ws:List[Feature], 
            ids:Optional[List[PairID]]=None
        ) -> List[PairID]:
        if self.needs_init:
            self.init((zs.shape[-1], ws.shape[-1]))

        if ids is None:
            ids = idx = np.arange(self.n, self.n+len(zs))
            self.n += len(zs)
        else:
            # ids which are present are mapped to idx
            # other ids are new idx
            idx = self.id_to_idx[ids]
            id_missing = idx==-1
            n_missing = np.count_nonzero(id_missing)
            if n_missing > 0:
                idx[id_missing] = np.arange(self.n, self.n+n_missing)
                self.n += n_missing

        if self.n >= self.max_n:
            print('WARNING: anguilla: IndexConvex full')
            return []
        
        self.id_to_idx[ids] = idx
        self.idx_to_id[idx] = ids
        self.z_data[idx] = zs
        self.w_data[idx] = ws

        return ids
    
    def remove(self, ids:List[PairID]) -> List[PairID]:
        print(ids)
        print(self.id_to_idx)
        for i in ids:
            idx = self.id_to_idx[i]
            last_idx = self.n-1
            if idx == -1:
                raise ValueError
            elif idx > last_idx:
                raise ValueError
            elif idx < last_idx:
                # move the last item into place of removed item
                j = self.idx_to_id[last_idx]
                self.id_to_idx[j] = idx
                self.idx_to_id[idx] = j
                self.z_data[idx] = self.z_data[last_idx]
                self.w_data[idx] = self.w_data[last_idx]
            self.n -= 1
            self.idx_to_id[last_idx] = -1
            self.id_to_idx[i] = -1
                
        return ids
    
    def get(self, ids:PairID) -> IOPair:
        """not batched (mainly for testing purposes)"""
        if self.needs_init:
            print(f'WARNING: anguilla: uninitialized index')
            return None
        idx = self.id_to_idx[ids]
        if idx==-1:
            print(f'WARNING: anguilla: no point with ID {ids} in index')
            return None
        return IOPair(self.z_data[idx], self.w_data[idx])
    
    def search(self, z:List[Feature], k:int=None, 
            return_inputs=True,
            return_outputs=True, 
            return_ids=True) -> SearchResult:
        """
        Args:
            z: batch x feature
        Returns:
            zs: [batch, k, input feature]
            ws: [batch, k, output feature]
            ids: [batch, k]
            scores: [batch, k]
        """
        k = k or self.default_k
        z, = np_coerce(z)
        # compute distances 
        data = self.z_data[:self.n]
        if isinstance(self.metric, sqL2):
            scores = cdist(z, data, metric='sqeuclidean')
        else:
            scores = self.metric(z[:,None,:], data) # [batch, n]

        # idx = np.argsort(scores, axis=1)[:,:k] # [batch, min(n,k)]
        if self.n <= k:
            idx = np.arange(self.n)[None]
        else:
            idx = np.argpartition(scores, k, axis=1)[:,:k] # [batch, min(n,k)]
        # print(data.shape)

        scores = np.take_along_axis(scores, idx, axis=1)
        zs = self.z_data[idx] if return_inputs else None
        ws = self.w_data[idx] if return_outputs else None
        ids = self.idx_to_id[idx] if return_ids else None
        # print(idx.shape, scores.shape, self.z_data.shape, self.idx_to_id.shape)
        # print(zs.shape, ws.shape, ids.shape)

        return SearchResult(zs, ws, ids, scores)
    
    def reset(self):
        self.n = 0
        if not self.needs_init:
            self.id_to_idx[:] = -1
            self.idx_to_id[:] = -1
    
    @property
    def ids(self):
        return self.idx_to_id[:self.n]

try:
    import faiss
    from faiss import IndexFlatL2
    class IndexFast(Index):
        """
        Optimized for fast `search` on large vectors / datasets.
        Only L2 distance supported. 
        `remove` may be slow.

        This is currently a wrapper around `faiss.FlatIndexL2` which provides stable ids when using `remove`.
        In the future could support dot product and/or approximate search indices.
        """
        def __init__(self, 
            d:Optional[Tuple[int,int]]=None, 
            metric:Callable=sqL2, 
            k=10):
            """
            Args:
                d: dimensions of (input, output)
                metric:currently must be instance of `sqL2`
            """
            super().__init__(d=d, metric=metric, k=k)
            self.is_batched = True # `search` supports batching

            if isinstance(metric, type) and issubclass(metric, Metric):
                self.metric = metric()
            else:
                self.metric = metric

            self.z_index = None
            self.w_index = None
            if d is not None:
                self.init(d)

        def init(self, d):
            d_in, d_out = d
            if isinstance(self.metric, sqL2):
                self.z_index = IndexFlatL2(d_in)
                self.w_index = IndexFlatL2(d_out)
            else:
                raise ValueError("""IndexFast supports only sqL2 metric""")
            self.reset()

        @property
        def d_in(self):
            return self.z_index.d
        @property
        def d_out(self):
            return self.w_index.d

        def add(self, zs:Feature, ws:Feature, ids:Optional[PairID]=None):
            """add a new feature, return its ID.
            Args:
                zs: batch of input features to add
                ws: batch of output features to add
                ids: if not supplied, generate new IDs;
                    otherwise, use the supplied ids.
                    already existing ids will be replaced.
            """
            zs, ws, ids = np_coerce(zs, ws, ids)
            assert zs.ndim==2, zs.shape
            assert ws.ndim==2, ws.shape
            assert ids is None or ids.ndim==1, ids.shape

            if self.z_index is None:
                self.init((zs.shape[-1], ws.shape[-1]))

            zs = zs.astype(np.float32)
            ws = ws.astype(np.float32)

            if ids is None:
                # no ids supplied case
                # generate new unique ids
                n = max(self.id_to_idx, default=-1) + 1
                ids = list(range(n, n+len(zs)))
            else:
                # remove any existing ids
                self.remove([i for i in ids if i in self.id_to_idx])

            self.z_index.add(zs)
            self.w_index.add(ws)

            n = self.z_index.ntotal
            idx = np.arange(n - len(zs), n)

            # map external ID to/from faiss index
            for id_ext, idx_int in zip(ids, idx):
                self.id_to_idx[id_ext] = idx_int
                self.idx_to_id[idx_int] = id_ext

            return ids

        def remove(self, ids:PairIDs):
            """remove points by ID"""
            removed_ids = []
            for i in ids:
                if i not in self.id_to_idx:
                    print(f'WARNING: anguilla: no point with ID {i} in index')
                    continue
                removed_ids.append(i)
                idx = self.id_to_idx[i]
                del self.id_to_idx[i]
                del self.idx_to_id[idx]
                idx = np.array(idx)[None]
                self.z_index.remove_ids(idx)
                self.w_index.remove_ids(idx)
                # faiss shifts its internal index to stay dense
                self.id_to_idx = {
                    k:(v-1 if v > idx else v) for k,v in self.id_to_idx.items()}
                self.idx_to_id = {
                    (k-1 if k > idx else k):v for k,v in self.idx_to_id.items()}
            return removed_ids
        
        def get(self, i:PairID) -> Tuple[Feature, Feature]:
            """get a feature pair by ID"""
            if self.z_index is None or i not in self.id_to_idx:
                print(f'WARNING: anguilla: no point with ID {i} in index')
                return None
            else:
                idx = int(self.id_to_idx[i])
                return (
                    self.z_index.reconstruct(idx), 
                    self.w_index.reconstruct(idx))

        def search(self, 
                z:List[Feature], k:int=None, 
                return_inputs=True,
                return_outputs=True, 
                return_ids=True
                ) -> SearchResult:
            """get outputs(s), distances, and IDs by proximity
            
            Returns:
                zs: [batch, k, input feature]
                ws: [batch, k, output feature]
                ids: [batch, k]
                scores: [batch, k]
            """
            k = k or self.default_k
            z, = np_coerce(z)
            # if z.ndim==1:
                # z = z[None]
                # batch = False
            # elif z.ndim==2:
                # batch = True
            # else:
                # raise ValueError
            if z.ndim!=2: raise ValueError
            # print(f'{batch=}', z.ndim)
            z = z.astype(np.float32) 

            # nearest neighbor search
            scores, idxs = self.z_index.search(z, k)

            # remove -1 ids
            # assuming pattern of missing is same across batch
            # should be, since only reason for missing is <k data points
            b = [i>=0 for i in idxs[0]] 
            scores, idxs = scores[:,b], idxs[:,b]

            if return_inputs:    
                zs = self.z_index.reconstruct_batch(
                    idxs.ravel()).reshape([*idxs.shape, -1])
            else:
                zs = None #if batch else [None]

            if return_outputs:    
                ws = self.w_index.reconstruct_batch(
                    idxs.ravel()).reshape([*idxs.shape, -1])
            else:
                ws = None #if batch else [None]
             
            # map back to ids
            if return_ids:
                ids = np.array([[self.idx_to_id[i] for i in idx] for idx in idxs])
            else:
                ids = None #if batch else [None]
            
            # if not batch:
                # remove batch dim
                # zs, ws, scores, ids = zs[0], ws[0], scores[0], ids[0]

            return SearchResult(zs, ws, ids, scores)
        
        def reset(self):
            if self.z_index is not None:
                pass
                self.z_index.reset()
                self.w_index.reset()
            self.idx_to_id:Dict[int, PairID] = {}
            self.id_to_idx:Dict[PairID, int] = {}

        @property
        def ids(self):
            return self.id_to_idx.keys()

except ImportError:
    class IndexFastL2(Index):
        def __init__(self, *a, **kw):
            raise NotImplementedError("""install faiss for IndexFastL2""")

# class NNSearch(JSONSerializable):
#     """
#     This class is the mid-level interface for neighbor search,
#     providing some common utilities over the Index subclasses.
#     Users will generally use `IML.search` instead of calling `NNSearch` directly.
#     """
#     # TODO: possibly get rid of this class and fold it into IML?
#     #     * currently adds only complexity to the IML implementation
#     #     * but could be useful if needing NNSearch without Embed/Interpolate?
#     def __init__(self, index:Index, k=10):
#         """
#         Args:
#             index: instance of `Index`
#             k: default k-nearest neighbors (but can be overridden later)
#         """
#         super().__init__(index=index, k=k)
#         self.index = index
#         self.default_k = k

#     def __call__(self, feature:Feature, k:int=None) -> Tuple[PairIDs, Scores]:
#         """
#         find the k-nearest neighbors of `feature`
#         Args:
#             feature: query feature vector
#             k: maximum number of neighbors to return
#         Returns:
#             ids: ids of neighbors
#             scores: similarity scores of neighbors (higher is more similar)
#         """
#         k = k or self.default_k
#         return self.index.search(feature, k)
    
#     def distance(self, a:Feature, b:Feature) -> float:
#         """compute distance between two features"""
#         return self.index.metric(a, b)

#     def add(self, feature: Feature, id:Optional[PairID]=None) -> PairID:
#         """add a feature vector to the index and return its ID"""
#         return self.index.add(feature, id)
    
#     def get(self, id:PairID) -> Feature:
#         """look up a feature by ID"""
#         try:
#             return self.index.get(id)
#         except Exception:
#             print(f"NNSearch: WARNING: can't `get` ID {id} which doesn't exist or has been removed")

    
#     def remove(self, id: Union[PairID, PairIDs], batch:bool=False):
#         """
#         Remove point(s) from the index by ID

#         Args:
#             id: id or sequence of ids
#             batch: True if removing a batch of ids, False if a single id.
#         """        
#         if batch:
#             return [self.remove(i) for i in id]
#         else:
#             try:
#                 return self.index.remove(id)
#             except Exception:
#                 print(f"NNSearch: WARNING: can't `remove` ID {id} which doesn't exist or has already been removed")

#     def remove_near(self, feature:Feature, k:int=None) -> PairIDs:
#         """
#         Remove point(s) from the index by proximity.
#         Use k=1 to remove a single point.
#         """
#         # TODO: batching support?
#         k = k or self.default_k
#         ids, _ = self(feature, k=k)
#         self.remove(ids, batch=True)
#         return ids
    
#     def reset(self):
#         """clear all data from the index"""
#         self.index.reset()

#     def __iter__(self):
#         """iterate over IDs in the index"""
#         return iter(self.index.ids)
    
#     def items(self) -> Generator[IDFeaturePair, None, None]:
#         """iterate over ID, Feature pairs"""
#         def iterator():
#             for id in self.index.ids:
#                 yield IDFeaturePair(id, self.index.get(id))
#         return iterator()

