import itertools as it

from .types import *

from . import nnsearch as _nnsearch
from . import embed as _embed
from . import interpolate as _interpolate
from . import serialize
from .nnsearch import Index, IndexFast
from .embed import Embedding, Identity
from .interpolate import Interpolate, Smooth

import importlib.metadata
__version__ = importlib.metadata.version('anguilla')

# # # TODO: state serialization
# # # TODO: serialize defaults where possible

def issubtype(t1, t2):
    return isinstance(t1, type) and issubclass(t1, t2)

def construct(item, module, parent, default, **kw):
    if item is None:
        return default(**kw)
    elif isinstance(item, str):
        return getattr(module, item)(**kw)
    elif issubtype(item, parent):
        return item(**kw)
    elif isinstance(item,parent):
        return item
    else:
        raise ValueError

class IML(serialize.JSONSerializable):
    def __init__(self, 
            embed_input:Union[str,embed.Embedding]=None, 
            embed_output:Union[str,embed.Embedding]=None, 
            interpolate:Union[str,interpolate.Interpolate]=None,
            index:nnsearch.Index=None,
            verbose=False):
        """
        Args:
            embed_input: instance, type or name of Feature subclass 
                (defaults to Identity)
            embed_output: instance, type or name of Feature subclass 
                (defaults to Identity). must be invertible.
            interp: instance, type or name of Interpolate subclass 
                (defaults to Smooth)
            index: instance, type or name of Index subclass 
                (defaults to IndexFast)
        """
        self.verbose = verbose
        # Feature converts Inputs to Features
        self.embed_input = construct(
            embed_input, _embed, Embedding, Identity)
        self.embed_output = construct(
            embed_output, _embed, Embedding, Identity)
        self.interpolate = construct(
            interpolate, _interpolate, Interpolate, Smooth)
        self.index = construct(
            index, _nnsearch, Index, IndexFast)
        
        super().__init__(
            embed_input=self.embed_input, embed_output=self.embed_output, 
            interpolate=self.interpolate, index=self.index)

        self.reset()

    def reset(self, keep_near:Input=None, k:int=None):
        """
        delete all data

        Args:
            keep_near: don't remove the neighbors of this input
            k: number of neighbors for above
        """
        print('reset')
        res = None
        if keep_near is not None and len(self.pairs)>0:
            if len(keep_near)!=len(self.pairs[0][0]):
                print('ERROR: iml: keep_near should be an Input vector')
                keep_near = None
            else:
                print('searching neighbors for keep_near')
                res = self.search(keep_near, k=k)

        self.pairs: Dict[PairID, IOPair] = {}
        
        self.index.reset()

        if res is not None:
            print(f'restoring {len(res.ids)} neighbors')
            for id,inp,out in zip(res.ids, res.inputs, res.outputs):
                self.add(inp,out,id=id)

    def embed_batch(self, embedding, inputs:List[Input], inv=False):
        emb = embedding.inv if inv else embedding
        if embedding.is_batched:
            return emb(inputs)
        else:
            return np.stack([emb(x) for x in inputs], 0)

    def add_batch(self,
            inputs: List[Input], 
            outputs: List[Output], 
            ids: Optional[PairIDs]=None,
            ) -> PairIDs:
        """
        Add a batch of data points to the mapping.
        
        Args:
            input: list of inputs or ndarray with leading batch dimension
            output:  list of inputs or ndarray with leading batch dimension 
            id: list of PairIDs to use.
                if any are an existing id, replace those points.
                if not supplied, ids will be chosen automatically.

        Returns:
            ids: ids of the new data points if you need to reference them later
        """
        zs = self.embed_batch(self.embed_input, inputs)
        ws = self.embed_batch(self.embed_output, outputs)
        # if either is 1d at this point, it should be a batch of scalars
        # convert batches of scalars to batches of vectors
        if zs.ndim==1:
            zs = zs[:,None]
        if ws.ndim==1:
            ws = ws[:,None]
        ids = self.index.add(zs, ws, ids=ids)

        for x,y,i in zip(inputs, outputs, ids):
            i = PairID(i)
            self.pairs[i] = IOPair(x,y)

        return ids

    def add(self, 
            input: Input, 
            output: Output, 
            id: Optional[PairID]=None,
            batch: bool=False
            ) -> PairID:
        """
        Add a data point to the mapping.
        
        Args:
            input: Input item
            output: Output item
            id: PairID to use; if an existing id, replace the point.
                if not supplied, id will be chosen automatically.
            batch: if True, equivalent to `add_batch`

        Returns:
            id: id of the new data point if you need to reference it later
        """
        if batch:
            return self.add_batch(input, output, id)
        else:
            id = id if id is None else (id,)
            return self.add_batch((input,), (output,), id)[0]
    
    def get(self, id:PairID) -> IOPair:
        """
        look up an Input/Output pair by ID

        Args:
            id: ID to look up.
        """
        try:
            return self.pairs[id]
        except Exception:
            print("NNSearch: WARNING: can't `get` ID which doesn't exist or has been removed")

    def remove_batch(self, ids:PairIDs):
        self.remove(ids, batch=True)

    def remove(self, id:PairID, batch:bool=None):
        """
        Remove from mapping by ID(s)

        Args:
            ids: ID or collection of IDs of points to remove from the mapping.
            batch: True if removing a batch of ids, False if a single id.
                will attempt to infer from `id` if not supplied.
        """
        if batch is None:
            batch = not isinstance(id, str) and hasattr(id, '__len__')

        ids = id if batch else (id,)
        for i in ids:
            try:    
                del self.pairs[i]
            except Exception:
                print(f"IML: WARNING: can't `remove` ID {i} which doesn't exist or has already been removed")

        self.index.remove(ids)

    def remove_near(self, input:Input, k:int=None, batch=False) -> PairIDs:
        """
        Remove from mapping by proximity to Input.
        """
        if not batch:
            input = (input,)
        zs = self.embed_batch(self.embed_input, input)
        ids = self.index.remove_near(zs, k=k)
        for i in ids:
            del self.pairs[i]
    
    def map_batch(self, inputs:List[Input], k:int=None, **kw) -> List[Output]:
        """convert a batch of Input to batch of Output using search + interpolate

        Args:
            input: [batch x ...]
            k: max neighbors
            **kw: additional arguments are passed to interpolate

        Returns:
            batch of outputs
        """
        zs = self.embed_batch(self.embed_input, inputs)
        _, ws, _, scores = self.index.search(
            zs, k, return_inputs=False, return_outputs=True, return_ids=False)
        ws = np.moveaxis(ws,0,1)
        scores = np.moveaxis(scores,0,1)
        # print(f'map_batch: {k=} {zs.shape=} {ws.shape=} {scores.shape=}')
        w = self.interpolate(ws, scores, **kw)
        return self.embed_batch(self.embed_output, w, inv=True)

    def map(self, input:Input, k:int=None, **kw) -> Output:
        """convert an Input to an Output using search + interpolate

        Args:
            input: input
            k: max neighbors
            **kw: additional arguments are passed to interpolate

        Returns:
            output
        """
        return self.map_batch((input,), k, **kw)[0]

    def save_state(self):
        """
        return dataset from this IML object.

        Returns:
            state: data in this IML object
        """
        return {
            'pairs': self.pairs
        }
    
    def load_state(self, state):
        """
        load dataset into this IML object.

        Args:
            state: data as obtained from `save_state`
        """
        for id,pair in state['pairs'].items():
            self.add(*pair, id=PairID(id))        

    def save(self, path:str):
        """
        serialize the whole IML object to JSON

        Args:
            path: path to JSON file
        """
        serialize.save(path, self)

    @classmethod
    def load(cls, path):
        """
        deserialize a new IML object from JSON

        Args:
            path: path to JSON file

        Returns:
            new IML instance
        """
        inst = serialize.load(path)
        assert isinstance(inst, cls), type(inst)
        return inst

