import itertools as it

from .types import *
from . import nnsearch
from . import embed
from . import interpolate
from . import serialize

# TODO: state serialization
# TODO: serialize defaults where possible

class IML(serialize.JSONSerializable):

    def __init__(self, 
            emb:Union[str,embed.Embedding]=None, 
            interp:Union[str,interpolate.Interpolate]=None,
            index:nnsearch.Index=None,
            k:int=10,
            verbose=False):
        """
        Args:
            embed: instance, type or name of Feature subclass (defaults to Identity)
            interp: instance, type or name of Interpolate subclass (defaults to Smooth)
            index: instance of Index (defaults to IndexBrute)
            k: default k-nearest neighbors (can be overridden later)
        """
        self.verbose = verbose
        # Feature converts Inputs to Features
        if emb is None:
            emb = embed.Identity()
        elif isinstance(emb, str):
            emb = getattr(embed, emb)()
        elif isinstance(emb, type) and issubclass(emb, embed.Embedding):
            emb = emb()
        elif isinstance(emb, embed.Embedding):
            pass
        else:
            raise ValueError

         # Interpolate combines a set of Outputs according to their Scores
        if interp is None:
            interp = interpolate.Smooth()
        elif isinstance(interp, str):
            interp = getattr(interpolate, interp)()
        elif isinstance(interp, type) and issubclass(interp, interpolate.Interpolate):
            interp = interp()
        elif isinstance(interp, interpolate.Interpolate):
            pass
        else:
            raise ValueError

        # Index determines the distance metric and efficiency
        if index is None:
            index = nnsearch.IndexBrute(emb.size)
        elif isinstance(index, str):
            index = getattr(nnsearch, index)()
        elif isinstance(index, type) and issubclass(index, nnsearch.Index):
            index = index(emb.size)
        elif isinstance(index, nnsearch.Index):
            pass
        else:
            raise ValueError
        
        super().__init__(
            emb=emb, interp=interp, index=index,
            k=k)

        self.interpolate = interp
        self.embed = emb
        self.neighbors = nnsearch.NNSearch(index, k=k)
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
        # NNSearch converts feature to output IDs and scores
        self.neighbors.reset()

        if res is not None:
            print(f'restoring {len(res.ids)} neighbors')
            for id,inp,out in zip(res.ids, res.inputs, res.outputs):
                self.add(inp,out,id=id)

    def embed_batch(self, inputs:List[Input]):
        if self.embed.is_batched:
            return self.embed(inputs)
        else:
            return [self.embed(input) for input in inputs]

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
        features = self.embed_batch(inputs)

        # TODO: batched `neighbors.add`
        new_ids = []
        if ids is None:
            ids = it.repeat(None)
        for input, feature, id, output in zip(inputs, features, ids, outputs):
            id = self.neighbors.add(feature, id)
            self.pairs[id] = IOPair(input, output)
            new_ids.append(id)

        return new_ids

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
            return self.add_batch((input,), (output,), (id,))[0]
        # if not isinstance(id, str) and hasattr(id, '__len__'):
        #     # refuse any id which is a non-string sequence
        #     # sequences are used for batches of ids
        #     raise ValueError("can't use object with __len__ as a PairID")
        # if self.verbose: print(f'add {input=}, {output=}')
        # feature = self.embed(input)
        # id = self.neighbors.add(feature, id)
        # # track the mapping from output IDs back to outputs
        # self.pairs[id] = IOPair(input, output)
        # return id
    
    def get(self, id:PairID) -> IOPair:
        """
        look up an Input/Output pair by ID

        Args:
            ids: ID to look up.
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

        if not batch:
            ids = (id,)
        for i in ids:
            try:    
                del self.pairs[i]
            except Exception:
                print(f"IML: WARNING: can't `remove` ID {i} which doesn't exist or has already been removed")

        self.neighbors.remove(id, batch=batch)

    def remove_near(self, input:Input, k:int=None) -> PairIDs:
        """
        Remove from mapping by proximity to Input.
        """
        feature = self.embed(input)
        return self.neighbors.remove_near(feature, k=k)
    
    def search_batch(self, inputs:List[Input], k:int=None, from_map=False) -> SearchResult:
        """
        find k-nearest neighbors for each batch item

        Args:
            input: input item
            k: max number of neighbors
            from_map: if True, skip collating inputs and ids and set them to None 

        Returns:
            inputs: neighboring Inputs
            outputs: corresponding Outputs
            ids: ids of Input/Output pairs
            scores: dissimilarity Scores

        Note: neighbor dimension is first
        """
        features = self.embed_batch(inputs)

        if self.neighbors.index.is_batched:
            ids_batch, scores_batch = self.neighbors(features, k=k)
        else:
            # index does not support batching case
            ids_batch = []
            scores_batch = []
            for feature in features:
                ids, scores = self.neighbors(feature, k=k)
                ids_batch.append(ids)
                scores_batch.append(scores)

        inputs_batch = []
        outputs_batch = []

        # get i/o pairs from ids
        # NOTE: bottleneck is here for `map` on large batches
        for feature, ids, scores in zip(features, ids_batch, scores_batch):
            # handle case where there are fewer than k neighbors
            if not len(ids):
                raise RuntimeError('no points in mapping. add some!')
            
            # inputs, outputs = zip(*(self.pairs[i] for i in ids))
            outputs_batch.append([self.pairs[i].output for i in ids])
            if not from_map:
                inputs_batch.append([self.pairs[i].input for i in ids])

        # neighbor dimension goes first
        if from_map:
            inputs, ids = None, None
        else:
            inputs = np.stack(inputs_batch, 1)
            ids = np.stack(ids_batch, 1)
        outputs = np.stack(outputs_batch, 1)
        scores = np.stack(scores_batch, 1)

        return SearchResult(inputs, outputs, ids, scores)

    def search(self, input:Input, k:int=None) -> SearchResult:
        """
        find k-nearest neighbors

        Args:
            input: input item
            k: max number of neighbors

        Returns:
            inputs: neighboring Inputs
            outputs: corresponding Outputs
            ids: ids of Input/Output pairs
            scores: dissimilarity Scores
        """
        feature = self.embed(input)
        ids, scores = self.neighbors(feature, k=k)
        # handle case where there are fewer than k neighbors
        if not len(ids):
            raise RuntimeError('no points in mapping. add some!')
        
        inputs, outputs = zip(*(self.pairs[i] for i in ids))

        # TODO: text-mode visualize scores
        # s = ' '*len(self.pairs)

        return SearchResult(inputs, outputs, ids, scores)
    
    def map_batch(self, inputs:List[Input], k:int=None, **kw):
        """convert a batch of Input to batch of Output using search + interpolate

        Args:
            input: [batch x ...]
            k: max neighbors
            **kw: additional arguments are passed to interpolate

        Returns:
            output instance
        """
        _, outputs, _, scores = self.search_batch(inputs, k, from_map=True)
        return self.interpolate(outputs, scores, **kw)

    def map(self, input:Input, k:int=None, **kw) -> Output:
        """convert an Input to an Output using search + interpolate

        Args:
            input: input
            k: max neighbors
            **kw: additional arguments are passed to interpolate

        Returns:
            output instance
        """
        # print(f'map {input=}')
        _, outputs, _, scores = self.search(input, k)
        return self.interpolate(outputs, scores, **kw)

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

