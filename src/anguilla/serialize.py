import json
import io
import base64
from typing import Any
from copy import deepcopy

import numpy as np

class JSONSerializable:
    """JSON serialization for Python classes.
    Saves keyword arguments at construction,
    and also any state returned by the `save_state` method.
    Does *not* attempt to serialize code or any other attributes.

    to make class a serializable, subclass JSONSerializable, 
    and in the constructor use e.g. `super().__init__(a=0, b=1 ...)`
    with any keyword args which should be serialized.

    override `save_state` and `load_state` to handle any mutable state.

    Constructor args and return values of `save_state` can be other JSONSerializable objects, or any types which are already serializable by
    the standard library `json` module.
    """
    def __init__(self, **kw):
        self._kw = deepcopy(kw)
        self._kw['__inst__'] = '.'.join((
            self.__class__.__module__,
            self.__class__.__name__))

    def _store(self):
        return {'__state__': self.save_state(), **self._kw}

    def save_state(self):
        """return object state in JSON serializable form"""
        return None

    def load_state(self, state):
        """restore from de-serialized state"""
        pass

def torch_serializable(o):
    if o.__class__.__module__=='torch':
        return True
    
    try:
        import torch
        if isinstance(o, torch.Tensor):
            return True
    except ImportError:
        pass

    return False

def np_serializable(o):
    if o.__class__.__module__=='numpy':
        return True
    
    if isinstance(o, np.array):
        return True
    
    return False

class JSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, JSONSerializable):
            # instance of JSONSerializable
            return o._store()
        elif isinstance(o, type):
            # type 
            return {'__type__':'.'.join((o.__module__, o.__name__))}
        elif torch_serializable(o) or np_serializable(o):
            # try to serialize with numpy or torch + base85
            buf = io.BytesIO()

            if torch_serializable(o):
                import torch
                torch.save(buf, o)
                k = '__pt__'
            else:
                np.save(buf, o)
                k = '__npy__'
            return {k:base64.b85encode(buf.getvalue())}
        
        else:
            return super().default(o)
        
class JSONDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(
            self, object_hook=self.object_hook, *args, **kwargs)
    def object_hook(self, d):
        if '__inst__' in d:
            cls = get_cls(d.pop('__inst__'))
            state = d.pop('__state__')
            inst = cls(**d)
            inst.load_state(state)
            return inst
        elif '__type__' in d:
            assert len(d)==1, d
            cls = get_cls(d['__type__'])
            return cls
        elif '__pt__' in d:
            import torch
            buf = io.BytesIO(base64.base85decode(d['__pt__']))
            return torch.load(buf)
        elif '__npy__' in d:
            buf = io.BytesIO(base64.base85decode(d['__npy__']))
            return np.load(buf)

        return d
    
def get_cls(s):
    """convert a type name to a type, importing dependencies as needed
     
    e.g. "somepackage.submodule.SomeClass" -> SomeClass 

    this works by doing:
    import somepackage
    import somepackage.submodule.SomeClass
    eval("somepackage.submodule.SomeClass")

    this should work in most cases, but could possibly break if the package containing the type has a weird structure.

    In cases where dependencies rename / reorganize their types, breaking old
    anguilla files, any translation of type names can be done here.
    """

    # sanitize inputs
    assert all(item.isidentifier() for item in s.split('.')), s
    parts = s.split('.')

    ### backward compat translations ###
    if parts[0]=='iml': parts[0] = 'anguilla'

    pkg = parts[0]
    mod = '.'.join(parts[:-1])
    # import top level package the type belongs to
    exec(f'import {pkg}') 
    # import submodule the type is in
    exec(f'import {mod}')
    # convert string to type
    return eval(s)

    
def load(path):
    with open(path, 'r') as f:
        return json.load(f, cls=JSONDecoder)        
    
def save(path, obj):
    with open(path, 'w') as f:
        return json.dump(obj, f, cls=JSONEncoder)
