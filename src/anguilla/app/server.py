"""
Authors:
  Victor Shepardson
  Intelligent Instruments Lab 2023
"""

import anguilla
from anguilla import IML
from iipyper import OSC, run
from iipyper.types import *
from collections import defaultdict
from typing import Optional

def get_handle(address):
    # return ''.join(address.split('/')[3:]).strip('/')
    s = address.split('/')
    print(s)
    assert s[0]==''
    if len(s)==4: 
        return s[-2]
    elif len(s)==3:
        return 'default'
    else:
        raise ValueError(address, s)

def main(
    osc_port:int=8732,
    osc_return_port:Optional[int]=None,
    osc_host:str='',
    verbose=1,
    ):
    """
    Args:
        osc_port: listen for OSC controls on this port
        osc_return_port: if supplied, reply on a different port than osc_port
        osc_host: leave this as empty string to get all traffic on the port

    OSC Methods:

        /anguilla/config "embed_input" "Identity"
            set input embedding to Identity (the default)
        /anguilla/config "embed_input" "ProjectAndSort"
            set input embedding to ProjectAndSort

        /anguilla/config "interpolate" "Smooth"
            set interpolator to Smooth (the default)
        /anguilla/config "interpolate" "Softmax"
            set interpolator to Softmax
        /anguilla/config "interpolate" "Ripple"
            set interpolator to Ripple

        /anguilla/add "input" ... "output"... 
            add a point to the mapping

        /anguilla/add_batch "input" <json> "output" <json> 
            add a batch of points to the mapping
            each <json> is a JSON string: { "shape": [B, D], "data": [...] }
            where B is the batch size and D is the input dimension,
            and data is a list of numbers of length B*D.

        /anguilla/remove id 
            remove a point from the mapping by ID

        /anguilla/remove_near "input" ... ["k" k]
            remove k points from the mapping by proximity
            here you would replace '...' with a series of numbers
            '[]' denotes optional parts of the message (not OSC lists)

        /anguilla/map "input" ... ["k" k] ["ripple" r] ["temp" t]
            map an input to an output using k neighbors
            "temp" 1 > t > 0 when using Softmax interpolator
            "ripple" r > 0 when using Ripple interpolator
            
        /anguilla/map_batch "input" <json> ["k" k] ["ripple" r] ["temp" t] 
            add a batch of points to the mapping
            <json> is a JSON string: { "shape": [B, D], "data": [...] }
            where B is the batch size and D is the input dimension,
            and data is a list of numbers of length B*D.
            the output is returned as JSON in the same format. 

        /anguilla/reset
            remove all points
        /anguilla/reset "keep_near" ... ["k" k]
            remove all points except the k neighbors of "keep_near"

        /anguilla/load path
            load IML from file at `path`
        /anguilla/save path
            save IML to file at `path`

        all of the above methods accept and additional `instance <name>`
        pair of arguments, where `<name>` refers to an `IML` instance.
        if not provided, the value of `<name>` is the string "default"

    """
    osc = OSC(osc_host, osc_port, verbose=verbose)

    instances = {}
    configs = defaultdict(dict)

    @osc.handle
    def _(address, *a):
        print(address)

    @osc.kwargs('/anguilla/config')
    def _(address, instance='default', **kw):
        # print(k)
        # TODO: validate input
        configs[instance].update(kw)
        if verbose > 0:
            print(f'{configs=}') 

    @osc.handle('/anguilla/add', return_port=osc_return_port)
    def _(address, 
            input:Splat[None], output:Splat[None], id:int=None, 
            instance='default'):
        key = instance

        if key not in instances:
            print(f'new IML object with handle "{key}" with config {configs[key]}')
            instances[key] = IML(**configs[key])

        return '/return'+address, instances[key].add(input, output, id=id)
    
    @osc.handle('/anguilla/add_batch', return_port=osc_return_port)
    def _(address, 
            inputs:NDArray, outputs:NDArray, ids:NDArray=None, 
            instance='default',):
        key = instance

        if key not in instances:
            print(f'new IML object with handle "{key}" with config {configs[key]}')
            instances[key] = IML(**configs[key])

        ids = instances[key].add_batch(inputs, outputs, ids=ids)

        return '/return'+address, *ids
    
    @osc.handle('/anguilla/remove')
    def _(address, id:int, instance='default'):
        key = instance
        if key in instances:
            instances[key].remove(id)
        else:
            print(f'ERROR: anguilla: {address}: no instance "{key}" exists')

    @osc.handle('/anguilla/remove_near')
    def _(address, input:Splat[None], k:int=None, instance='default'):
        key = instance
        if key not in instances:
            print(f'ERROR: anguilla: {address}: no instance "{key}" exists')
            return
        
        instances[key].remove_near(input, k=k)

    @osc.handle('/anguilla/map', return_port=osc_return_port)
    def _(address, input:Splat[None], k:int=None, instance='default', **kw):
        key = instance
        if key not in instances:
            print(f'ERROR: anguilla: {address}: no instance "{key}" exists')
            print(f'ERROR: anguilla: call {address.replace("map", "add")} at least once before {address}')
            return

        result = instances[key].map(input, k=k, **kw).tolist()

        return '/return'+address, *result
    
    @osc.handle('/anguilla/map_batch', return_port=osc_return_port)
    def _(address, inputs:NDArray, k:int=None, instance='default', **kw):
        key = instance
        if key not in instances:
            print(f'ERROR: anguilla: {address}: no instance "{key}" exists')
            print(f'ERROR: anguilla: call {address.replace("map", "add")} at least once before {address}')
            return

        result = ndarray_to_json(instances[key].map_batch(inputs, k=k, **kw))

        return '/return'+address, result
    
    @osc.handle('/anguilla/reset')
    def _(address, keep_near:Splat[None]=None, k:int=None, instance='default'):
        key = instance
        if key not in instances:
            print(f'ERROR: anguilla: {address}: no instance "{key}" exists')
            return
    
        instances[key].reset(keep_near, k=k)

    @osc.handle('/anguilla/load')
    def _(address, path:str, instance=None):
        key = instance
        
        assert path.endswith('.json'), \
            f"ERROR: anguilla {address}: path should end with .json"
        
        if key is None:
            print(f'loading all IML objects from {path}')
            d = anguilla.serialize.load(path)
            assert isinstance(d, dict)
            print(f'found IML instances: {list(d.keys())}')
            instances.update(d)
        else:
            print(f'load IML object at "{key}" from {path}')
            instances[key] = IML.load(path)

    @osc.handle('/anguilla/save')
    def _(address, path:str, instance=None):
        key = instance
        if key is not None and key not in instances:
            print(f'ERROR: anguilla: {address}: no instance "{key}" exists')
            return
        
        assert path.endswith('.json'), \
            f"ERROR: anguilla {address}: path should end with .json"
        
        if key is None:
            print(f'saving all IML objects to {path}')
            anguilla.serialize.save(path, instances)
        else:
            print(f'saving IML object at "{key}" to {path}')
            instances[key].save(path)

if __name__=='__main__':
    run(main)
