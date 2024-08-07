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
    s = address.split('/')
    # print(s)
    assert s[0]==''
    assert s[1]=='anguilla'
    return s[2]

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

        /anguilla/instance/config "embed_input" "Identity"
            set input embedding to Identity (the default)
        /anguilla/instance/config "embed_input" "ProjectAndSort"
            set input embedding to ProjectAndSort

        /anguilla/instance/config "interpolate" "Smooth"
            set interpolator to Smooth (the default)
        /anguilla/instance/config "interpolate" "Softmax"
            set interpolator to Softmax
        /anguilla/instance/config "interpolate" "Ripple"
            set interpolator to Ripple

        /anguilla/instance/add "input" ... "output"... 
            add a point to the mapping

        /anguilla/instance/add_batch "inputs" <json> "outputs" <json> 
            add a batch of points to the mapping
            each <json> is a JSON string: { "shape": [B, D], "data": [...] }
            where B is the batch size and D is the input dimension,
            and data is a list of numbers of length B*D.

        /anguilla/instance/remove id 
            remove a point from the mapping by ID

        /anguilla/instance/remove_near "input" ... ["k" k]
            remove k points from the mapping by proximity
            here you would replace '...' with a series of numbers
            '[]' denotes optional parts of the message (not OSC lists)

        /anguilla/instance/map "input" ... ["k" k] ["ripple" r] ["temp" t]
            map an input to an output using k neighbors
            "temp" 1 > t > 0 when using Softmax interpolator
            "ripple" r > 0 when using Ripple interpolator
            
        /anguilla/instance/map_batch "inputs" <json> ["k" k] ["ripple" r] ["temp" t] 
            add a batch of points to the mapping
            <json> is a JSON string: { "shape": [B, D], "data": [...] }
            where B is the batch size and D is the input dimension,
            and data is a list of numbers of length B*D.
            the output is returned as JSON in the same format. 

        /anguilla/instance/reset
            remove all points
        /anguilla/instance/reset "keep_near" ... ["k" k]
            remove all points except the k neighbors of "keep_near"

        /anguilla/instance/load path
            load IML from file at `path`
        /anguilla/instance/save path
            save IML to file at `path`

        replace 'instance' with any string to address a unique IML instance.
    """
    osc = OSC(osc_host, osc_port, verbose=verbose)

    if verbose:
        print(main.__doc__)

    instances = {}
    configs = defaultdict(dict)
    
    def get_instance(address, create=True):
        key = get_handle(address)
        if key not in instances:
            if create:
                print(f'{address} new IML object with handle "{key}" with config {configs[key]}')
                instances[key] = IML(**configs[key])
            else:
                print(f'ERROR: {address}: no instance "{key}" exists')
                return None
        return instances[key]

    @osc.handle
    def _(address, *a):
        print(address)

    @osc.kwargs('/anguilla/*/config')
    def _(address, **kw):
        key = get_handle(address)
        # print(k)
        # TODO: validate input
        configs[key].update(kw)
        if verbose > 0:
            print(f'{configs=}') 

        '/return'+address, str(configs[key])

    @osc.handle('/anguilla/*/add', return_port=osc_return_port)
    def _(address, input:Splat[None], output:Splat[None], id:int=None):
        iml = get_instance(address)
        return '/return'+address, iml.add(input, output, id=id)
     
    @osc.handle('/anguilla/*/add_batch', return_port=osc_return_port)
    def _(address, inputs:NDArray, outputs:NDArray, ids:NDArray=None):
        iml = get_instance(address)
        ids = iml.add_batch(inputs, outputs, ids=ids)
        return '/return'+address, *ids
    
    @osc.handle('/anguilla/*/remove')
    def _(address, id:int):
        iml = get_instance(address, create=False)
        if iml is not None:
            iml.remove(id)

    @osc.handle('/anguilla/*/remove_near')
    def _(address, input:Splat[None], k:int=None):
        iml = get_instance(address, create=False)
        if iml is not None:
            iml.remove_near(input, k=k)

    @osc.handle('/anguilla/*/map', return_port=osc_return_port)
    def _(address, input:Splat[None], k:int=None, **kw):
        iml = get_instance(address, create=False)
        if iml is not None:
            return '/return'+address, *iml.map(input, k=k, **kw).tolist()
        else:
            print(f'ERROR: anguilla: call {address.replace("map", "add")} at least once before {address}')

    @osc.handle('/anguilla/*/map_batch', return_port=osc_return_port)
    def _(address, inputs:NDArray, k:int=None, **kw):
        iml = get_instance(address, create=False)
        if iml is not None:        
            return '/return'+address, ndarray_to_json(iml.map_batch(
                inputs, k=k, **kw))
        else:
            print(f'ERROR: anguilla: call {address.replace("map", "add")} at least once before {address}')
    
    @osc.handle('/anguilla/*/reset')
    def _(address, keep_near:Splat[None]=None, k:int=None):
        iml = get_instance(address, create=False)
        if iml is not None:
            iml.reset(keep_near, k=k)
    
    @osc.handle('/anguilla/*/load')
    def _(address, path:str):
        key = get_handle(address)
        
        assert path.endswith('.json'), \
            f"ERROR: anguilla {address}: path should end with .json"
        
        print(f'load IML object at "{key}" from {path}')
        instances[key] = IML.load(path)

    @osc.handle('/anguilla/*/save')
    def _(address, path:str):
        iml = get_instance(address, create=False)
        
        assert path.endswith('.json'), \
            f"ERROR: anguilla {address}: path should end with .json"
        
        if iml is not None:
            print(f'saving IML object at "{get_handle(address)}" to {path}')
            iml.save(path)

    ## TODO: add load_all, save_all?

    @osc.handle('/anguilla/*/seq_input', return_port=osc_return_port)
    def _(address, input:Splat[None]):
        iml = get_instance(address)
        iml.seq_input(input)

    @osc.handle('/anguilla/*/seq_output', return_port=osc_return_port)
    def _(address, output:Splat[None]):
        iml = get_instance(address)
        iml.seq_output(output)

    @osc.handle('/anguilla/*/seq_end', return_port=osc_return_port)
    def _(address, n=100):
        iml = get_instance(address)
        return '/return'+address, iml.seq_end(n)
    
    @osc.handle('/anguilla/*/seq_reset', return_port=osc_return_port)
    def _(address):
        iml = get_instance(address)
        return '/return'+address, iml.seq_reset()

if __name__=='__main__':
    run(main)
