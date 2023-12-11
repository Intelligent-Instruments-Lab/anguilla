"""
Authors:
  Victor Shepardson
  Intelligent Instruments Lab 2023
"""

import anguilla
from anguilla import IML
from iipyper import OSC, run
from iipyper.types import *
import numpy as np
from time import time
from collections import defaultdict
from typing import Optional

def get_handle(address):
    return ''.join(address.split('/')[3:]).strip('/')

def main(
    osc_port:int=8732,
    osc_return_port:Optional[int]=None,
    osc_host:str='',
    ):
    """
    Args:
        osc_port: listen for OSC controls on this port
        osc_return_port: if supplied, reply on a different port than osc_port
        osc_host: leave this as empty string to get all traffic on the port

    OSC Routes:

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

        /anguilla/reset
            remove all points
        /anguilla/reset "keep_near" ... ["k" k]
            remove all points except the k neighbors of "keep_near"

        /anguilla/load path
            load IML from file at `path`
        /anguilla/save path
            save IML to file at `path`

        an additional segment in a route is the name of an IML instance which
        it will target, e.g. /anguilla/add/myinstance
    """
    osc = OSC(osc_host, osc_port)

    instances = {}
    configs = defaultdict(dict)

    @osc.handle('/anguilla/config*')
    def _(address, **kw):
        k = get_handle(address)

        print(k)
        # TODO: validate input
        configs[k].update(kw)
        print(configs[k]) 

    @osc.handle('/anguilla/add*')
    def _(address, input:Splat[None], output:Splat[None], id:int=None):
        key = get_handle(address)

        # d = len(input)
        # config['feature_size'] = d
        if key not in instances:
            # print(f'new IML object with Input dimension {d}')
            print(f'new IML object with handle "{key}" with config {configs[key]}')
            instances[key] = IML(**configs[key])

        return '/return'+address, instances[key].add(input, output, id=id)
    
    @osc.handle('/anguilla/remove*')
    def _(address, id:int):
        key = get_handle(address)
        if key in instances:
            instances[key].remove(id)
        else:
            print(f'ERROR: anguilla: {address}: no instance "{key}" exists')

    @osc.handle('/anguilla/remove_near*')
    def _(address, input:Splat[None], k:int=None):
        key = get_handle(address)
        if key not in instances:
            print(f'ERROR: anguilla: {address}: no instance "{key}" exists')
            return
        
        instances[key].remove_near(input, k=k)

    @osc.handle('/anguilla/map*', return_port=osc_return_port)
    def _(address, input:Splat[None], k:int=None, **kw):
        key = get_handle(address)
        if key not in instances:
            print(f'ERROR: anguilla: {address}: no instance "{key}" exists')
            print(f'ERROR: anguilla: call {address.replace("map", "add")} at least once before {address}')
            return

        # print(f'{kw=}')
        result = instances[key].map(input, k=k, **kw).tolist()

        return '/return'+address, *result
    
    @osc.handle('/anguilla/reset*')
    def _(address, keep_near:Splat[None]=None, k:int=None):
        key = get_handle(address)
        if key not in instances:
            print(f'ERROR: anguilla: {address}: no instance "{key}" exists')
            return
    
        instances[key].reset(keep_near, k=k)

    @osc.handle('/anguilla/load*')
    def _(address, path:str):
        k = get_handle(address)
        
        assert path.endswith('.json'), \
            "ERROR: anguilla: path should end with .json"
        
        if k=='':
            print(f'loading all IML objects from {path}')
            d = anguilla.serialize.load(path)
            assert isinstance(d, dict)
            print(f'found IML instances: {list(d.keys())}')
            instances.update(d)
        else:
            print(f'load IML object at "{k}" from {path}')
            instances[k] = IML.load(path)

    @osc.handle('/anguilla/save*')
    def _(address, path:str):
        k = get_handle(address)
        if k!='' and k not in instances:
            print(f'ERROR: anguilla: {address}: no instance "{k}" exists')
            return
        
        assert path.endswith('.json'), \
            "ERROR: anguilla: path should end with .json"
        
        if k=='':
            print(f'saving all IML objects to {path}')
            anguilla.serialize.save(path, instances)
        else:
            print(f'saving IML object at "{k}" to {path}')
            instances[k].save(path)

if __name__=='__main__':
    run(main)
