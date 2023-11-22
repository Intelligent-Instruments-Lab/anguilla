"""
Authors:
  Victor Shepardson
  Intelligent Instruments Lab 2023
"""

import anguilla
from anguilla import IML
from iipyper import OSC, run
import numpy as np
from time import time
from collections import defaultdict
from typing import Optional
import json

def vector_args(a, scalars=None):
    a = list(a)
    kw = defaultdict(list)
    k = None
    while len(a):
        item = a.pop(0)
        # print(type(item), item)
        if isinstance(item, str):
            k = item
        else:
            if k is None:
                print(f'ERROR: anguilla: bad OSC syntax in {a}')
            kw[k].append(item)
    # unwrap scalars
    for item in scalars or []:
        if item in kw:
            kw[item] = kw[item][0]
    return kw

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

        /anguilla/config/emb "Identity"
            set embedding to Identity (the default)
        /anguilla/config/emb "ProjectAndSort"
            set embedding to ProjectAndSort

        /anguilla/config/interp "Smooth"
            set interpolator to Smooth (the default)
        /anguilla/config/interp "Softmax"
            set interpolator to Softmax
        /anguilla/config/interp "Ripple"
            set interpolator to Ripple

        -- or --
        /anguilla/config "emb" ... "interp" ...

        /anguilla/add "input" ... "output"... 
            add a point to the mapping

        /anguilla/remove id 
            remove a point from the mapping by ID

        /anguilla/remove_near "input" ... ["k" k]
            remove k points from the mapping by proximity

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
    """
    osc = OSC(osc_host, osc_port)

    instances = {}
    configs = defaultdict(dict)

    @osc.kwargs('/anguilla/config/*')
    def _(address, **kw):
        k = address.split('config')[-1]
        # TODO: validate input
        configs[k].update(kw)
        print(configs[k]) 

    # @osc.args('/anguilla/config/interp')
    # def _(address, name):
    #     if iml is None:
    #         config['interp'] = name
    #     else:
    #         iml.set_interp(name)

    # @osc.args('/anguilla/config/emb')
    # def _(address, name):
    #     if iml is None:
    #         config['emb'] = name
    #     else:
    #         iml.set_emb(name)

    @osc.args('/anguilla/add*')
    def _(address, *a):
        k = ''.join(address.split('/')[3:]).strip('/')

        kw = vector_args(a)

        if 'input' not in kw:
            print('ERROR: anguilla: no input vector supplied')
            return
        if 'output' not in kw:
            print('ERROR: anguilla: no output vector supplied')
            return

        # d = len(kw['input'])
        # config['feature_size'] = d
        if k not in instances:
            # print(f'new IML object with Input dimension {d}')
            print(f'new IML object with handle "{k}" with config {configs[k]}')
            instances[k] = IML(**configs[k])

        return '/return'+address, instances[k].add(**kw)
    
    @osc.args('/anguilla/remove*')
    def _(address, id):
        k = ''.join(address.split('/')[3:]).strip('/')
        if k in instances:
            instances[k].remove(id)
        else:
            print(f'ERROR: anguilla: {address}: no instance "{k}" exists')

    @osc.args('/anguilla/remove_near*')
    def _(address, *a):
        k = ''.join(address.split('/')[3:]).strip('/')
        if k not in instances:
            print(f'ERROR: anguilla: {address}: no instance "{k}" exists')
            return

        kw = vector_args(a, scalars=['k'])

        if 'input' not in kw:
            print(f'ERROR: anguilla: {address}: no input vector supplied')
            return
        
        instances[k].remove_near(**kw)

    @osc.args('/anguilla/map*', return_port=osc_return_port)
    def _(address, *a):
        k = ''.join(address.split('/')[3:]).strip('/')
        if k not in instances:
            print(f'ERROR: anguilla: {address}: no instance "{k}" exists')
            print('ERROR: anguilla: call /anguilla/add at least once before /map')
            return
    
        kw = vector_args(a, scalars=['k', 'temp', 'ripple'])

        if 'input' not in kw:
            print('ERROR: anguilla: no input vector supplied')
            return
        
        result = instances[k].map(**kw).tolist()

        return '/return'+address, *result
    
    @osc.args('/anguilla/reset*')
    def _(address, *a):
        k = ''.join(address.split('/')[3:]).strip('/')
        if k not in instances:
            print(f'ERROR: anguilla: {address}: no instance "{k}" exists')
            return
    
        kw = vector_args(a, scalars=['k'])

        instances[k].reset(**kw)

    @osc.args('/anguilla/load*')
    def _(address, path):
        k = ''.join(address.split('/')[3:]).strip('/')
        assert isinstance(path, str)
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

    @osc.args('/anguilla/save*')
    def _(address, path):
        k = ''.join(address.split('/')[3:]).strip('/')
        if k!='' and k not in instances:
            print(f'ERROR: anguilla: {address}: no instance "{k}" exists')
            return
        
        assert isinstance(path, str)
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
