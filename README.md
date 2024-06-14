# anguilla ([Documentation](https://intelligent-instruments-lab.github.io/anguilla/))

`anguilla` is a mapping and interactive machine learning package for digital musical instrument design in Python.

This is an early stage project. Currently, the main interface is the `IML` class, which allows adding input-output pairs (`IML.add`) and subsequently mapping points in the input space to outputs (`IML.map`). 

`anguilla` is designed to be modular and hackable. An `IML` object is composed of several exchangeable parts:

* an `Embedding` embeds input points into a feature space
* an `NNSearch` implements nearest-neighbor search in the feature space
* an `Interpolate` combines a set of output points using the distances of their corresponding input points from a neighboring query point.

`anguilla server` will expose the Python API over [Open Sound Control](https://en.wikipedia.org/wiki/Open_Sound_Control) (OSC) using [iipyper](https://github.com/intelligent-instruments-lab/iipyper).

For examples and tutorials of how to use `anguilla`, see our [examples repo](https://github.com/intelligent-instruments-lab/iil-examples) (TBC).

## Install

`anguilla` can be installed from [PyPI](https://pypi.org/project/anguilla-iml):

```sh
pip install anguilla-iml
```

### with pytorch

If you encounter an incompatibility between the PyPI versions of pytorch and faiss-cpu, try installing them both from conda (`conda install -c pytorch pytorch faiss-cpu`) before `pip install anguilla-iml`.

## Develop

See the [iil-dev](https://github.com/Intelligent-Instruments-Lab/iil-dev) repo for a recommended dev environment. 

It's also possible to develop `anguilla` in isolation. You will need [Poetry](https://python-poetry.org/) and your Python environment manager of choice. With `conda`, for example:

```sh
conda create -n anguilla-env python=3.10 poetry
conda activate anguilla-env
```

then:

```sh
git clone git@github.com:Intelligent-Instruments-Lab/anguilla.git
cd anguilla
poetry install
```

## Contact

`anguilla` is developed by the [Intelligent Instruments Lab](https://iil.is/about). Get in touch to [collaborate](https://iil.is/collaborate):

 ◦ <a href="https://iil.is" target="_blank" rel="noopener" title="Intelligent Instrumets Lab">iil.is</a> ◦ 
<a href="https://facebook.com/intelligentinstrumentslab" target="_blank" rel="noopener" title="facebook.com">Facebook</a> ◦ 
<a href="https://instagram.com/intelligentinstruments" target="_blank" rel="noopener" title="instagram.com">Instagram</a> ◦ 
<a href="https://x.com/_iil_is" target="_blank" rel="noopener" title="x.com">X (Twitter)</a> ◦ 
<a href="https://youtube.com/@IntelligentInstruments" target="_blank" rel="noopener" title="youtube.com">YouTube</a> ◦ 
<a href="https://discord.gg/fY9GYMebtJ" target="_blank" rel="noopener" title="discord.gg">Discord</a> ◦ 
<a href="https://github.com/intelligent-instruments-lab" target="_blank" rel="noopener" title="github.com">GitHub</a> ◦ 
<a href="https://www.linkedin.com/company/intelligent-instruments-lab" target="_blank" rel="noopener" title="www.linkedin.com">LinkedIn</a> ◦ 
<a href="mailto:iil@lhi.is" target="_blank" rel="noopener" title="">Email</a> ◦ 

## Funding

The Intelligent Instruments project (INTENT) is funded by the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (Grant agreement No. 101001848).
