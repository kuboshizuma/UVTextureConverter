# UVTextureConverter

This library is converter to convert atlas texuture (defined in Densepose) to normal texture (defined in SMPL), and vice versa.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/uvtextureconverter.svg)](https://badge.fury.io/py/uvtextureconverter)
[![Python Versions](https://img.shields.io/pypi/pyversions/uvtextureconverter.svg)](https://pypi.org/project/uvtextureconverter/)
[![Downloads](https://pepy.tech/badge/uvtextureconverter)](https://pepy.tech/project/uvtextureconverter)
[![Test@master](https://github.com/kuboshizuma/UVTextureConverter/actions/workflows/test_master.yml/badge.svg?branch=master)](https://github.com/kuboshizuma/UVTextureConverter/actions/workflows/test_master.yml)

## 1. Install

```
$ pip install UVTextureConverter
(or $ pip install uvtextureconverter)
```

## 2. How To Use

### 2.1. Quick Use

#### nomal texture -> atlas texture

```python
from UVTextureConverter import Normal2Atlas
from PIL import Image
import numpy as np

normal_tex = np.array(Image.open('input/normal.jpg'))
converter = Normal2Atlas(normal_size=512, atlas_size=200)
atlas_tex = converter.convert(normal_tex)
```

#### atlas texture -> normal texture

```python
from UVTextureConverter import Atlas2Normal
from PIL import Image
import numpy as np

im = np.array(Image.open('input/atlas.png').convert('RGB')).transpose(1, 0, 2)
atlas_tex_stack = Atlas2Normal.split_atlas_tex(im)
converter = Atlas2Normal(atlas_size=200, normal_size=512)
normal_tex = converter.convert(atlas_tex_stack)
```

### 2.2. Notebooks (for details and examples)

See the following notebooks for details and examples.

- [convert_texture_between_normal_and_altas.ipynb](notebook/convert_texture_between_normal_and_altas.ipynb): how to convert from atlas texture to normal texture, and vice versa.
- [create_uv_texture_from_image_by_using_densepose.ipynb](notebook/create_uv_texture_from_image_by_using_densepose.ipynb): how to convert a single rgb image to atlas texute by densepose and convert to normal texture.
- [create_uv_texture_from_video_by_using_densepose.ipynb](notebook/create_uv_texture_from_video_by_using_densepose.ipynb): how to convert video to atlas texture.

## 3. developers install

### 3.1. docker build & run

```
$ docker-compose build
$ docker-compose run --rm uvtex
```

### 3.2. notebook usage

```
$ export UVTEX_PORT=8888
$ docker-compose run --rm -p $UVTEX_PORT:$UVTEX_PORT uvtex jupyter notebook --port $UVTEX_PORT --ip=0.0.0.0 --allow-root
```

### 3.3. flake8 & test

```
$ docker-compose run --rm uvtex
(in docker)
$ flake8
$ python -m unittest
```
