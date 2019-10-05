# UVTextureConverter

## 1. Install

```
$ pip install UVTextureConverter
```

## 2. How To Use

### 2.1. Quick Use

You can check details in notebooks (2.2).

#### nomal texture -> atlas texture

```
$ python
>>
from UVTextureConverter import Normal2Atlas
from PIL import Image
import numpy as np

normal_tex = np.array(Image.open('input/normal.jpg'))
converter = Normal2Atlas(normal_size=512, atlas_size=200)
atlas_texture = converter.convert(normal_tex)
```

#### atlas texture -> normal texture

```
$ python
>>
from UVTextureConverter import Atlas2Normal
from PIL import Image
import numpy as np

im = np.array(Image.open('input/atlas.png').convert('RGB')).transpose(1, 0, 2)
size = 200
atlas_tex  = np.zeros([24,size,size,3])
for i in range(4):
    for j in range(6):
        atlas_tex[(6*i+j) , :,:,:] = im[(size*i):(size*i+size),(size*j):(size*j+size),: ]
converter = Atlas2Normal(atlas_size=200, normal_size=512)
normal_tex = converter.convert(atlas_tex)
```

### 2.2. Notebooks

See the following notebooks.

- [convert_demo.ipynb](https://github.com/kuboshizuma/UVTextureConverter/blob/master/notebook/convert_demo.ipynb): how to convert from atlas texture to normal texture, and vice versa.
- [densepose_convert_demo.ipynb](https://github.com/kuboshizuma/UVTextureConverter/blob/master/notebook/densepose_convert_demo.ipynb): how to convert a single rgb image to atlas texute by densepose and convert to atlas texture.

## 3. developers install

### 3.1. docker build & run

```
$ docker-compose build
$ docker-compose run --rm uvtexture_module
```

### 3.2. notebook usage

```
$ docker-compose run --rm -p 8888:8888 uvtexture_module
$ jupyter notebook --port 8888 --ip=0.0.0.0 --allow-root (in docker)
```
