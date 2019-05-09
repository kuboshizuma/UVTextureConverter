from PIL import Image
import numpy as np
im = Image.open("input/normal.jpg")

from UVTextureConverter import Normal2Atlas
normal_tex = np.array(im)
converter = Normal2Atlas(normal_size=512, atlas_size=200)
atlas_texture = converter.convert(normal_tex)
