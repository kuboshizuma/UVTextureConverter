import unittest

import numpy as np
from PIL import Image

from UVTextureConverter import Atlas2Normal, Normal2Atlas, UVConverter

NORMAL_TEXTURE_SAMPLE_PATH = 'input/normal.jpg'
ATLAS_TEXTURE_SAMPLE_PATH = 'input/atlas.png'

HUMAN_IMAGE_SAMPLE_PATH = 'input/human.jpg'
HUMAN_IUV_SAMPLE_PATH = 'input/human_IUV.jpg'

NORMAL_SIZE = 512
ATLAS_SIZE = 200


class Convert(unittest.TestCase):
    def test_convert_from_normal_to_atlas(self):
        normal_tex = np.array(Image.open(NORMAL_TEXTURE_SAMPLE_PATH).convert('RGB'))

        converter = Normal2Atlas(normal_size=NORMAL_SIZE, atlas_size=ATLAS_SIZE)
        atlas_texture = converter.convert(normal_tex)

        atlas_texture_for_display = Normal2Atlas.concat_atlas_tex(atlas_texture)

        self.assertEqual(atlas_texture.shape, (24, ATLAS_SIZE, ATLAS_SIZE, 3))
        self.assertEqual(atlas_texture_for_display.shape, (ATLAS_SIZE * 4, ATLAS_SIZE * 6, 3))

    def test_convert_from_atlas_to_normal(self):
        atlas_tex = np.array(Image.open(ATLAS_TEXTURE_SAMPLE_PATH).convert('RGB')).transpose(1, 0, 2)
        atlas_tex_stack = Atlas2Normal.split_atlas_tex(atlas_tex)

        converter = Atlas2Normal(normal_size=NORMAL_SIZE, atlas_size=ATLAS_SIZE)
        normal_texture = converter.convert(atlas_tex_stack)

        self.assertEqual(atlas_tex_stack.shape, (24, ATLAS_SIZE, ATLAS_SIZE, 3))
        self.assertEqual(normal_texture.shape, (NORMAL_SIZE, NORMAL_SIZE, 3))

    def test_convert_from_iuv_and_image_to_atlas(self):
        tex_trans, mask_trans = UVConverter.create_texture(
            HUMAN_IMAGE_SAMPLE_PATH, HUMAN_IUV_SAMPLE_PATH, parts_size=ATLAS_SIZE, concat=False)

        self.assertEqual(tex_trans.shape, (24, ATLAS_SIZE, ATLAS_SIZE, 3))
        self.assertEqual(mask_trans.shape, (24, ATLAS_SIZE, ATLAS_SIZE))

        # for display
        tex = UVConverter.concat_atlas_tex(tex_trans)
        mask = UVConverter.concat_atlas_tex(mask_trans)

        self.assertEqual(tex.shape, (ATLAS_SIZE * 4, ATLAS_SIZE * 6, 3))
        self.assertEqual(mask.shape, (ATLAS_SIZE * 4, ATLAS_SIZE * 6))


if __name__ == '__main__':
    unittest.main()
