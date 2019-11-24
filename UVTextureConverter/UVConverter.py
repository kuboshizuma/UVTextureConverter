import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat


class UVConverter(object):
    def __init__(self):
        file_path = Path(__file__).parent.resolve()
        alp_uv = loadmat(file_path / 'config/UV_Processed.mat')

        # (13774,), array([ 1,  1,  1, ..., 24, 24, 24], dtype=uint8)
        self.FaceIndices = np.array(alp_uv['All_FaceIndices']).squeeze()
        self.FacesDensePose = alp_uv['All_Faces'] - 1  # (13774, 3)
        # (7829,), array([0.02766012, 0.02547753, 0.2280528 , ..., 0.55260428, 0.54398063, 0.00275962])
        self.U_norm = alp_uv['All_U_norm'].squeeze()
        self.V_norm = alp_uv['All_V_norm'].squeeze()
        # (7829,), array([ 607,  608,  707, ..., 3494, 3495, 3500], dtype=uint32)
        self.All_vertices = alp_uv['All_vertices'][0]

        # 上の説明https://github.com/facebookresearch/DensePose/issues/146
        # Face: たぶん面のこと、3点の組み合わせで1つの面をなしている
        # normal texture側:
        # https://blender.stackexchange.com/questions/49341/how-to-get-the-uv-corresponding-to-a-vertex-via-the-python-api

        # blenderでdumpしたfacesをload
        with open(file_path / 'config/normal_faces.pickle', mode='rb') as f:
            self.normal_faces = pickle.load(f)
        with open(file_path / 'config/normal.pickle', mode='rb') as f:
            self.normal_hash = pickle.load(f)

        self.atlas_hash = {}
        for parts_num in range(1, 25):
            parts_list_ids = self.get_list_ids_by_parts(parts_num)
            for u, v, ver in zip(
                list(
                    self.U_norm[parts_list_ids]), list(
                    self.V_norm[parts_list_ids]), list(
                    self.All_vertices[parts_list_ids])):
                if (ver - 1) in self.atlas_hash:
                    self.atlas_hash[ver - 1].append([parts_num, u, v])
                else:
                    self.atlas_hash[ver - 1] = [[parts_num, u, v]]

        # atlas_hash # 0~6889 [[u, v], ...]
        # normal_hash # 0~6889 [[parts_id, u, v], ...]

    def get_list_ids_by_parts(self, parts_num):
        """
        parts_num = 1 # 1~24 全部で24、(背景0)
        """

        self.FaceIndicesNow = np.where(self.FaceIndices == parts_num)
        self.FacesNow = self.FacesDensePose[self.FaceIndicesNow]  # (1284, 3)
        # 値は0~7828の範囲 U_normと、V_normのlistのidと対応、parts_numにあたるlistのid
        parts_list_ids = np.unique(self.FacesNow.flatten())

        return parts_list_ids

    def barycentric_coordinates_exists(self, p0, p1, p2, p):
        """
        三点p0、p1、p2からなる三角形の中に点pが存在するかどうか
        from: https://github.com/facebookresearch/DensePose/blob/master/detectron/utils/densepose_methods.py
        """
        u = p1 - p0
        v = p2 - p0
        w = p - p0

        v_cross_w = np.cross(v, w)
        v_cross_u = np.cross(v, u)
        if (np.dot(v_cross_w, v_cross_u) < 0):
            return False

        u_cross_w = np.cross(u, w)
        u_cross_v = np.cross(u, v)

        if (np.dot(u_cross_w, u_cross_v) < 0):
            return False

        denom = np.sqrt((u_cross_v**2).sum())
        r = np.sqrt((v_cross_w**2).sum()) / denom
        t = np.sqrt((u_cross_w**2).sum()) / denom

        return((r <= 1) & (t <= 1) & (r + t <= 1))

    def barycentric_coordinates(self, p0, p1, p2, p):
        """
        重心座標系の3点それぞれの係数を算出
        参考: http://zellij.hatenablog.com/entry/20131207/p1
        """
        u = p1 - p0
        v = p2 - p0
        w = p - p0

        v_cross_w = np.cross(v, w)
        # v_cross_u = np.cross(v, u)

        u_cross_w = np.cross(u, w)
        u_cross_v = np.cross(u, v)

        denom = np.sqrt((u_cross_v**2).sum())
        r = np.sqrt((v_cross_w**2).sum()) / denom
        t = np.sqrt((u_cross_w**2).sum()) / denom

        return(1 - (r + t), r, t)

    @classmethod
    def concat_atlas_tex(cls, given_tex):
        tex = None
        for i in range(0, 4):
            tex_tmp = given_tex[6 * i]
            for i in range(1 + 6 * i, 6 + 6 * i):
                tex_tmp = np.concatenate((tex_tmp, given_tex[i]), axis=1)
            if tex is None:
                tex = tex_tmp
            else:
                tex = np.concatenate((tex, tex_tmp), axis=0)
        return tex

    @classmethod
    def split_atlas_tex(cls, given_tex):
        h, w, _ = given_tex.shape
        assert int(h / 4) == int(w / 6), 'expected aspect ratio ... height:width = 4:6'
        size = int(h / 4)
        atlas_tex = np.zeros([24, size, size, 3])
        for i in range(4):
            for j in range(6):
                atlas_tex[(6 * i + j), :, :, :] = given_tex[(size * i):(size * i + size), (size * j):(size * j + size), :]
        return atlas_tex

    @classmethod
    def create_smpl_from_images(cls, im, iuv, img_size=200):
        i_id, u_id, v_id = 2, 1, 0
        parts_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        parts_num = len(parts_list)

        # generate parts
        if isinstance(im, str) and isinstance(iuv, str):
            im = Image.open(im)
            iuv = Image.open(iuv)
        elif isinstance(im, type(Image)) and isinstance(iuv, type(Image)):
            im = im
            iuv = iuv
        elif isinstance(im, np.ndarray) and isinstance(iuv, np.ndarray):
            im = Image.fromarray(np.uint8(im * 255))
            iuv = Image.fromarray(np.uint8(iuv * 255))
        else:
            raise ValueError('im and iuv must be str or PIL.Image or np.ndarray.')

        im = (np.array(im) / 255).transpose(2, 1, 0)
        iuv = (np.array(iuv)).transpose(2, 1, 0)

        texture = np.zeros((parts_num, 3, img_size, img_size))
        mask = np.zeros((parts_num, img_size, img_size))
        for j, parts_id in enumerate(parts_list):
            im_gen = np.zeros((3, img_size, img_size))
            im_gen[0][(iuv[v_id][iuv[i_id] == parts_id] / 255 * (img_size - 1)).astype(int),
                      (iuv[u_id][iuv[i_id] == parts_id] / 255 * (img_size - 1)).astype(int)] = im[0][iuv[i_id] == parts_id]
            im_gen[1][(iuv[v_id][iuv[i_id] == parts_id] / 255 * (img_size - 1)).astype(int),
                      (iuv[u_id][iuv[i_id] == parts_id] / 255 * (img_size - 1)).astype(int)] = im[1][iuv[i_id] == parts_id]
            im_gen[2][(iuv[v_id][iuv[i_id] == parts_id] / 255 * (img_size - 1)).astype(int),
                      (iuv[u_id][iuv[i_id] == parts_id] / 255 * (img_size - 1)).astype(int)] = im[2][iuv[i_id] == parts_id]
            texture[j] = im_gen[:, ::-1, :]

            mask[j][(iuv[v_id][iuv[i_id] == parts_id] / 255 * (img_size - 1)).astype(int),
                    (iuv[u_id][iuv[i_id] == parts_id] / 255 * (img_size - 1)).astype(int)] = 1
            mask[j] = mask[j][::-1, :]

        return texture, mask

    @classmethod
    def create_texture(cls, im, iuv, parts_size=200, concat=True):
        tex, mask = cls.create_smpl_from_images(im, iuv, img_size=parts_size)
        tex_trans = np.zeros((24, parts_size, parts_size, 3))
        mask_trans = np.zeros((24, parts_size, parts_size))
        for i in range(tex.shape[0]):
            tex_trans[i] = tex[i].transpose(2, 1, 0)
            mask_trans[i] = mask[i].transpose(1, 0)
        if concat:
            return cls.concat_atlas_tex(tex_trans), cls.concat_atlas_tex(mask_trans)
        else:
            return tex_trans, mask_trans

    @classmethod
    def create_texture_from_video(cls, im_list, iuv_list, parts_size=200, concat=True):
        tex_sum = np.zeros((24, parts_size, parts_size, 3))
        mask_sum = np.zeros((24, parts_size, parts_size))
        for i in range(len(im_list)):
            tex, mask = UVConverter.create_texture(im_list[i], iuv_list[i], parts_size=parts_size, concat=False)
            tex_sum += tex
            mask_sum += mask

        mask = mask_sum + ((mask_sum) == 0)
        tex_res = tex_sum / mask[:, :, :, np.newaxis]
        mask_res = np.where(mask_sum != 0, 1, 0)
        if concat:
            return cls.concat_atlas_tex(tex_res), cls.concat_atlas_tex(mask_res)
        else:
            return tex_res, mask_res
