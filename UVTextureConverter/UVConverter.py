import pickle
from pathlib import Path

import numpy as np
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
