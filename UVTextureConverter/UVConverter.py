import numpy as np
from scipy.io  import loadmat
import pickle
from pathlib import Path


class UVConverter(object):
    def __init__(self):
        file_path = Path(__file__).parent.resolve()
        ALP_UV = loadmat(file_path / 'config/UV_Processed.mat')

        self.FaceIndices = np.array( ALP_UV['All_FaceIndices']).squeeze() # (13774,), array([ 1,  1,  1, ..., 24, 24, 24], dtype=uint8)
        self.FacesDensePose = ALP_UV['All_Faces']-1 # (13774, 3)
        self.U_norm = ALP_UV['All_U_norm'].squeeze() # (7829,), array([0.02766012, 0.02547753, 0.2280528 , ..., 0.55260428, 0.54398063, 0.00275962])
        self.V_norm = ALP_UV['All_V_norm'].squeeze()
        self.All_vertices =  ALP_UV['All_vertices'][0] # (7829,), array([ 607,  608,  707, ..., 3494, 3495, 3500], dtype=uint32)

        # 上の説明https://github.com/facebookresearch/DensePose/issues/146
        # Face: たぶん面のこと、3点の組み合わせで1つの面をなしている
        # normal texture側: https://blender.stackexchange.com/questions/49341/how-to-get-the-uv-corresponding-to-a-vertex-via-the-python-api

        # blenderでdumpしたfacesをload
        with open(file_path / 'config/normal_faces.pickle', mode='rb') as f:
            self.normal_faces = pickle.load(f)
        with open(file_path / 'config/normal.pickle', mode='rb') as f:
            self.normal_hash = pickle.load(f)

        self.atlas_hash = {}
        for parts_num in range(1, 25):
            parts_list_ids = self.get_list_ids_by_parts(parts_num)
            for u,v,ver in zip(list(self.U_norm[parts_list_ids]), list(self.V_norm[parts_list_ids]), list(self.All_vertices[parts_list_ids])):
                if (ver-1) in self.atlas_hash:
                    self.atlas_hash[ver-1].append([parts_num, u, v])
                else:
                    self.atlas_hash[ver-1] = [[parts_num, u, v]]

        # atlas_hash # 0~6889 [[u, v], ...]
        # normal_hash # 0~6889 [[parts_id, u, v], ...]

    def get_list_ids_by_parts(self, parts_num):
        """
        parts_num = 1 # 1~24 全部で24、(背景0)
        """

        self.FaceIndicesNow = np.where(self.FaceIndices == parts_num)
        self.FacesNow = self.FacesDensePose[self.FaceIndicesNow] # (1284, 3)
        parts_list_ids = np.unique(self.FacesNow.flatten()) # 値は0~7828の範囲 U_normと、V_normのlistのidと対応、parts_numにあたるlistのid

        return parts_list_ids

    def barycentric_coordinates_exists(self, P0, P1, P2, P):
        """
        三点P0、P1、P2からなる三角形の中に点Pが存在するかどうか
        from: https://github.com/facebookresearch/DensePose/blob/master/detectron/utils/densepose_methods.py
        """
        u = P1 - P0
        v = P2 - P0
        w = P - P0

        vCrossW = np.cross(v,w)
        vCrossU = np.cross(v, u)
        if (np.dot(vCrossW, vCrossU) < 0):
            return False;

        uCrossW = np.cross(u, w)
        uCrossV = np.cross(u, v)

        if (np.dot(uCrossW, uCrossV) < 0):
            return False;

        denom = np.sqrt((uCrossV**2).sum())
        r = np.sqrt((vCrossW**2).sum())/denom
        t = np.sqrt((uCrossW**2).sum())/denom

        return((r <=1) & (t <= 1) & (r + t <= 1))


    def barycentric_coordinates(self, P0, P1, P2, P):
        """
        重心座標系の3点それぞれの係数を算出
        参考: http://zellij.hatenablog.com/entry/20131207/p1
        """
        u = P1 - P0
        v = P2 - P0
        w = P - P0

        vCrossW = np.cross(v,w)
        vCrossU = np.cross(v, u)

        uCrossW = np.cross(u, w)
        uCrossV = np.cross(u, v)

        denom = np.sqrt((uCrossV**2).sum())
        r = np.sqrt((vCrossW**2).sum())/denom
        t = np.sqrt((uCrossW**2).sum())/denom

        return(1-(r+t),r,t)
