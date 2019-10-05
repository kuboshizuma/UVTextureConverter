import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .UVConverter import UVConverter


class Atlas2Normal(UVConverter):
    def __init__(self, atlas_size=200, normal_size=512):
        super().__init__()
        self.atlas_size = atlas_size
        self.normal_size = normal_size
        self.normal_tex = None
        self.normal_ex = None

        self.file_path = Path(__file__).parent.resolve()
        if (self.file_path / f'mapping_relations/atlas2normal_{atlas_size}_{normal_size}.pickle').is_file():
            with open(self.file_path / f'mapping_relations/atlas2normal_{atlas_size}_{normal_size}.pickle', mode='rb') as f:
                self.mapping_relation = pickle.load(f)
        else:
            self.mapping_relation = []

    def convert(self, atlas_tex, mask=None):
        self._mapping(atlas_tex, mask)
        if len(self.mapping_relation) == 0:
            for k in tqdm(range(self.FacesDensePose.shape[0])):
                face = self.FacesDensePose[k]  # 3点からなるfaceの1つの組み合わせ
                face_vertex = [self.All_vertices[face[0]] - 1,
                               self.All_vertices[face[1]] - 1,
                               self.All_vertices[face[2]] - 1]  # vertex番号を取得

                min_index = [0, 0, 0]
                min_val = -1
                for a, lst_a in enumerate(self.normal_hash[face_vertex[0]]):
                    for b, lst_b in enumerate(self.normal_hash[face_vertex[1]]):
                        for c, lst_c in enumerate(self.normal_hash[face_vertex[2]]):
                            total = np.sqrt(((np.array(lst_a) - np.array(lst_b))**2).sum()) + np.sqrt(
                                ((np.array(lst_b) - np.array(lst_c))**2).sum()) + np.sqrt(((np.array(lst_c) - np.array(lst_a))**2).sum())
                            if min_val == -1:
                                min_val = total
                            elif total < min_val:
                                min_val = total
                                min_index = [a, b, c]
                normal_a = self.normal_hash[face_vertex[0]][min_index[0]]  # vertexのnormal UVでの位置を取得
                normal_b = self.normal_hash[face_vertex[1]][min_index[1]]
                normal_c = self.normal_hash[face_vertex[2]][min_index[2]]
                i_min = int(min([normal_a[0], normal_b[0], normal_c[0]]) * self.normal_size)
                i_max = int(max([normal_a[0], normal_b[0], normal_c[0]]) * self.normal_size)
                j_min = int(min([normal_a[1], normal_b[1], normal_c[1]]) * self.normal_size)
                j_max = int(max([normal_a[1], normal_b[1], normal_c[1]]) * self.normal_size)
                for i in range(self.normal_size):
                    if i < i_min or i > i_max:
                        continue
                    for j in range(self.normal_size):
                        if j < j_min or j > j_max:
                            continue
                        ex = self.normal_ex[(self.normal_size - 1) - j, i]
                        if ex == 0:
                            if self.barycentric_coordinates_exists(np.array(normal_a), np.array(normal_b), np.array(
                                    normal_c), np.array([i / (self.normal_size - 1), j / (self.normal_size - 1)])):
                                a, b, c = self.barycentric_coordinates(np.array(normal_a), np.array(normal_b), np.array(
                                    normal_c), np.array([i / (self.normal_size - 1), j / (self.normal_size - 1)]))
                                face_id = self.FaceIndices[k]
                                a_vertex, b_vertex, c_vertex = [], [], []
                                for f in self.atlas_hash[face_vertex[0]]:
                                    if f[0] == face_id:
                                        a_vertex = f[1:]
                                for f in self.atlas_hash[face_vertex[1]]:
                                    if f[0] == face_id:
                                        b_vertex = f[1:]
                                for f in self.atlas_hash[face_vertex[2]]:
                                    if f[0] == face_id:
                                        c_vertex = f[1:]
                                if len(a_vertex) == 0 or len(b_vertex) == 0 or len(c_vertex) == 0:
                                    continue
                                atlas_tex_pos = a * np.array(a_vertex) + b * np.array(b_vertex) + c * np.array(c_vertex)
                                self.mapping_relation.append([(self.normal_size - 1) - j,
                                                              i,
                                                              face_id - 1,
                                                              int(atlas_tex_pos[0] * self.atlas_size),
                                                              (self.atlas_size - 1) - int(atlas_tex_pos[1] * self.atlas_size)])

            with open(self.file_path / f'mapping_relations/atlas2normal_{self.atlas_size}_{self.normal_size}.pickle', mode='wb') as f:
                pickle.dump(self.mapping_relation, f)

        painted_normal_tex = np.copy(self.normal_tex)
        painted_normal_ex = np.copy(self.normal_ex)
        for relation in self.mapping_relation:
            new_tex = atlas_tex[relation[2], relation[3], relation[4]]
            painted_normal_tex[relation[0], relation[1]] = new_tex / 255
            if mask is not None:
                painted_normal_ex[relation[0], relation[1]] = mask[relation[2], relation[3], relation[4]]

        if mask is not None:
            return painted_normal_tex, painted_normal_ex
        else:
            return painted_normal_tex

    def _mapping(self, atlas_tex, mask, return_exist_area=False):
        self.normal_tex, self.normal_ex = self._mapping_atlas_to_normal(atlas_tex, mask)
        if return_exist_area:
            return self.normal_tex, self.normal_ex
        else:
            return self.normal_tex

    def _mapping_atlas_to_normal(self, atlas_tex, mask):
        """
        atlas textureをnormal textureに変換するための関数。

        params:
        atlas_tex: 変換前のatlas texture。
        """
        vertex_tex = {}
        vertex_mask = {}
        _, h, w, _ = atlas_tex.shape

        for k, v in self.atlas_hash.items():
            # 1つのvertexに対して複数候補がある可能性はあるが、textureは同じとみなして最初の候補を使う。
            vertex_tex[k] = atlas_tex[v[0][0] - 1, int(v[0][1] * (h - 1)), (w - 1) - int(v[0][2] * (w - 1))]
            if mask is not None:
                vertex_mask[k] = mask[v[0][0] - 1, int(v[0][1] * (h - 1)), (w - 1) - int(v[0][2] * (w - 1))]

        normal_tex = np.zeros((self.normal_size, self.normal_size, 3))
        normal_tex_exist = np.zeros((self.normal_size, self.normal_size))

        for k, v in self.normal_hash.items():
            for t in v:
                normal_tex[(self.normal_size - 1) - int(t[1] * (self.normal_size - 1)),
                           int(t[0] * (self.normal_size - 1)), :] = vertex_tex[k]
                if mask is not None:
                    normal_tex_exist[(self.normal_size - 1) - int(t[1] * (self.normal_size - 1)),
                                     int(t[0] * (self.normal_size - 1))] = vertex_mask[k]
                else:
                    normal_tex_exist[(self.normal_size - 1) - int(t[1] * (self.normal_size - 1)),
                                     int(t[0] * (self.normal_size - 1))] = 1

        return normal_tex / 255, normal_tex_exist
