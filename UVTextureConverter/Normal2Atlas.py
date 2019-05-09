import numpy as np
from tqdm import tqdm
import pickle
from pathlib import Path
from .UVConverter import UVConverter


class Normal2Atlas(UVConverter):
    def __init__(self, normal_size=512, atlas_size=200):
        super().__init__()
        self.normal_size = normal_size
        self.atlas_size = atlas_size
        self.atlas_tex = None
        self.atlas_ex = None
        self.file_path = Path(__file__).parent.resolve()
        if (self.file_path / f'mapping_relations/normal2atlas_{normal_size}_{atlas_size}.pickle').is_file():
            with open(self.file_path / f'mapping_relations/normal2atlas_{normal_size}_{atlas_size}.pickle', mode='rb') as f:
                self.mapping_relation = pickle.load(f)
        else:
            self.mapping_relation = []

    def mapping(self, normal_tex):
        self.atlas_tex, self.atlas_ex= self._mapping_normal_to_atlas(normal_tex)
        return self.atlas_tex, self.atlas_ex

    def convert(self, normal_tex):
        if self.atlas_tex is None or self.atlas_ex is None: self.mapping(normal_tex)
        if len(self.mapping_relation)==0:
            for k in tqdm(range(len(self.normal_faces))):
                face_vertex = self.normal_faces[k] # vertex番号を取得

                min_index = [0, 0, 0]
                flag = False
                for a,lst_a in enumerate(self.atlas_hash[face_vertex[0]]):
                    for b,lst_b in enumerate(self.atlas_hash[face_vertex[1]]):
                        for c,lst_c in enumerate(self.atlas_hash[face_vertex[2]]):
                            if lst_a[0]==lst_b[0] and lst_b[0] == lst_c[0]:
                                min_index = [a, b, c]
                                flag = True
                if not flag: continue

                atlas_a = self.atlas_hash[face_vertex[0]][min_index[0]] # vertexのnormal UVでの位置を取得
                atlas_b = self.atlas_hash[face_vertex[1]][min_index[1]]
                atlas_c = self.atlas_hash[face_vertex[2]][min_index[2]]

                i_min = int(min([atlas_a[1], atlas_b[1], atlas_c[1]])*self.atlas_size)
                i_max = int(max([atlas_a[1], atlas_b[1], atlas_c[1]])*self.atlas_size)
                j_min = int(min([atlas_a[2], atlas_b[2], atlas_c[2]])*self.atlas_size)
                j_max = int(max([atlas_a[2], atlas_b[2], atlas_c[2]])*self.atlas_size)

                face_id = atlas_a[0]
                for i in range(self.atlas_size):
                    if i < i_min or i > i_max: continue
                    for j in range(self.atlas_size):
                        if j < j_min or j > j_max: continue
                        ex = self.atlas_ex[face_id-1, i, (self.atlas_size-1)-j]
                        if ex==0:
                            if self.barycentric_coordinates_exists(np.array(atlas_a[1:]), np.array(atlas_b[1:]), np.array(atlas_c[1:]), np.array([i/(self.atlas_size-1), j/(self.atlas_size-1)])):
                                a, b, c = self.barycentric_coordinates(np.array(atlas_a[1:]), np.array(atlas_b[1:]), np.array(atlas_c[1:]), np.array([i/(self.atlas_size-1), j/(self.atlas_size-1)]))
                                a_vertex, b_vertex, c_vertex = [],[],[]

                                min_index = [0,0,0]
                                min_val = -1
                                if len(self.normal_hash[face_vertex[0]])>1 or len(self.normal_hash[face_vertex[1]])>1 or len(self.normal_hash[face_vertex[2]])>1:
                                    for ind_a,lst_a in enumerate(self.normal_hash[face_vertex[0]]):
                                        for ind_b,lst_b in enumerate(self.normal_hash[face_vertex[1]]):
                                            for ind_c,lst_c in enumerate(self.normal_hash[face_vertex[2]]):
                                                total = np.sqrt(((np.array(lst_a)-np.array(lst_b))**2).sum()) + np.sqrt(((np.array(lst_b)-np.array(lst_c))**2).sum()) + np.sqrt(((np.array(lst_c)-np.array(lst_a))**2).sum())
                                                if min_val == -1:
                                                    min_val = total
                                                elif total < min_val:
                                                    min_val = total
                                                    min_index = [ind_a,ind_b,ind_c]
                                a_vertex = self.normal_hash[face_vertex[0]][min_index[0]]
                                b_vertex = self.normal_hash[face_vertex[1]][min_index[1]]
                                c_vertex = self.normal_hash[face_vertex[2]][min_index[2]]

                                if len(a_vertex)== 0 or len(b_vertex)== 0 or len(c_vertex)== 0: continue
                                normal_tex_pos = a*np.array(a_vertex) + b*np.array(b_vertex) + c*np.array(c_vertex)
                                self.mapping_relation.append([i, (self.atlas_size-1)-j, face_id-1, (self.normal_size-1)-int(normal_tex_pos[1]*self.normal_size), int(normal_tex_pos[0]*self.normal_size)])

            with open(self.file_path / f'mapping_relations/normal2atlas_{self.normal_size}_{self.atlas_size}.pickle', mode='wb') as f:
                pickle.dump(self.mapping_relation, f)
        for relation in self.mapping_relation:
            new_tex = normal_tex[relation[3], relation[4]]
            self.atlas_tex[relation[2], relation[0], relation[1]] = new_tex/255

        return self.atlas_tex

    def concat_atlas_tex(self):
        tex = None
        for i in range(0, 4):
            tex_tmp = self.atlas_tex[6*i]
            for i in range(1+6*i, 6+6*i):
                tex_tmp = np.concatenate((tex_tmp, self.atlas_tex[i]), axis=1)
            if tex is None:
                tex = tex_tmp
            else:
                tex = np.concatenate((tex, tex_tmp), axis=0)
        return tex

    def _mapping_to_each_atlas_parts(self, vertex_tex, parts_num):
        """
        normal textureをatlas textureの各パーツごとに変換するための関数。

        params:
        vertex_tex: SMPLモデルの各点ごとのtextureを格納。
        atlas_hash: atlas textureのhash、vertex番号に対応するパーツIDとUV位置が格納されている。
        parts_num: パーツの番号。1~24。
        size: atlas展開するときの各パーツのwidth、heightの値。ただし、width=height。
        """
        tex = np.zeros((self.atlas_size, self.atlas_size, 3))
        tex_ex = np.zeros((self.atlas_size, self.atlas_size))

        for k, v in self.atlas_hash.items():
            for t in v:
                if t[0]==parts_num:
                    tex[int(t[1]*(self.atlas_size-1)), (self.atlas_size-1)-int(t[2]*(self.atlas_size-1)), :]=vertex_tex[k]
                    tex_ex[int(t[1]*(self.atlas_size-1)), (self.atlas_size-1)-int(t[2]*(self.atlas_size-1))] = 1

        return tex/255, tex_ex

    def _mapping_normal_to_atlas(self, normal_tex):
        """
        normal textureをatlas textureに変換するための関数。

        params:
        normal_tex: 変換前のnormal texture。
        normal_hash: normal textureのhash、vertex番号に対応するUV位置が格納されている。
        atlas_hash: atlas textureのhash、vertex番号に対応するパーツIDとUV位置が格納されている。
        size: atlas展開するときの各パーツのwidth、heightの値。ただし、width=height。
        """
        vertex_tex = {}
        h, w, _ = normal_tex.shape

        for k,v in self.normal_hash.items():
            # 1つのvertexに対して複数候補がある可能性はあるが、textureは同じとみなして最初の候補を使う。
            vertex_tex[k] = normal_tex[int(h-v[0][1]*(h-1)), int(v[0][0]*(w-1)), :]

        atlas_texture = np.zeros((24, self.atlas_size, self.atlas_size, 3))
        atlas_ex = np.zeros((24, self.atlas_size, self.atlas_size))
        for i in range(24):
            atlas_texture[i], atlas_ex[i] = self._mapping_to_each_atlas_parts(vertex_tex, parts_num=i+1)

        return atlas_texture, atlas_ex
