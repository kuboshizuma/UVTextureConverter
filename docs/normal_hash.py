# blenderを使ってnormal_hashを用意する。 / vertexに対するuvを算出しdumpする。

import pickle

import bmesh
import bpy

obj = bpy.context.edit_object
me = obj.data
bm = bmesh.from_edit_mesh(me)


def uv_from_vert_list(uv_layer, v):
    uvs = []
    for l in v.link_loops:
        uv_data = l[uv_layer]
        uvs.append(uv_data.uv)
    return uvs


def get_unique_list(seq):
    seen = []
    return [x for x in seq if x not in seen and not seen.append(x)]


# Example using the functions above
uv_layer = bm.loops.layers.uv.active

output_hash = {}
count = 0
for v in bm.verts:
    uvs = uv_from_vert_list(uv_layer, v)
    print('Vertex: %r' % v.index)
    uv_points = []
    for uv in uvs:
        uv_points.append(list(uv))
    count += len(get_unique_list(list(uv_points)))
    output_hash[v.index] = get_unique_list(uv_points)

with open('/Users/seishin/python/playground/texture/normal.pickle', mode='wb') as f:
    pickle.dump(output_hash, f)

"""
# python側: blenderでdumpしたhashをload

import pickle

with open('/Users/seishin/python/playground/texture/normal.pickle', mode='rb') as f:
    normal_hash = pickle.load(f)
"""
