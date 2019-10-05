# blenderを使ってnormal_facesを取得する。

import pickle

import bmesh
import bpy

obj = bpy.context.edit_object
me = obj.data
bm = bmesh.from_edit_mesh(me)

faces = []
for f in bm.faces:
    face = []
    for v in f.verts:
        face.append(v.index)
    faces.append(face)

with open('/Users/seishin/python/playground/texture/normal_faces.pickle', mode='wb') as f:
    pickle.dump(faces, f)

"""
# python: blenderでdumpしたfacesをload

import pickle

with open('/Users/seishin/python/playground/texture/normal_faces.pickle', mode='rb') as f:
    normal_faces = pickle.load(f)
"""
