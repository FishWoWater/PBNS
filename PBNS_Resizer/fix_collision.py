#!/usr/bin/env python
# coding=utf-8

import numpy as np

def pairwise_distance(A, B):
    rA = np.sum(np.square(A), axis=1)
    rB = np.sum(np.square(B), axis=1)
    distances = - 2*np.matmul(A, np.transpose(B)) + rA[:, np.newaxis] + rB[np.newaxis, :]
    return distances


def find_nearest_neighbour(A, B, dtype=np.int32):
    nearest_neighbour = np.argmin(pairwise_distance(A, B), axis=1)
    return nearest_neighbour.astype(dtype)


def compute_vertex_normals(vertices, faces):
    # Vertex normals weighted by triangle areas:
    # http://www.iquilezles.org/www/articles/normals/normals.htm

    normals = np.zeros(vertices.shape, dtype=vertices.dtype)
    triangles = vertices[faces]

    e1 = triangles[::, 0] - triangles[::, 1]
    e2 = triangles[::, 2] - triangles[::, 1]
    n = np.cross(e2, e1) 

    np.add.at(normals, faces[:,0], n)
    np.add.at(normals, faces[:,1], n)
    np.add.at(normals, faces[:,2], n)

    return normalize(normals)


def normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + 1e-10)

def fix_collisions(vc, vb, fb, eps=0.002):
    """
    Fix the collisions between the clothing and the body by projecting
    the clothing's vertices outside the body's surface
    """

    # Compute body normals
    nb = compute_vertex_normals(vb, fb)

    # For each vertex of the cloth, find the closest vertices in the body's surface
    closest_vertices = find_nearest_neighbour(vc, vb)
    vb = vb[closest_vertices]
    nb = nb[closest_vertices]

    # Test penetrations
    penetrations = np.sum(nb*(vc - vb), axis=1) - eps
    penetrations = np.minimum(penetrations, 0)

    # Fix the clothing
    corrective_offset = -np.multiply(nb, penetrations[:,np.newaxis])
    vc_fixed = vc + corrective_offset

    return vc_fixed


