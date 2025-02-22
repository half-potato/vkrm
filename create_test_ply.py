import os
import sys
import numpy as np
from plyfile import PlyData, PlyElement

# vertices: N_verts x 3
# colors:   N_tets x 4
# indices:  N_tets x 4
def save_model_ply(path:str, vertices, colors, indices):
	pardir = os.path.dirname(path)
	if not os.path.exists(pardir):
		os.makedirs(pardir)

	vertex_elements = np.empty(vertices.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
	vertex_elements[:] = list(map(tuple, vertices))
	verts = PlyElement.describe(vertex_elements, 'vertex')

	tets = np.array([ (colors[i][0], colors[i][1], colors[i][2], colors[i][3], indices[i].tolist()) for i in range(indices.shape[0]) ],
					dtype=[ ('r', 'f4'), ('g', 'f4'), ('b', 'f4'), ('s', 'f4'), ('vertex_indices', 'O') ])
	tets = PlyElement.describe(tets, 'tetrahedron')

	PlyData([verts, tets]).write(path)


def gen_tets(N):
	verts = []
	cols = []
	tets = []

	for i in range(N):
		for j in range(4):
			v = np.array([ i*2.0, 0.0, 0.0 ], dtype=np.float32)
			if j>0: v[j-1] += 1.0
			verts += [ v ]

		c = np.array([ 0.5, 0.5, 0.5, 40 ], dtype=np.float32)
		c[i%3] = 1.0
		cols += [ c ]

		tets += [ [ i*4 + 0, i*4 + 1, i*4 + 2, i*4 + 3 ] ]

	return verts, cols, tets

def gen_tets_sphere(N, r=1):
	verts = []
	cols = []
	tets = []

	invShape = np.ones(3) / np.array(N + (1,), dtype=np.float32)

	for i in range(N[0]):
		for j in range(N[1]):
			p00 = np.array([i,   j  , 1], dtype=np.float32) * invShape
			p10 = np.array([i+1, j  , 1], dtype=np.float32) * invShape
			p01 = np.array([i,   j+1, 1], dtype=np.float32) * invShape
			p11 = np.array([i+1, j+1, 1], dtype=np.float32) * invShape
			pc  = np.array([i+0.5, j+0.5, 0.75], dtype=np.float32) * invShape

			def spherical_uv_to_cartesian(x):
				cosPhi = np.cos(x[0] * 2 * np.pi)
				sinPhi = np.sin(x[0] * 2 * np.pi)
				cosTheta = np.cos(x[1] * np.pi)
				sinTheta = np.sin(x[1] * np.pi)
				return np.array([
					r * x[2] * sinTheta * cosPhi,
					r * x[2] * cosTheta,
					r * x[2] * sinTheta * sinPhi,
				], dtype=np.float32)

			p00 = spherical_uv_to_cartesian(p00)
			p10 = spherical_uv_to_cartesian(p10)
			p01 = spherical_uv_to_cartesian(p01)
			p11 = spherical_uv_to_cartesian(p11)
			pc  = spherical_uv_to_cartesian(pc)

			v0 = len(verts)
			verts += [ p00, p10, p01, p11, pc ]

			c = np.array([ i*invShape[0], j*invShape[1], 0.5, 40 ], dtype=np.float32)
			cols += [ c ]
			tets += [ [ v0 + 0, v0 + 1, v0 + 2, v0 + 4 ] ]

			c = np.array([ (i+0.5)*invShape[0], (j+0.5)*invShape[1], 0.5, 40 ], dtype=np.float32)
			cols += [ c ]
			tets += [ [ v0 + 1, v0 + 3, v0 + 2, v0 + 4 ] ]

	return np.array(verts, dtype=np.float32), np.array(cols, dtype=np.float32), np.array(tets, dtype=np.float32)

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Usage: python %s [path]" % sys.argv[0])
		exit(1)

	#verts, cols, tets = gen_tets(5)
	verts, cols, tets = gen_tets_sphere((16,8), 1.0)

	#print(verts)

	save_model_ply(sys.argv[1], verts, cols, tets)