import os
import numpy as np
from plyfile import PlyData, PlyElement
from tqdm import tqdm

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

	tets = np.empty((indices.shape[0],), dtype=[('r', 'f4'), ('g', 'f4'), ('b', 'f4'), ('s', 'f4'), ('vertex_indices', 'O')])
	tets['r'], tets['g'], tets['b'], tets['s'] = colors.T
	tets['vertex_indices'] = list(indices)

	tets = PlyElement.describe(tets, 'tetrahedron')

	PlyData([verts, tets]).write(path)

def load_model_ply(path:str):
	ply = PlyData.read(path, 'c', {'tetrahedron': {'vertex_indices': 4}})

	vertexElements = ply['vertex']
	tetElement     = ply['tetrahedron']

	vertices = np.empty((len(vertexElements), 3), dtype=np.float32)
	indices  = np.empty((len(tetElement),     4), dtype=np.uint32)
	colors   = np.empty((len(tetElement),     4), dtype=np.float32)

	vertices[:,0] = vertexElements['x']
	vertices[:,1] = vertexElements['y']
	vertices[:,2] = vertexElements['z']

	colors[:,0] = tetElement['r']
	colors[:,1] = tetElement['g']
	colors[:,2] = tetElement['b']
	colors[:,3] = tetElement['s']

	indices[:] = tetElement['vertex_indices']

	return vertices, colors, indices