import sys
import numpy as np
from ply_model_helper import load_model_ply, save_model_ply
from tqdm import tqdm

def MortonCode64(vertices):
    def SplitBy3Bits21(x):
        r = x.astype(np.uint64)
        r = (r | r << 32) & 0x1f00000000ffff
        r = (r | r << 16) & 0x1f0000ff0000ff
        r = (r | r << 8) & 0x100f00f00f00f00f
        r = (r | r << 4) & 0x10c30c30c30c30c3
        r = (r | r << 2) & 0x1249249249249249
        return r

    mn = np.min(vertices, axis=0)
    mx = np.max(vertices, axis=0)
    center = (mn + mx) / 2
    extent = (mx - mn) / 2
    vn = (vertices - center) / extent
    p = np.clip(vn * 1024, -(1 << 20), (1 << 20) - 1).astype(np.int32)
    #assert( (-(1 << 20) <= p).all() and (p < (1 << 20)).all() )

    # move sign bit to bit 20
    x = ((p[:,0] & 0x80000000) >> 11) | (p[:,0] & 0x0fffff)
    y = ((p[:,1] & 0x80000000) >> 11) | (p[:,1] & 0x0fffff)
    z = ((p[:,2] & 0x80000000) >> 11) | (p[:,2] & 0x0fffff)

    data = SplitBy3Bits21(x) | (SplitBy3Bits21(y) << 1) | (SplitBy3Bits21(z) << 2)

    # invert sign bits
    return data ^ 0x7000000000000000

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python %s [path] [output]" % sys.argv[0])
        exit(1)

    verts, cols, tets = load_model_ply(sys.argv[1])

    print("Calculating morton codes...")
    morton_codes = MortonCode64(verts)

    print("Sorting...")
    index_map = np.argsort(morton_codes)
    #index_map = np.arange(verts.shape[0])

    tmp = np.copy(verts)
    verts[index_map] = tmp

    for i in tqdm(range(tets.shape[0]), desc="Remapping indices"):
        tets[i,0] = index_map[ tets[i,0] ]
        tets[i,1] = index_map[ tets[i,1] ]
        tets[i,2] = index_map[ tets[i,2] ]
        tets[i,3] = index_map[ tets[i,3] ]

    print("Writing PLY...")
    save_model_ply(sys.argv[2], verts, cols, tets)