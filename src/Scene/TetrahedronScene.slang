#pragma once

namespace vkDelTet {

struct TetrahedronScene {
    static const uint3 kTetTriangles[4] = {
        /* 
        *    y
        *    ^
        *    |
        *    *v2    z
        *    |     /
        *    |    *v3
        *    |   /
        *    |  /
        *    | /          v1
        *    v0------------*--->x
        *                 
        */
        uint3(0, 2, 1),
        uint3(1, 2, 3),
        uint3(0, 3, 2),
        uint3(3, 0, 1),
    };

    ByteAddressBuffer        vertices;
    StructuredBuffer<float4> colors;
    ByteAddressBuffer        indices;
	StructuredBuffer<float4> spheres;

	float3 aabbMin;
	uint   numTets;
    float3 aabbMax;
    float  densityScale;

    float3 load_vertex(uint vertexId) {
        return vertices.Load<float3>(vertexId * sizeof(float3));
    }

    float3 load_vertex(uint tetId, uint tetVertexId) {
        uint vertexId = indices.Load((tetId * 4 + tetVertexId) * sizeof(uint));
        return load_vertex(vertexId);
    }
    
    uint4 load_tet_indices(uint tetId) {
        return indices.Load4(tetId * sizeof(uint4));
    }

    float4x3 load_tet_vertices(uint4 tetIndices) {
        float4x3 v;
        v[0] = load_vertex(tetIndices[0]);
        v[1] = load_vertex(tetIndices[1]);
        v[2] = load_vertex(tetIndices[2]);
        v[3] = load_vertex(tetIndices[3]);
        return v;
	}

    float4x3 load_tet_vertices(uint tetId) { 
        return load_tet_vertices(load_tet_indices(tetId));
    }
    
    float4 load_rgb_density(uint tetId) {
        float4 c = colors[tetId];
        c.a *= densityScale;
        return c;
	}

};

bool intersect_ray_tetrahedron(float3 orig, float3 dir, float4x3 verts, out float2 t) {
    t = -1;

    // --------------------------------------------------------------------
    // 1. Compute plane equations for the 4 faces (outward normals)
    //    n[i].x, n[i].y, n[i].z, and d[i]
    // --------------------------------------------------------------------

    // Face 0: (verts[0], verts[1], verts[2])
    float3 n0 = cross(verts[1] - verts[0], verts[2] - verts[0]);
    if (dot(n0, verts[3] - verts[0]) > 0) {
        n0 = -n0;
    }
    float d0 = -dot(n0, verts[0]);

    // Face 1: (verts[0], verts[1], verts[3])
    float3 n1 = cross(verts[1] - verts[0], verts[3] - verts[0]);
    if (dot(n1, verts[2] - verts[0]) > 0) {
        n1 = -n1;
    }
    float d1 = -dot(n1, verts[0]);

    // Face 2: (verts[0], verts[2], verts[3])
    float3 n2 = cross(verts[2] - verts[0], verts[3] - verts[0]);
    if (dot(n2, verts[1] - verts[0]) > 0) {
        n2 = -n2;
    }
    float d2 = -dot(n2, verts[0]);

    // Face 3: (verts[1], verts[2], verts[3])
    float3 n3 = cross(verts[2] - verts[1], verts[3] - verts[1]);
    if (dot(n3, verts[0] - verts[1]) > 0) {
        n3 = -n3;
    }
    float d3 = -dot(n3, verts[1]);

    // --------------------------------------------------------------------
    // 2. Initialize the intersection range [t_enter, t_exit]
    //    You might prefer (-∞, +∞), but this variant starts at [0, big].
    // --------------------------------------------------------------------
    float t_enter = 0.0f;
    float t_exit  = 1e30f;
    float eps     = 1e-10f;

    // --------------------------------------------------------------------
    // 3. Clip against Plane 0
    // --------------------------------------------------------------------
    {
        float dist0  = dot(n0, orig) + d0;
        float denom0 = dot(n0, dir);

        if (fabs(denom0) < eps) {
            // Ray is almost parallel to plane
            if (dist0 > 0.0f) {
                // Entire ray is outside
                return false;
            }
        }
        else {
            float tplane = -dist0 / denom0;
            if (denom0 < 0.0f) {
                // Ray entering at this t
                if (tplane > t_enter) {
                    t_enter = tplane;
                }
            }
            else {
                // Ray exiting
                if (tplane < t_exit) {
                    t_exit = tplane;
                }
            }
        }

        if (t_enter > t_exit) {
            return false;
        }
    }

    // --------------------------------------------------------------------
    // 4. Clip against Plane 1
    // --------------------------------------------------------------------
    {
        float dist1  = dot(n1, orig) + d1;
        float denom1 = dot(n1, dir);

        if (fabs(denom1) < eps) {
            if (dist1 > 0.0f) {
                return false;
            }
        }
        else {
            float tplane = -dist1 / denom1;
            if (denom1 < 0.0f) {
                if (tplane > t_enter) {
                    t_enter = tplane;
                }
            }
            else {
                if (tplane < t_exit) {
                    t_exit = tplane;
                }
            }
        }

        if (t_enter > t_exit) {
            return false;
        }
    }

    // --------------------------------------------------------------------
    // 5. Clip against Plane 2
    // --------------------------------------------------------------------
    {
        float dist2  = dot(n2, orig) + d2;
        float denom2 = dot(n2, dir);

        if (fabs(denom2) < eps) {
            if (dist2 > 0.0f) {
                return false;
            }
        }
        else {
            float tplane = -dist2 / denom2;
            if (denom2 < 0.0f) {
                if (tplane > t_enter) {
                    t_enter = tplane;
                }
            }
            else {
                if (tplane < t_exit) {
                    t_exit = tplane;
                }
            }
        }

        if (t_enter > t_exit) {
            return false;
        }
    }

    // --------------------------------------------------------------------
    // 6. Clip against Plane 3
    // --------------------------------------------------------------------
    {
        float dist3  = dot(n3, orig) + d3;
        float denom3 = dot(n3, dir);

        if (fabs(denom3) < eps) {
            if (dist3 > 0.0f) {
                return false;
            }
        }
        else {
            float tplane = -dist3 / denom3;
            if (denom3 < 0.0f) {
                if (tplane > t_enter) {
                    t_enter = tplane;
                }
            }
            else {
                if (tplane < t_exit) {
                    t_exit = tplane;
                }
            }
        }

        if (t_enter > t_exit) {
            return false;
        }
    }

    t = float2(t_enter, t_exit);
    return true;
}

}