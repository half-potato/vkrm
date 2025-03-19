#pragma once

#include <Rose/Core/Bitfield.h>
import Rose.Core.MathUtils;

namespace vkDelTet {

using namespace RoseEngine;

static const float kPlaneEpsilon = 1e-10f;

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

    struct TetrahedronLight {
        float3 color;
        float  roughness;
        float3 direction;

        float3 eval(const float3 viewDir) {
            const float f = max(0, dot(direction, viewDir));
            return color * pow(f, roughness);
        }
    };

    ByteAddressBuffer        vertices;
    ByteAddressBuffer        indices;
    
    Buffer<float4>           colors;
    Buffer<float>            densities;
    
    Buffer<float4>           lightColors;
    Buffer<float2>           lightDirections;

	StructuredBuffer<float4> spheres;

	float3 aabbMin;
	uint   numTets;
    float3 aabbMax;
    float  densityScale;

    uint numLights;

    uint load_index(uint tetId, uint tetVertexId) {
        const uint idx = 4 * tetId + tetVertexId;
        return indices.Load(idx * sizeof(uint));
    }

    uint4 load_tet_indices(uint tetId) {
        return indices.Load4(tetId * sizeof(uint4));
    }

    float3 load_vertex(uint vertexId) {
        return vertices.Load<float3>(vertexId * sizeof(float3));
    }

    float3 load_vertex(uint tetId, uint tetVertexId) {
        return load_vertex(load_index(tetId, tetVertexId));
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
        float4 c = colors.Load(tetId);
        if (c.w > 0) c.w = densities.Load(tetId) * densityScale;
        return c;
    }

    TetrahedronLight load_light(uint tetId, uint lightId) {
        const uint i = tetId * numLights + lightId;
        const float4 c = lightColors.Load(i);
        TetrahedronLight l;
        l.color     = c.rgb;
        l.roughness = c.a;

        const float2 sph = lightDirections.Load(i);
        const float sinTheta = sin(sph.y);
        l.direction = float3(
            cos(sph.x) * sinTheta,
            sin(sph.x) * sinTheta,
            cos(sph.y)
        );
        return l;
    }
};

float4 EvalTetColor(const float4 tetColor, const float dt = 1) {
    const float T = exp(-dt * tetColor.a);
    return float4((1 - T) * tetColor.rgb, T);
}

#if 0

bool intersect_ray_tetrahedron(float3 orig, float3 dir, float3 v0, float3 v1, float3 v2, float3 v3, out float2 t) {
    t = -1;

    // --------------------------------------------------------------------
    // 1. Compute plane equations for the 4 faces (outward normals)
    //    n[i].x, n[i].y, n[i].z, and d[i]
    // --------------------------------------------------------------------

    // Face 0: (v0, v1, v2)
    float3 n0 = cross(v1 - v0, v2 - v0);
    if (dot(n0, v3 - v0) > 0) {
        n0 = -n0;
    }
    float d0 = -dot(n0, v0);

    // Face 1: (v0, v1, v3)
    float3 n1 = cross(v1 - v0, v3 - v0);
    if (dot(n1, v2 - v0) > 0) {
        n1 = -n1;
    }
    float d1 = -dot(n1, v0);

    // Face 2: (v0, v2, v3)
    float3 n2 = cross(v2 - v0, v3 - v0);
    if (dot(n2, v1 - v0) > 0) {
        n2 = -n2;
    }
    float d2 = -dot(n2, v0);

    // Face 3: (v1, v2, v3)
    float3 n3 = cross(v2 - v1, v3 - v1);
    if (dot(n3, v0 - v1) > 0) {
        n3 = -n3;
    }
    float d3 = -dot(n3, v1);

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

#else

bool intersect_plane_convex_hull(const float dist, const float denom, inout float2 t) {
    if (abs(denom) < kPlaneEpsilon) {
        // Ray is almost parallel to plane
        if (dist > 0) {
            // Entire ray is outside
            return false;
        }
    } else {
        const float t_i = dist / denom;
        if (denom > 0) {
            // Ray entering at this t
            if (t_i > t.x) {
                t.x = t_i;
            }
        } else {
            // Ray exiting
            if (t_i < t.y) {
                t.y = t_i;
            }
        }
    }

    if (t.x > t.y) {
        return false;
    }

    return true;
}

bool intersect_ray_tetrahedron(const float3 orig, const float3 dir, const float4x3 verts, out float2 t) {
    t = float2(0, FLT_MAX);
    [ForceUnroll]
    for (uint i = 0; i < 4; i++) {
        // outward facing normal
        const float3 n = cross(
            verts[TetrahedronScene::kTetTriangles[i][2]] - verts[TetrahedronScene::kTetTriangles[i][0]],
            verts[TetrahedronScene::kTetTriangles[i][1]] - verts[TetrahedronScene::kTetTriangles[i][0]]);

        const float dist  = dot(n, verts[TetrahedronScene::kTetTriangles[i][0]] - orig);
        const float denom = dot(n, dir);

        if (!intersect_plane_convex_hull(dist, denom, t))
            return false;
    }
    return true;
}

bool intersect_ray_tetrahedron(const float3 orig, const float3 dir, const float3 v0, const float3 v1, const float3 v2, const float3 v3, out float2 t) {
    return intersect_ray_tetrahedron(orig, dir, float4x3(v0, v1, v2, v3), t);
}

#endif

}