#pragma once

#include <Rose/Core/Bitfield.h>
import Rose.Core.MathUtils;
import safe_math;

namespace vkDelTet {

using namespace RoseEngine;

static const float kPlaneEpsilon = 1e-10f;

struct TetrahedronScene {
    // 4 triangles per tet
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

    ByteAddressBuffer vertices;
    ByteAddressBuffer tetIndices;
    Buffer<float>     tetDensities;
    ByteAddressBuffer tetGradients;

	float3 aabbMin;
	uint   numTets;
    float3 aabbMax;
    float  densityScale;

    uint numVertices;

    uint load_index(uint tetId, uint tetVertexId) {
        const uint idx = 4 * tetId + tetVertexId;
        return tetIndices.Load(idx * sizeof(uint));
    }

    uint4 load_tet_indices(uint tetId) {
        return tetIndices.Load4(tetId * sizeof(uint4));
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

    float load_tet_density(uint tetId) {
        return tetDensities.Load(tetId) * densityScale;
    }

    float3 load_tet_gradient(uint tetId) {
        return tetGradients.Load<float3>(tetId * sizeof(float3));
    }
};

float3 linear_color(float3 grad, float3 v) {
    return max(dot(grad, v), 0.f);
}

// bool intersect_plane_convex_hull(const float dist, const float denom, inout float2 t) {
//     if (abs(denom) < kPlaneEpsilon) {
//         // Ray is almost parallel to plane
//         if (dist > 0) {
//             // Entire ray is outside
//             return false;
//         }
//     } else {
//         const float t_i = dist / denom;
//         if (denom > 0) {
//             // Ray entering at this t
//             if (t_i > t.x) {
//                 t.x = t_i;
//             }
//         } else {
//             // Ray exiting
//             if (t_i < t.y) {
//                 t.y = t_i;
//             }
//         }
//     }
//
//     if (t.x > t.y) {
//         return false;
//     }
//
//     return true;
// }

bool intersect_plane_convex_hull(const float dist, const float denom, inout float2 t) {
    // A specific epsilon is better than a variable.
    // This rare case is okay as a branch, but the logic is simpler.
    if (abs(denom) < 1e-6f) {
        return dist <= 0.0f;
    }

    // This is the main optimization.
    // It replaces the nested if/else structure with clear max/min operations.
    // This pattern is easily converted to fast, non-branching
    // "conditional move" instructions by the compiler.
    const float t_i = dist / denom;
    if (denom > 0.0f) {
        t.x = max(t.x, t_i);
    } else {
        t.y = min(t.y, t_i);
    }

    // The final validity check remains the same.
    return t.x <= t.y;
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

[Differentiable]
float compute_integral_1D(float c0, float c1, float d_dt) {

    float alpha = exp(-d_dt);
    float X = (-d_dt*alpha + 1 - alpha);
    float Y = (d_dt-1) + alpha;
    return (X*c0+Y*c1) / d_dt;
}

[Differentiable]
inline float integrate_channel(
    float t_n, float t_f,
    float c_at_t0, float dc_dt, float density)
{
    // Find where the linear color function C(t) would cross zero.
    float t_zero = clamp(-(c_at_t0 / dc_dt), t_n, t_f);

    // Determine the start and end of the segment where C(t) > 0.
    // This replaces the main "if (change_within)" and nested branches.
    float t_start = (dc_dt > 0.0f) ? t_zero : t_n;
    float t_end   = (dc_dt < 0.0f) ? t_zero : t_f;

    // Clamp this "positive" segment to the actual integration bounds [t_n, t_f].
    float dt_pos_segment = t_end - t_start;
    float d_dt = dt_pos_segment * density;
    if (d_dt < 1e-3) {
        return 0.f;
    }

    // Calculate transmittance through the initial "zero-color" segment [t_n, t_start].
    float dt_zero_segment = t_start - t_n;
    float T_zero_segment = exp(-density * dt_zero_segment);

    // Calculate the integral over the clamped positive segment [t_start, t_end].
    float c_start = c_at_t0 + dc_dt * t_start;
    float c_end   = c_at_t0 + dc_dt * t_end;

    // The final result is the attenuated integral over the positive part.
    return T_zero_segment * compute_integral_1D(c_end, c_start, d_dt);
}



}
