import Rose.Core.MathUtils;
import Rose.Core.Quaternion;
import Scene.TetrahedronScene;
import SortUtils;
import Rose.Core.PackedTypes;

#include <Rose/Core/Bitfield.h>

using namespace vkDelTet;
using namespace RoseEngine;

ParameterBlock<TetrahedronScene> scene;
StructuredBuffer<uint> sortPayloads;
ByteAddressBuffer tetColors; // precomputed SH data
ByteAddressBuffer visibleTets;
ByteAddressBuffer blockSumAtomicCounter;

uniform float4x4 viewProjection;
uniform float4x4 invProjection;
uniform float3   rayOrigin;
uniform float    densityThreshold;
uniform float2   outputResolution;

struct v2f {
    float4 pos : SV_Position;
    
    // Intersection data, interpolated over triangles in a tet
    [[vk::location(0)]] float3 rayDir;
    [[vk::location(1)]] float4 planeNumerators;
    [[vk::location(2)]] float4 planeDenominators;
    
    // Tet data, constant for vertices in a tet
    [[vk::location(3)]] nointerpolation float  tetDensity;
    [[vk::location(4)]] nointerpolation float3 baseColor;
    [[vk::location(6)]] float dc_dt;
};

[shader("vertex")]
v2f vsmain(in uint vertexId: SV_VertexID) {
    // 12 vertices per tet
    const uint rawTetId = sortPayloads[vertexId / 12];
    const uint tetId = rawTetId;
    // const uint tetId = visibleTets.Load(rawTetIdr* sizeof(uint));
    
    v2f o = {};

    o.tetDensity = scene.load_tet_density(tetId);
    if (o.tetDensity > densityThreshold)
    {
        // o.tetVertices   = scene.load_tet_vertices(tetId);
        float4x3 verts = scene.load_tet_vertices(tetId);

        // 4 tris per tet
        const uint triId = (vertexId%12) / 3;
        const uint3 tri = TetrahedronScene::kTetTriangles[triId];
        const float3 vertex = verts[tri[vertexId%3]];

        // o.pos = mul(viewProjection, float4(vertex, 1));
        o.pos = mul(float4(vertex, 1), transpose(viewProjection));
        o.rayDir = vertex - rayOrigin;
        for (uint i = 0; i < 4; i++) {
            // outward facing normal
            const float3 n = cross(
                verts[TetrahedronScene::kTetTriangles[i][2]] - verts[TetrahedronScene::kTetTriangles[i][0]],
                verts[TetrahedronScene::kTetTriangles[i][1]] - verts[TetrahedronScene::kTetTriangles[i][0]]);
            o.planeNumerators[i] = dot(n, verts[TetrahedronScene::kTetTriangles[i][0]] - rayOrigin);
            o.planeDenominators[i] = dot(n, o.rayDir);
        }
        // o.v0 = verts[0];
        o.baseColor     = tetColors.Load<float3>(tetId * sizeof(float3));
        float3 colorGradient = scene.load_tet_gradient(tetId);
        o.dc_dt = dot(colorGradient, o.rayDir);

    }
    else
    {
        o.pos = 0;
    }

    return o;
}

[shader("fragment")]
float4 fsmain(v2f v, float4 fragPos : SV_Position) : SV_Target {
    // float2 ndc_xy = (fragPos.xy / outputResolution) * 2.0 - 1.0;
    // float4 ndc = float4(ndc_xy, fragPos.z, 1.0);
    // float4 world_pos_h = mul(invProjection, ndc);
    // float3 world_pos = world_pos_h.xyz / world_pos_h.w;
    // float3 rayDir = normalize(world_pos - rayOrigin);

    // if (v.tetDensity <= densityThreshold)
    //     return float4(0,0,0,1);

    {
        const float d = length(v.rayDir);
        v.rayDir /= d;
        v.planeDenominators /= d;
        v.dc_dt /= d;
    }

    float3 rayDir = v.rayDir;
    float opticalDepth = v.tetDensity;
    float4 all_t = v.planeNumerators / v.planeDenominators;

    float4 t_enter = (v.planeDenominators > 0.0f) ? all_t : float4(-FLT_MAX);
    float4 t_exit  = (v.planeDenominators < 0.0f) ? all_t : float4(FLT_MAX);

    // 3. Find the latest entry point and the earliest exit point using horizontal max/min.
    // This finds the final intersection interval [t.x, t.y].
    float2 t = float2(
        max(t_enter.x, max(t_enter.y, max(t_enter.z, t_enter.w))),
        min(t_exit.x, min(t_exit.y, min(t_exit.z, t_exit.w)))
    );
    
    // bool parallel_miss = any((abs(v.planeDenominators) < 1e-6f) & (v.planeNumerators > 0.0f));
    // if (parallel_miss) return float4(0,0,0,1);

    opticalDepth *= t.y - t.x;
    const float T = exp(-opticalDepth);
    float3 color = {
        integrate_channel(t.x, t.y, v.baseColor.r, v.dc_dt, v.tetDensity),
        integrate_channel(t.x, t.y, v.baseColor.g, v.dc_dt, v.tetDensity),
        integrate_channel(t.x, t.y, v.baseColor.b, v.dc_dt, v.tetDensity)
    };

    return float4(color, T);
}
