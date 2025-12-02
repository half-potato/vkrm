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

uniform float4x4 viewProjection;
uniform float4x4 invProjection;
uniform float3   rayOrigin;
uniform float    densityThreshold;
uniform float2   outputResolution;

struct v2f {
    float4 pos : SV_Position;

    float dc_dt;
    float3 rayDir;
    float4 planeNumerators;
    float4 planeDenominators;

    nointerpolation float tetDensity;
    nointerpolation float3 baseColor;
};


[shader("vertex")]
v2f vsmain(in uint vertexId : SV_VertexID, in uint instanceId : SV_InstanceID) {
    const uint tetId = sortPayloads[instanceId];
    v2f o = {};

    o.tetDensity = scene.load_tet_density(tetId);
    if (o.tetDensity > densityThreshold)
    {
        // o.tetVertices   = scene.load_tet_vertices(tetId);
        float4x3 verts = scene.load_tet_vertices(tetId);

        const float3 vertex = verts[vertexId];

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

    opticalDepth *= t.y - t.x;

    float3 c_start = max(v.baseColor + v.dc_dt * t.x, 0.f);
    float3 c_end = max(v.baseColor + v.dc_dt * t.y, 0.f);
    float4 color = compute_integral(c_end, c_start, opticalDepth);

    return color;
}
