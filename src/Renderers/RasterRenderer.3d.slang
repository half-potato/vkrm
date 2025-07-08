import Rose.Core.MathUtils;
import Rose.Core.Quaternion;
import Scene.TetrahedronScene;
import SortUtils;
import Rose.Core.PackedTypes;
import safe_math;

#include <Rose/Core/Bitfield.h>

using namespace vkDelTet;
using namespace RoseEngine;

ParameterBlock<TetrahedronScene> scene;
StructuredBuffer<uint2> sortBuffer;
ByteAddressBuffer tetColors; // precomputed SH data

uniform float4x4 viewProjection;
uniform float3   rayOrigin;
uniform float    densityThreshold;

struct v2f {
    float4 pos : SV_Position;
    
    // Intersection data, interpolated over triangles in a tet
    [[vk::location(0)]] float3 rayDir;
    
    // Tet data, constant for vertices in a tet
    [[vk::location(1)]] nointerpolation float  tetDensity;
    [[vk::location(2)]] nointerpolation float3 baseColor;
    [[vk::location(3)]] nointerpolation float3 colorGradient;
    [[vk::location(4)]] nointerpolation float4x3 tetVertices;
};

[shader("vertex")]
v2f vsmain(in uint vertexId: SV_VertexID) {
    // 12 vertices per tet
    const uint tetId = sortBuffer[vertexId / 12].y;
    
    v2f o = {};

    o.tetDensity = scene.load_tet_density(tetId);
    if (o.tetDensity > densityThreshold)
    {
        o.tetVertices   = scene.load_tet_vertices(tetId);
        o.baseColor     = tetColors.Load<float3>(tetId * sizeof(float3));
        o.colorGradient = scene.load_tet_gradient(tetId);

        // 4 tris per tet
        const uint triId = (vertexId%12) / 3;
        const uint3 tri = TetrahedronScene::kTetTriangles[triId];
        const float3 vertex = o.tetVertices[tri[vertexId%3]];

        o.pos = mul(viewProjection, float4(vertex, 1));
        o.rayDir = vertex - rayOrigin;
    }
    else
    {
        o.pos = 0;
    }

    return o;
}

[Differentiable]
float compute_integral_1D(float c0, float c1, float dt, float d) {
    float alpha = safe_exp(-d*dt);
    float X = (-d*dt*alpha + 1 - alpha);
    float Y = (d*dt-1) + alpha;
    return safe_div(X*c0+Y*c1, d*dt);
}

 

[Differentiable]
inline float integrate_channel(
    float t_n, float t_f,
    float c_at_t0, float dc_dt, float density)
{
    // Find zero-crossing and adjust integration bounds to the positive segment
    float t_zero = - safe_div(c_at_t0, dc_dt);
    bool change_within = (t_n < t_zero) && (t_zero < t_f);
    float c0 = max(c_at_t0 + dc_dt * t_n, 0.f);
    float c1 = max(c_at_t0 + dc_dt * t_f, 0.f);
    if (change_within) {
        float cm = max(c_at_t0 + dc_dt * t_zero, 0.f);
        float dt1 = max(t_zero - t_n, 0);
        float dt2 = max(t_f - t_zero, 0);
        float onemalpha = min(max(safe_exp(-density * dt1), 0.f), 1.f);
        if (dc_dt < 0.f) {
            return compute_integral_1D(cm, c0, dt1, density);
        }
        if (dc_dt > 0.f) {
            //return compute_integral_1D(0, 0, dt1, density) + onemalpha * compute_integral_1D(c1, cm, dt2, density);
            return onemalpha * compute_integral_1D(c1, cm, dt2, density);
        }
    }
    float dt = max(t_f - t_n, 0.f);
    return compute_integral_1D(c1, c0, dt, density);
}

[shader("fragment")]
float4 fsmain(v2f v) : SV_Target {
    if (v.tetDensity <= densityThreshold)
        return float4(0,0,0,1);

    v.rayDir = normalize(v.rayDir);

    float2 t;
    if (!intersect_ray_tetrahedron(rayOrigin, v.rayDir, v.tetVertices, t))
        return float4(0, 0, 0, 1); // early exit in case HW raster disagrees with ray-tet intersector

    const float opticalDepth = v.tetDensity * (t.y - max(t.x, 0));
    const float T = exp(-opticalDepth);

    // interpolated color
    float3 v_enter = t.x * v.rayDir + (rayOrigin - v.tetVertices[0]);
    float3 v_exit  = t.y * v.rayDir + (rayOrigin - v.tetVertices[0]);


    float dc_dt = dot(v.colorGradient, v.rayDir);
    float offset = dot(rayOrigin - v.tetVertices[0], v.colorGradient);
    float3 color = {
        integrate_channel(t.x, t.y, v.baseColor.r + offset, dc_dt, v.tetDensity),
        integrate_channel(t.x, t.y, v.baseColor.g + offset, dc_dt, v.tetDensity),
        integrate_channel(t.x, t.y, v.baseColor.b + offset, dc_dt, v.tetDensity)
    };
    //color = (1-T) * v.baseColor;
    color = opticalDepth > 0 ? color : 0;
    /*

    float3 c_enter = max(linear_color(v.colorGradient, v_enter) + v.baseColor, 0.f);
    float3 c_exit  = max(linear_color(v.colorGradient, v_exit)  + v.baseColor, 0.f);

    // integrate color through the tet
    float X = (-opticalDepth * T + 1 - T);
    float Y = (opticalDepth - 1) + T;
    float3 color = opticalDepth > 0 ? (X * c_exit + Y * c_enter) / opticalDepth : 0;
    */

    return float4(color, T);
}
