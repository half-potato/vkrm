import Rose.Core.MathUtils;
import Rose.Core.Quaternion;
import Scene.TetrahedronScene;
import SortUtils;
import Rose.Core.PackedTypes;

#include <Rose/Core/Bitfield.h>

using namespace vkDelTet;
using namespace RoseEngine;

ParameterBlock<TetrahedronScene> scene;
StructuredBuffer<uint2> sortBuffer;
Buffer<float4> vertexColors; // precomputed vertex SH

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
void vsmain(in uint vertexId: SV_VertexID) {
    const uint tetId = sortBuffer[vertexId / 12].y;
    
    v2f o = {};
    o.pos = 0;

    o.tetDensity = scene.load_tet_density(tetId);
    if (o.tetDensity > densityThreshold)
    {
        const uint3 tri = TetrahedronScene::kTetTriangles[vertexId / 4];
        const float4x3 tetVertices = scene.load_tet_vertices(tetId);
        const float3 vertex = tetVertices[tri[vertexId % 3]];

        o.pos = mul(viewProjection, float4(vertex, 1));
        o.rayDir = vertex - rayOrigin;

        float3 color = vertexColors.Load(vertexId).rgb;

        o.tetVertices = tetVertices;
    }
}

[shader("fragment")]
float4 fsmain(v2f v) : SV_Target {
    if (v.tetDensity <= densityThreshold)
        return float4(0,0,0,1);

    v.rayDir = normalize(v.rayDir);

    // intersect every plane

    float opticalDepth = v.tetDensity;

    float2 t;
    if (!intersect_ray_tetrahedron(rayOrigin, v.rayDir, v.tetVertices, t))
        return float4(0, 0, 0, 1); // this shouldnt be possible

    opticalDepth *= t.y - max(t.x, 0);

    const float T = exp(-opticalDepth);

    // interpolated color
    float3 v_enter = t.x * v.rayDir + (rayOrigin - v.tetVertices[0]);
    float3 v_exit  = t.y * v.rayDir + (rayOrigin - v.tetVertices[0]);
    // float3 c_enter = max(linear_color(g.vertex_color, v_enter), 0.f);
    // float3 c_exit = max(linear_color(g.vertex_color, v_exit), 0.f);
    float3 c_enter = max(linear_color(v.colorGradient, v_enter) + v.baseColor, 0.f);
    float3 c_exit  = max(linear_color(v.colorGradient, v_exit)  + v.baseColor, 0.f);
    //*/

    // integrate color through the tet
    float X = (-opticalDepth * T + 1 - T);
    float Y = (opticalDepth - 1) + T;
    float3 color = opticalDepth > 0 ? (X * c_exit + Y * c_enter) / opticalDepth : 0;

    return float4(color, T);
}
