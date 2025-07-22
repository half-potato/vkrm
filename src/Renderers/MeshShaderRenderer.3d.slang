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
ByteAddressBuffer tetColors; // precomputed SH data

uniform float4x4 viewProjection;
uniform float3   rayOrigin;
uniform float    densityThreshold;

struct v2f {
    float4 pos : SV_Position;

    float dc_dt;
    float3 rayDir;
    float4 planeNumerators;
    float4 planeDenominators;

    nointerpolation float tetDensity;
    nointerpolation float3 baseColor;

    // [[vk::location(4)]]  nointerpolation float3 c0;
    // [[vk::location(5)]]  nointerpolation float3 c1;
    // [[vk::location(6)]]  nointerpolation float3 c2;
    // [[vk::location(7)]]  nointerpolation float3 c3;
    // [[vk::location(8)]]  nointerpolation float3 v0;
    // [[vk::location(9)]]  nointerpolation float3 v1;
    // [[vk::location(10)]] nointerpolation float3 v2;
    // [[vk::location(11)]] nointerpolation float3 v3;
};

#define GROUP_SIZE 32
#define TETS_PER_GROUP (GROUP_SIZE / 4)

#define softplus(x, beta) ( (1.0 / beta) * log(1.0 + exp(beta*x)) )

[shader("mesh")]
[outputtopology("triangle")]
[numthreads(GROUP_SIZE, 1, 1)]
void meshmain(
    in uint3 threadId: SV_DispatchThreadID,
    OutputVertices<v2f, GROUP_SIZE>  vertices,
    OutputIndices<uint3, GROUP_SIZE> triangles) {
        
    SetMeshOutputCounts(GROUP_SIZE, GROUP_SIZE);

    const uint meshVertexId = threadId.x % GROUP_SIZE;
    const uint tetVertexId  = threadId.x % 4;

    v2f o = {};
    o.pos = 0;

    if (threadId.x/4 < scene.numTets)
    {
        const uint tetId = sortBuffer[threadId.x/4].y;
        o.baseColor     = tetColors.Load<float3>(tetId * sizeof(float3));

        o.tetDensity = scene.load_tet_density(tetId);
        if (o.tetDensity > densityThreshold)
        {
            const uint vertexId = scene.load_index(tetId, tetVertexId);
            const float3 vertex = scene.load_vertex(vertexId);
            o.pos = mul(viewProjection, float4(vertex, 1));
            o.rayDir = vertex - rayOrigin;

            const uint baseLaneId = (WaveGetLaneIndex() / 4) * 4;

            #define WaveReadTetVar(x, id)        WaveReadLaneAt(x, baseLaneId + id)
            #define WaveReadTetVarRel(x, offset) WaveReadTetVar(x, (WaveGetLaneIndex() + offset) % 4)

            const uint3 tri = TetrahedronScene::kTetTriangles[tetVertexId];

            // outward facing normal
            const float3 n = cross(
                WaveReadTetVar(vertex, tri[2]) - WaveReadTetVar(vertex, tri[0]),
                WaveReadTetVar(vertex, tri[1]) - WaveReadTetVar(vertex, tri[0]));

            const float num = dot(n, WaveReadTetVar(vertex, tri[0]) - rayOrigin);

            o.planeNumerators = float4(
                WaveReadTetVar(num, 0),
                WaveReadTetVar(num, 1),
                WaveReadTetVar(num, 2),
                WaveReadTetVar(num, 3) );
            o.planeDenominators = float4(
                dot(WaveReadTetVar(n, 0), o.rayDir),
                dot(WaveReadTetVar(n, 1), o.rayDir),
                dot(WaveReadTetVar(n, 2), o.rayDir),
                dot(WaveReadTetVar(n, 3), o.rayDir) );

            float3 colorGradient = scene.load_tet_gradient(tetId);
            o.dc_dt = dot(colorGradient, o.rayDir);
            // o.v0 = WaveReadTetVar(vertex, 0);

            // float3 color = 0;
            // color += scene.load_tet_vertex_color(tetId, tetVertexId).rgb;
            // color += vertexColors.Load(vertexId).rgb;
            // color = softplus(color, 10);

            // o.c0 = WaveReadTetVar(color, 0);
            // o.c1 = WaveReadTetVar(color, 1);
            // o.c2 = WaveReadTetVar(color, 2);
            // o.c3 = WaveReadTetVar(color, 3);
            //
            // o.v0 = WaveReadTetVar(vertex, 0);
            // o.v1 = WaveReadTetVar(vertex, 1);
            // o.v2 = WaveReadTetVar(vertex, 2);
            // o.v3 = WaveReadTetVar(vertex, 3);
        }
    }

    vertices[meshVertexId] = o;
    triangles[meshVertexId] = (meshVertexId/4)*4 + TetrahedronScene::kTetTriangles[tetVertexId];
}

[shader("fragment")]
float4 fsmain(v2f v) : SV_Target {
    if (v.tetDensity <= densityThreshold)
        return float4(0,0,0,1);
    // normalize ray direction
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
