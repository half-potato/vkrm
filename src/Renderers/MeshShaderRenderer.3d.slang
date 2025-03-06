import Rose.Core.MathUtils;
import Rose.Core.Quaternion;
import Scene.TetrahedronScene;
import SortUtils;


#define INTERSECTION_MODE_PER_PIXEL 0
#define INTERSECTION_MODE_PER_TET   1
#define INTERSECTION_MODE_COMBINED  2
#define INTERSECTION_MODE_NONE      3

#ifndef INTERSECTION_MODE
#define INTERSECTION_MODE INTERSECTION_MODE_COMBINED
#endif

#include <Rose/Core/Bitfield.h>

using namespace vkDelTet;

ParameterBlock<TetrahedronScene> scene;
StructuredBuffer<uint2> sortBuffer;
ByteAddressBuffer       transformedVertices;

uniform float4x4 viewProjection;
uniform float3   rayOrigin;
uniform float    densityThreshold;

struct v2f {
    float4 pos : SV_Position;

    [[vk::location(0)]] float4 tetColor : COLOR;

    #if INTERSECTION_MODE == INTERSECTION_MODE_PER_PIXEL
    
    [[vk::location(1)]] float3 scenePos;
    [[vk::location(2)]] float3 v0;
    [[vk::location(3)]] float3 v1;
    [[vk::location(4)]] float3 v2;
    [[vk::location(5)]] float3 v3;
    
    #elif INTERSECTION_MODE == INTERSECTION_MODE_COMBINED
    
    [[vk::location(1)]] float3 rayDir;
    [[vk::location(2)]] float4 planeNumerators;
    [[vk::location(3)]] float4 planeDenominators;
    
    #endif
};

#define GROUP_SIZE 32
#define TETS_PER_GROUP (GROUP_SIZE / 4)

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

    const uint tetId = sortBuffer[threadId.x/4].y;

    v2f o = {};
    o.pos = 0;
    o.tetColor = scene.load_rgb_density(tetId);
    if (o.tetColor.a > densityThreshold) {
        const float3 vertex = scene.load_vertex(scene.load_index(tetId, tetVertexId));
        o.pos = mul(viewProjection, float4(vertex, 1));

        const uint baseLaneId = (WaveGetLaneIndex() / 4) * 4;

        #define WaveReadTetVar(x, id)        WaveReadLaneAt(x, baseLaneId + id)
        #define WaveReadTetVarRel(x, offset) WaveReadTetVar(x, (WaveGetLaneIndex() + offset) % 4)

        #if INTERSECTION_MODE == INTERSECTION_MODE_PER_PIXEL
        {
            o.scenePos = vertex;

            o.v0 = WaveReadTetVar(vertex, 0);
            o.v1 = WaveReadTetVar(vertex, 1);
            o.v2 = WaveReadTetVar(vertex, 2);
            o.v3 = WaveReadTetVar(vertex, 3);
        }
        #elif INTERSECTION_MODE == INTERSECTION_MODE_PER_TET
        {
            // constant color, scale density by tet thickness

            const float3 centroid = (WaveReadTetVarRel(vertex, 0) + WaveReadTetVarRel(vertex, 1) + WaveReadTetVarRel(vertex, 2)) / 3.0;
            
            float2 t;
            if (!intersect_ray_tetrahedron(
                rayOrigin,
                normalize(centroid  - rayOrigin),
                WaveReadTetVar(vertex, 0),
                WaveReadTetVar(vertex, 1),
                WaveReadTetVar(vertex, 2),
                WaveReadTetVar(vertex, 3),
                t))
                t = 0;

            // thickness = distance between vertex and plane opposite from vertex
            const float thickness = t.y - max(0, t.x);

            const float d = (WaveReadTetVar(thickness, 0) +
                            WaveReadTetVar(thickness, 1) +
                            WaveReadTetVar(thickness, 2) +
                            WaveReadTetVar(thickness, 3)) / 4.0;

            o.tetColor.w *= d;
        }
        #elif INTERSECTION_MODE == INTERSECTION_MODE_COMBINED
        {
            o.rayDir = vertex - rayOrigin;

            #if 0

            [ForceUnroll]
            for (uint i = 0; i < 4; i++)
            {
                const uint3 tri = TetrahedronScene::kTetTriangles[i];

                // outward facing normal
                const float3 n = cross(
                    WaveReadTetVar(vertex, tri[2]) - WaveReadTetVar(vertex, tri[0]),
                    WaveReadTetVar(vertex, tri[1]) - WaveReadTetVar(vertex, tri[0]));

                o.planeNumerators[i]   = dot(n, WaveReadTetVar(vertex, tri[0]) - rayOrigin);
                o.planeDenominators[i] = dot(n, o.rayDir);
            }

            #else

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
            
            #endif
        }
        #else // if INTERSECTION_MODE == INTERSECTION_MODE_NONE

        // constant color, scale density by circumcenter diameter
        o.tetColor.w *= 2*scene.spheres[tetId].w;

        #endif
    }

    vertices[meshVertexId] = o;
    triangles[meshVertexId] = (meshVertexId/4)*4 + TetrahedronScene::kTetTriangles[tetVertexId];
}

[shader("fragment")]
float4 fsmain(v2f v) : SV_Target {
    float4 c = v.tetColor;

    #if INTERSECTION_MODE == INTERSECTION_MODE_PER_PIXEL
    {
        float2 t;
        if (intersect_ray_tetrahedron(rayOrigin, normalize(v.scenePos - rayOrigin), v.v0, v.v1, v.v2, v.v3, t)) {
            if (t.y > 0) {
                c.a *= t.y - max(t.x, 0);
            } else
                c.a = 0;
        } else
            c.a = 0;
    }
    #elif INTERSECTION_MODE == INTERSECTION_MODE_COMBINED
    
    // normalize ray direction
    {
        const float d = length(v.rayDir);
        v.rayDir /= d;
        v.planeDenominators /= d;
    }

    // intersect every plane

    float2 t = float2(0, FLT_MAX);
    [ForceUnroll]
    for (uint i = 0; i < 4; i++) {
        if (!intersect_plane_convex_hull(v.planeNumerators[i], v.planeDenominators[i], t)) {
            c.a = 0;
            t = -1;
            break;
        }
    }
    if (t.y > 0) c.a *= t.y - max(t.x, 0);

    #endif

    const float T = exp(-c.a);
    return float4((1 - T) * c.rgb, T);
}
