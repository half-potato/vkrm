import Rose.Core.MathUtils;
import Rose.Core.Quaternion;
import Scene.TetrahedronScene;
import SortUtils;

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
    #if TET_INTERSECTION
    [[vk::location(1)]] float3 scenePos : TEXCOORD0;
    [[vk::location(2)]] float3 v0;
    [[vk::location(3)]] float3 v1;
    [[vk::location(4)]] float3 v2;
    [[vk::location(5)]] float3 v3;
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

        #if TET_INTERSECTION

        o.scenePos = vertex;

        o.v0 = WaveReadLaneAt(vertex, (WaveGetLaneIndex()/4)*4 + 0);
        o.v1 = WaveReadLaneAt(vertex, (WaveGetLaneIndex()/4)*4 + 1);
        o.v2 = WaveReadLaneAt(vertex, (WaveGetLaneIndex()/4)*4 + 2);
        o.v3 = WaveReadLaneAt(vertex, (WaveGetLaneIndex()/4)*4 + 3);

        #else

        // constant color, scale density by tet size
        o.tetColor.w *= scene.spheres[tetId].w;

        #endif
    }

    vertices[meshVertexId] = o;
    triangles[meshVertexId] = (meshVertexId/4)*4 + TetrahedronScene::kTetTriangles[tetVertexId];
}

[shader("fragment")]
float4 fsmain(v2f i) : SV_Target {
    float dt = 0;

    #if TET_INTERSECTION

    float2 t;
    if (intersect_ray_tetrahedron(rayOrigin, normalize(i.scenePos - rayOrigin), i.v0, i.v1, i.v2, i.v3, t)) {
        if (t.y > 0) {
            dt = t.y - max(t.x, 0);
        }
    }

    #else

    dt = 1;

    #endif

    const float T = exp(-dt * i.tetColor.a);
    return float4((1 - T) * i.tetColor.rgb, T);
}
