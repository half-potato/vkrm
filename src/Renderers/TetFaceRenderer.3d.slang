import Rose.Core.MathUtils;
import Rose.Core.Quaternion;
import Scene.TetrahedronScene;
import SortUtils;

using namespace vkDelTet;

ParameterBlock<TetrahedronScene> scene;
StructuredBuffer<uint2> sortBuffer;
ByteAddressBuffer       transformedVertices;

uniform float4x4 viewProjection;
uniform float4x4 invProjection;
uniform quat     cameraRotation;
uniform float3   rayOrigin;
uniform float    densityThreshold;
uniform float2   outputResolution;

struct v2f {
    float4 pos : SV_Position;

    nointerpolation float4 color : COLOR0;

    #if TET_INTERSECTION
    nointerpolation float4x3 tetVerts : TEXCOORD0;
    #endif
};

[shader("vertex")]
v2f vsmain(uint tetId: SV_InstanceID, uint vertexId: SV_VertexID) {
    tetId = sortBuffer[tetId].y;

    v2f o = {};
    o.color = scene.load_tet_density(tetId);

    if (o.color.a > densityThreshold) {
        const uint tetVertexId = TetrahedronScene::kTetTriangles[(vertexId / 3) % 4][vertexId % 3];

        #if PRE_TRANSFORM && TET_INTERSECTION

        const uint4 tetInds = sizeof(float3) * scene.load_tet_indices(tetId);
        o.tetVerts[0] = transformedVertices.Load<float3>(tetInds[0]);
        o.tetVerts[1] = transformedVertices.Load<float3>(tetInds[1]);
        o.tetVerts[2] = transformedVertices.Load<float3>(tetInds[2]);
        o.tetVerts[3] = transformedVertices.Load<float3>(tetInds[3]);
        o.pos = float4(o.tetVerts[tetVertexId], 1);

        #elif !PRE_TRANSFORM && TET_INTERSECTION

        o.tetVerts = scene.load_tet_vertices(tetId);
        o.pos = mul(viewProjection, float4(o.tetVerts[tetVertexId], 1));

        #elif PRE_TRANSFORM && !TET_INTERSECTION

        uint sceneVertexId = scene.load_index(tetId, tetVertexId);
        o.pos = float4(transformedVertices.Load<float3>(sizeof(float3) * sceneVertexId), 1);

        #else

        const float3 vertex = scene.load_vertex(tetId, tetVertexId);
        o.pos = mul(viewProjection, float4(vertex, 1));

        #endif

        #if TET_INTERSECTION == 0
        // constant color, scale by tet size
        o.color.w *= scene.spheres[tetId].w;
        #endif

    } else {
        o.pos = 0;
        o.color = 0;
    }
    return o;
}

[shader("fragment")]
float4 fsmain(v2f i) : SV_Target {
    float4 fragColor = float4(0, 0, 0, 1);

    float dt = 0;

    #if TET_INTERSECTION
    const float4 ndc = float4(((floor(i.pos.xy) + 0.5) / outputResolution) * 2 - 1, i.pos.z, 1);
    
    #if PRE_TRANSFORM

    float zFar = 1000;
    float zNear = 0.01;

    float4 z = float4(i.tetVerts[0].z, i.tetVerts[1].z, i.tetVerts[2].z, i.tetVerts[3].z);
    z *= (zFar - zNear) / (2 * zFar * zNear);

    float t1 = max(max(z[0], z[1]), max(z[2], z[3]));
    float t0 = min(min(z[0], z[1]), min(z[2], z[3]));

    dt = max(0, t1 - max(t0, 0));

    #else // #elif PRE_TRANSFORM && !TET_INTERSECTION

    const float3 rayDirection = normalize(q_rotate_vector(cameraRotation, mul(invProjection, ndc).xyz));

    float2 t;
    if (intersect_ray_tetrahedron(rayOrigin, rayDirection, i.tetVerts, t)) {
        if (t.y > 0) {
            dt = t.y - max(t.x, 0);
        }
    }

    #endif
    #else // TET_INTERSECTION

    dt = 1;

    #endif

    if (dt > 0)
        fragColor = EvalTetColor(i.color, dt);

    return fragColor;
}
