import Scene.TetrahedronScene;

using namespace vkDelTet;

// Define the parameters we'll receive from the C++ side
ParameterBlock<TetrahedronScene> scene;
uniform float4x4 viewProjection;
StructuredBuffer<uint> selection;
uniform float2 outputResolution;
uniform float2 mousePos;
uniform float selectionRadius;

RWByteAddressBuffer candidateCount;
RWStructuredBuffer<uint> sortKeys;
RWStructuredBuffer<float> sortPayloads;

struct v2f {
    float4 pos : SV_Position;
};

[shader("vertex")]
v2f vsmain(out float psize : SV_PointSize, uint instanceId : SV_InstanceID) {
    v2f o = {};

    // For testing, you're loading a fixed vertex. Remember to switch this back.
    uint vertexToDrawId = selection[instanceId];
    float3 vertexPos = scene.load_vertex(vertexToDrawId);

    // Transform the vertex position into clip space
    o.pos = mul(viewProjection, float4(vertexPos, 1.0));

    // --- 3D Sizing Logic ---

    // 1. Calculate a base size that shrinks with distance.
    //    o.pos.w is the distance from the camera's near plane.
    //    Dividing by it makes the size perspective-correct.
    //    We scale by the viewport height and a world-size factor.
    float perspectiveSize = (1.1) / o.pos.w;
    psize = max(5.0, perspectiveSize);

    return o;
}

[shader("fragment")]
float4 fsmain(v2f i) : SV_Target {
    // Output a bright, constant color for the highlight. Yellow is a good choice.
    return float4(1.0, 1.0, 0.0, 1.0); // Bright Yellow
}

[shader("compute")]
[numthreads(64, 1, 1)]
void select(uint3 threadId: SV_DispatchThreadID) {
    uint vertId = threadId.x;
    float3 vert = scene.load_vertex(vertId);
    float4 clipPos = mul(viewProjection, float4(vert, 1.0));
    float3 ndcPos = float3(clipPos.x, clipPos.y, clipPos.z) / clipPos.w;
    float2 screenPos = {
        (ndcPos.x + 1.0f) * 0.5f * outputResolution.x,
        (ndcPos.y + 1.0f) * 0.5f * outputResolution.y
    };

    float dx = screenPos.x - mousePos.x;
    float dy = screenPos.y - mousePos.y;
    if ((dx * dx + dy * dy) < (selectionRadius * selectionRadius)) {
        uint listIndex;
        candidateCount.InterlockedAdd(0, 1, listIndex);
        sortKeys[vertId] = vertId;
        sortPayloads[vertId] = clipPos.z;
    }
}
