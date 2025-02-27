import Rose.Core.MathUtils;
using namespace RoseEngine;

ByteAddressBuffer   vertices;
RWByteAddressBuffer transformedVertices;

[[vk::push_constant]]
cbuffer pc {
    float4x4 transform;
    uint     vertexCount;
};

[shader("compute")]
[numthreads(32,1,1)]
void main(uint3 id: SV_DispatchThreadID) {
    if (id.x >= vertexCount)
        return;
    const uint addr = id.x * sizeof(float3);
    const float3 vertex = vertices.Load<float3>(addr);
    const float4 vh = mul(transform, float4(vertex, 1));
    transformedVertices.Store<float3>(addr, vh.xyz / abs(vh.w));
}