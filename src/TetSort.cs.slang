import Rose.Core.MathUtils;
import Rose.Core.Quaternion;
import TetrahedronScene;

ParameterBlock<TetrahedronScene> scene;

RWStructuredBuffer<uint2>  sortPairs;
RWStructuredBuffer<float4> sortedColors;
RWStructuredBuffer<uint4>  sortedIndices;
RWStructuredBuffer<float4> sortedSpheres;

uniform float3 cameraPosition;

uint order_preserving_float_map(float value) {
    // For negative values, the mask becomes 0xffffffff.
    // For positive values, the mask becomes 0x80000000.
    uint uvalue = asuint(value);
    uint mask = -int(uvalue >> 31) | 0x80000000;
    uint bits = uvalue ^ mask;
    return bits;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void createPairs(uint3 threadId: SV_DispatchThreadID) {
    uint tetId = threadId.x;
    if (tetId >= scene.numTets)
        return;

    float4 sphere = scene.spheres[tetId];
    float3 toSphere = sphere.xyz - cameraPosition;
    float power = dot(toSphere, toSphere) - (sphere.w * sphere.w);

    uint key = order_preserving_float_map(power);

    if (sphere.w == 0 || power != power || isnan(power) || isinf(power))
        key = UINT32_MAX;

    sortPairs[tetId] = uint2(key, tetId);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void reorderTets(uint3 threadId: SV_DispatchThreadID) {
    uint tetId = threadId.x;
    if (tetId >= scene.numTets)
        return;

    uint srcTet = sortPairs[tetId].y;

    sortedColors[tetId]  = scene.colors[srcTet];
    sortedIndices[tetId] = scene.indices[srcTet];
    sortedSpheres[tetId] = scene.spheres[srcTet];
}
