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

    bits = ((bits << 24) & 0xFF000000) |
           ((bits << 8) & 0x00FF0000) |
           ((bits >> 8) & 0x0000FF00) |
           ((bits >> 24) & 0x000000FF);
    //bits = ((bits & 0x000000FF) << 24) |
    //       ((bits & 0x0000FF00) <<  8) |
    //       ((bits & 0x00FF0000) >>  8) |
    //       ((bits & 0xFF000000) >> 24);
    return bits;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void createPairs(uint3 threadId: SV_DispatchThreadID) {
    uint tetId = threadId.x;
    if (tetId >= scene.numTets)
        return;

    double4 sphere = scene.spheres[tetId];
    double3 toSphere = sphere.xyz - (double3)cameraPosition;
    float power = (float)(dot(toSphere, toSphere) - (sphere.w * sphere.w));

    if (power != power || isnan(power) || isinf(power))
        power = FLT_MAX;

    power = sqrt(abs(power)) * sign(power);

   	uint key = (uint)clamp(100*power, 0, UINT32_MAX);
    //uint key = order_preserving_float_map(power);

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
