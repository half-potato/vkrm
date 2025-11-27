import Rose.Core.MathUtils;
import Rose.Core.Quaternion;
import Scene.TetrahedronScene;
import SortUtils;

using namespace vkDelTet;

StructuredBuffer<float4>  spheres;
RWStructuredBuffer<uint> sortKeys;
RWStructuredBuffer<uint> sortPayloads;

RWBuffer<float4>           sortedColors;
RWStructuredBuffer<uint4>  sortedIndices;
RWStructuredBuffer<float4> sortedSpheres;
RWStructuredBuffer<uint> markedTets;        // The final, compact list of tet IDs

uniform float3 rayOrigin;
uniform uint numSpheres;

[shader("compute")]
[numthreads(64, 1, 1)]
void createPairs(uint3 threadId: SV_DispatchThreadID) {
    if (threadId.x >= numSpheres)
        return;
    sortKeys[threadId.x] = UINT32_MAX;
    sortPayloads[threadId.x] = threadId.x;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void updatePairs(uint3 threadId: SV_DispatchThreadID) {
    if (threadId.x >= numSpheres)
        return;

    const uint tetId = sortPayloads[threadId.x];
    if (markedTets.Load(tetId) == 0) {
        sortKeys[threadId.x] = UINT32_MAX;
        sortPayloads[threadId.x] = tetId;
    } else {
        const float4 sphere = spheres[tetId];
        const float3 toSphere = sphere.xyz - rayOrigin;
        float  power = dot(toSphere, toSphere) - (sphere.w * sphere.w);
        // if (length(toSphere) > 1e2 || (sphere.w * sphere.w) > 1e2) {
        if ((sphere.w * sphere.w) > 1e4) {
            power = 1e20;
        }

        uint key = order_preserving_float_map(power);
        if (sphere.w == 0 || power != power || isnan(power) || isinf(power))
            key = UINT32_MAX;

        sortKeys[threadId.x] = key;
        sortPayloads[threadId.x] = tetId;
    }
}