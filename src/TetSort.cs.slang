import Rose.Core.MathUtils;
import Rose.Core.Quaternion;
import Scene.TetrahedronScene;
import SortUtils;

using namespace vkDelTet;

StructuredBuffer<float4>  spheres;
RWStructuredBuffer<uint2> sortPairs;

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
    sortPairs[threadId.x] = uint2(UINT32_MAX, threadId.x);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void updatePairs(uint3 threadId: SV_DispatchThreadID) {
    if (threadId.x >= numSpheres)
        return;

    // const uint tetId = visibleTets[threadId.x];//sortPairs[threadId.x].y;
    const uint tetId = sortPairs[threadId.x].y;
    if (markedTets.Load(tetId) == 0) {
        sortPairs[threadId.x] = uint2(UINT32_MAX, tetId);
    } else {

        const float4 sphere = spheres[tetId];
        const float3 toSphere = sphere.xyz - rayOrigin;
        const float  power = dot(toSphere, toSphere) - (sphere.w * sphere.w);

        uint key = order_preserving_float_map(power);
        if (sphere.w == 0 || power != power || isnan(power) || isinf(power))
            key = UINT32_MAX;

        sortPairs[threadId.x] = uint2(key, tetId);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void reorderTets(uint3 threadId: SV_DispatchThreadID) {
    if (threadId.x >= numSpheres)
        return;/*
    const uint tetId = sortPairs[threadId.x].y;
    sortedColors.Store(threadId.x, scene.vertexColors.Load(tetId));
    sortedIndices[threadId.x] = scene.tetIndices.Load4(tetId*sizeof(uint4));
    sortedSpheres[threadId.x] = scene.tetCircumspheres[tetId];*/
}
