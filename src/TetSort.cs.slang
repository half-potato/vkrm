import Rose.Core.MathUtils;
import Rose.Core.Quaternion;
import Scene.TetrahedronScene;
import SortUtils;

using namespace vkDelTet;

ParameterBlock<TetrahedronScene> scene;
RWStructuredBuffer<uint2> sortPairs;
uniform float3 rayOrigin;

[shader("compute")]
[numthreads(64, 1, 1)]
void createPairs(uint3 threadId: SV_DispatchThreadID) {
    uint tetId = threadId.x;
    if (tetId >= scene.numTets)
        return;

    float4 sphere = scene.spheres[tetId];
    float3 toSphere = sphere.xyz - rayOrigin;
    float power = dot(toSphere, toSphere) - (sphere.w * sphere.w);

    uint key = order_preserving_float_map(power);

    if (sphere.w == 0 || power != power || isnan(power) || isinf(power))
        key = UINT32_MAX;

    sortPairs[tetId] = uint2(key, tetId);
}
