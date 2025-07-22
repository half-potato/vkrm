import Scene.TetrahedronScene;
import Rose.Core.PackedTypes;

using namespace vkDelTet;
using namespace RoseEngine;

#ifndef NUM_COEFFS
#define NUM_COEFFS 16
#endif

#define COEFFS_PER_BUF 16
static const bool kUseFP16 = true;


ParameterBlock<TetrahedronScene> scene;
ByteAddressBuffer shCoeffs[NUM_COEFFS / COEFFS_PER_BUF];
ByteAddressBuffer tetCentroids;
RWByteAddressBuffer outputColors;
uniform float3 rayOrigin;
uniform uint numPrimitives;
RWStructuredBuffer<float> tetOffsets;
RWStructuredBuffer<uint> visibleTets;        // The final, compact list of tet IDs
RWStructuredBuffer<uint> drawArgs;   // Buffer to hold arguments for an indirect draw call
// RWStructuredBuffer<float3> spheres;
RWStructuredBuffer<uint> markedTets;        // The final, compact list of tet IDs

#define softplus(x, beta) ( (1.0 / beta) * log(1.0 + exp(beta*x)) )

// --------------------------------------------------------------------------------
// Spherical harmonics

// Spherical harmonics coefficients
static const float SH_C0 = 0.28209479177387814f;
static const float SH_C1 = 0.4886025119029199f;
static const float SH_C2[] = {
   1.0925484305920792f,
  -1.0925484305920792f,
   0.31539156525252005f,
  -1.0925484305920792f,
   0.5462742152960396f
};
static const float SH_C3[] = {
  -0.5900435899266435f,
   2.890611442640554f,
  -0.4570457994644658f,
   0.3731763325901154f,
  -0.4570457994644658f,
   1.445305721320277f,
  -0.5900435899266435f
};

// helper to pull SH data for a tet

struct SHCoeffs {
    uint address;

    __init(uint tetId) {
        address = tetId * COEFFS_PER_BUF;
    }

    __subscript(uint i) -> float3 {
        get {
            const uint bufId = i / COEFFS_PER_BUF;
            const uint j = i % COEFFS_PER_BUF;
            if (kUseFP16) {
                // Each coefficient is 3 halfs = 6 bytes.
                const uint byte_address = (address + j) * sizeof(uint16_t) * 3;

                // Load the first 4 bytes, containing the packed R and G channels.
                const float r = f16tof32(shCoeffs[bufId].Load<uint16_t>(byte_address));
                const float g = f16tof32(shCoeffs[bufId].Load<uint16_t>(byte_address + sizeof(uint16_t)));
                const float b = f16tof32(shCoeffs[bufId].Load<uint16_t>(byte_address + 2*sizeof(uint16_t)));
// inline float2 D3DX_R16G16_FLOAT_to_FLOAT2(uint packedInput)

                return float3(r, g, b);
            } else {
                return shCoeffs[bufId].Load<float3>((address + j) * sizeof(float3));
            }
        }
    }
};

float3 eval_deg0(SHCoeffs c, float3 dir) {
    return SH_C0 * c[0] + 0.5f;
}

float3 eval_deg1(SHCoeffs c, float3 dir) {
    return -SH_C1 * dir.y * c[1] +
            SH_C1 * dir.z * c[2] +
           -SH_C1 * dir.x * c[3];
}

float3 eval_deg2(SHCoeffs c, float3 dir) {
    const float xx = dir.x * dir.x;
    const float yy = dir.y * dir.y;
    const float zz = dir.z * dir.z;
    const float xy = dir.x * dir.y;
    const float yz = dir.y * dir.z;
    const float xz = dir.x * dir.z;
    return SH_C2[0] * c[4] * xy +
           SH_C2[1] * c[5] * yz +
           SH_C2[2] * c[6] * (2.0f * zz - xx - yy) +
           SH_C2[3] * c[7] * xz +
           SH_C2[4] * c[8] * (xx - yy);
}

float3 eval_deg3(SHCoeffs c, float3 dir) {
    const float x = dir.x;
    const float y = dir.y;
    const float z = dir.z;
    const float xx = x * x, yy = y * y, zz = z * z;
    const float xy = x * y, yz = y * z, xz = x * z;
    return SH_C3[0] * y * (3.0f * xx - yy) * c[9] +
           SH_C3[1] * xy * z * c[10] +
           SH_C3[2] * y * (4.0f * zz - xx - yy) * c[11] +
           SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * c[12] +
           SH_C3[4] * x * (4.0f * zz - xx - yy) * c[13] +
           SH_C3[5] * z * (xx - yy) * c[14] +
           SH_C3[6] * x * (xx - 3.0f * yy) * c[15];
}
// --------------------------------------------------------------------------------

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 index: SV_DispatchThreadID) {
    // if (index.x >= drawArgs[0])
    //     return;
    // const uint tetId = visibleTets[index.x];//sortPairs[threadId.x].y;
    // if (tetId >= numPrimitives)
    //     return;

    // const uint tetId = index.x;
    const uint tetId = index.x;

    if ((tetId >= numPrimitives) || (markedTets.Load(tetId) == 0))
        return;

    const float3 pos = tetCentroids.Load<float3>(tetId * sizeof(float3));
    const float3 dir = normalize(pos - rayOrigin);


    const SHCoeffs coeffs = SHCoeffs(tetId);

    float3 c = 0;
    #if NUM_COEFFS > 0
    c += eval_deg0(coeffs, dir);
    #endif
    #if NUM_COEFFS > 1
    c += eval_deg1(coeffs, dir);
    #endif
    #if NUM_COEFFS > 4
    c += eval_deg2(coeffs, dir);
    #endif
    #if NUM_COEFFS > 9
    c += eval_deg3(coeffs, dir);
    #endif

    c = softplus(c, 10);

    // float3 center = spheres[tetId].xyz;
    // float4x3 tet   = scene.load_tet_vertices(tetId);
    float3 v0 = scene.load_vertex(tetId, 0);
    float3 colorGradient = scene.load_tet_gradient(tetId);
    // float offset = dot(colorGradient, tet[0] - center);

    const float offset = tetOffsets.Load(tetId);
    float offset2 = dot(rayOrigin - v0, colorGradient);
    c += offset + offset2;

    outputColors.Store<float3>(tetId * sizeof(float3), c);
    
    // note: color compression doesnt work for HDR
    // outputColors.Store(tetId * sizeof(uint), D3DX_FLOAT4_to_R10G10B10A2_UNORM(float4(c.bgr, 1)));
}
