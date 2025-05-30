import Scene.TetrahedronScene;
import Rose.Core.PackedTypes;

using namespace vkDelTet;
using namespace RoseEngine;

ByteAddressBuffer shCoeffs;
ByteAddressBuffer primPositions;
RWByteAddressBuffer outputColors;
uniform float3 rayOrigin;
uniform uint numPrimitives;

#define softplus(x, beta) ( (1.0 / beta) * log(1.0 + exp(beta*x)) )

#ifndef NUM_COEFFS
#define NUM_COEFFS 15
#endif

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

float3 load_sh_coeff(uint offset, uint coeff_i) {
    return shCoeffs.Load<float3>((offset + coeff_i) * sizeof(float3));
}

float3 eval_deg0(uint offset, float3 dir) {
    return SH_C0 * load_sh_coeff(offset, 0) + 0.5f;
}

float3 eval_deg1(uint offset, float3 dir) {
    return -SH_C1 * dir.y * load_sh_coeff(offset, 1) +
            SH_C1 * dir.z * load_sh_coeff(offset, 2) +
           -SH_C1 * dir.x * load_sh_coeff(offset, 3);
}

float3 eval_deg2(uint offset, float3 dir) {
    const float xx = dir.x * dir.x;
    const float yy = dir.y * dir.y;
    const float zz = dir.z * dir.z;
    const float xy = dir.x * dir.y;
    const float yz = dir.y * dir.z;
    const float xz = dir.x * dir.z;
    return SH_C2[0] * load_sh_coeff(offset, 4) * xy +
           SH_C2[1] * load_sh_coeff(offset, 5) * yz +
           SH_C2[2] * load_sh_coeff(offset, 6) * (2.0f * zz - xx - yy) +
           SH_C2[3] * load_sh_coeff(offset, 7) * xz +
           SH_C2[4] * load_sh_coeff(offset, 8) * (xx - yy);
}

float3 eval_deg3(uint offset, float3 dir) {
    const float x = dir.x;
    const float y = dir.y;
    const float z = dir.z;
    const float xx = x * x, yy = y * y, zz = z * z;
    const float xy = x * y, yz = y * z, xz = x * z;
    return SH_C3[0] * y * (3.0f * xx - yy) * load_sh_coeff(offset, 9) +
           SH_C3[1] * xy * z * load_sh_coeff(offset, 10) +
           SH_C3[2] * y * (4.0f * zz - xx - yy) * load_sh_coeff(offset, 11) +
           SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * load_sh_coeff(offset, 12) +
           SH_C3[4] * x * (4.0f * zz - xx - yy) * load_sh_coeff(offset, 13) +
           SH_C3[5] * z * (xx - yy) * load_sh_coeff(offset, 14) +
           SH_C3[6] * x * (xx - 3.0f * yy) * load_sh_coeff(offset, 15);
}
// --------------------------------------------------------------------------------

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 index: SV_DispatchThreadID) {
    if (index.x >= numPrimitives)
        return;

    const float3 pos = primPositions.Load<float3>((index.x / NUM_COEFFS) * sizeof(float3));
    const float3 dir = normalize(pos - rayOrigin);

    const uint offset = index.x * NUM_COEFFS;

    float3 c = 0;
    #if NUM_COEFFS > 0
    c += eval_deg0(offset, dir);
    #endif
    #if NUM_COEFFS > 1
    c += eval_deg1(offset, dir);
    #endif
    #if NUM_COEFFS > 4
    c += eval_deg2(offset, dir);
    #endif
    #if NUM_COEFFS > 9
    c += eval_deg3(offset, dir);
    #endif

    c = softplus(c, 10);

    // colors[index.x] = D3DX_FLOAT4_to_R10G10B10A2_UNORM(float4(c.bgr, 1));
    outputColors.Store<float3>(index.x*sizeof(float3), c);
}