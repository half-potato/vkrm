import Scene.TetrahedronScene;
import Rose.Core.PackedTypes;

using namespace vkDelTet;
using namespace RoseEngine;

ParameterBlock<TetrahedronScene> scene;
RWStructuredBuffer<float4> vertexColors;
uniform float3 rayOrigin;

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

struct SphericalHarmonicEvaluator {
    uint offset;

    float3 get_coeffs(uint coeff_i) {
        uint idx = 3 * (offset + coeff_i - 1);
        return float3(
            scene.vertexSH.Load(idx + 0),
            scene.vertexSH.Load(idx + 1),
            scene.vertexSH.Load(idx + 2));
    }

    /*
    float3 eval_deg0(float3 dir) {
        return SH_C0 * get_coeffs(0) + 0.5f;
    }
    */
    float3 eval_deg1(float3 dir) {
        return -SH_C1 * dir.y * get_coeffs(1) +
                SH_C1 * dir.z * get_coeffs(2) +
               -SH_C1 * dir.x * get_coeffs(3);
    }

    float3 eval_deg2(float3 dir) {
        const float xx = dir.x * dir.x;
        const float yy = dir.y * dir.y;
        const float zz = dir.z * dir.z;
        const float xy = dir.x * dir.y;
        const float yz = dir.y * dir.z;
        const float xz = dir.x * dir.z;
        return SH_C2[0] * get_coeffs(4) * xy +
               SH_C2[1] * get_coeffs(5) * yz +
               SH_C2[2] * get_coeffs(6) * (2.0f * zz - xx - yy) +
               SH_C2[3] * get_coeffs(7) * xz +
               SH_C2[4] * get_coeffs(8) * (xx - yy);
    }
};

// --------------------------------------------------------------------------------

[shader("compute")]
[numthreads(32,1,1)]
void main(uint3 index: SV_DispatchThreadID) {
    const uint vertexId = index.x;
    if (vertexId >= scene.numVertices)
        return;

    const float3 dir = normalize(scene.load_vertex(vertexId) - rayOrigin);

    SphericalHarmonicEvaluator sh = { vertexId * scene.numVertexSHCoeffs };

    float3 c = 0;
    // c += sh.eval_deg0(dir);
    c += sh.eval_deg1(dir);
    c += sh.eval_deg2(dir);

    // vertexColors[vertexId] = D3DX_FLOAT4_to_R10G10B10A2_UNORM(float4(c.bgr, 1));
    vertexColors[vertexId] = float4(c, 1);
}