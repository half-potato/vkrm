#define GROUP_SIZE 32
#define TETS_PER_GROUP (GROUP_SIZE / 4)

#define NUM_COEFFS 16

#define COEFFS_PER_BUF 8

import Rose.Core.MathUtils;
import Rose.Core.Quaternion;
import Scene.TetrahedronScene;
import SortUtils;
import Rose.Core.PackedTypes;

#include <Rose/Core/Bitfield.h>

using namespace vkDelTet;
using namespace RoseEngine;

ParameterBlock<TetrahedronScene> scene;
StructuredBuffer<uint> sortPayloads;
ByteAddressBuffer tetColors; // precomputed SH data

ByteAddressBuffer shCoeffs[NUM_COEFFS / COEFFS_PER_BUF];
ByteAddressBuffer tetCentroids;
RWStructuredBuffer<float> tetOffsets;

uniform float4x4 viewProjection;
uniform float3   rayOrigin;
uniform float    densityThreshold;

struct v2f {
    float4 pos : SV_Position;

    [[vk::location(0)]] float3 rayDir;
    [[vk::location(1)]] float4 planeNumerators;
    [[vk::location(2)]] float4 planeDenominators;
    
    // Tet data, constant for vertices in a tet
    [[vk::location(3)]] nointerpolation float  tetDensity;
    [[vk::location(4)]] nointerpolation float3 baseColor;
    [[vk::location(6)]] float dc_dt;

    // [[vk::location(4)]]  nointerpolation float3 c0;
    // [[vk::location(5)]]  nointerpolation float3 c1;
    // [[vk::location(6)]]  nointerpolation float3 c2;
    // [[vk::location(7)]]  nointerpolation float3 c3;
    // [[vk::location(8)]]  nointerpolation float3 v0;
    // [[vk::location(9)]]  nointerpolation float3 v1;
    // [[vk::location(10)]] nointerpolation float3 v2;
    // [[vk::location(11)]] nointerpolation float3 v3;
};

static const bool kUseFP16 = false;

#define softplus(x, beta) ( (1.0 / beta) * log(1.0 + exp(beta*(x))) )

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
                const uint2 bits = shCoeffs[bufId].Load<uint2>((address + j) * 6);
                return float3(f16tof32(bits[0]), f16tof32(bits[0] >> 16), f16tof32(bits[1]));
            } else {
                return shCoeffs[bufId].Load<float3>((address + j) * sizeof(float3));
            }
        }
    }
};

float3 eval_sh_partial(uint local_idx, SHCoeffs c, float3 dir)
{
    // Pre-calculate all direction-dependent terms needed by any thread
    const float x = dir.x;
    const float y = dir.y;
    const float z = dir.z;
    const float xx = dir.x * dir.x;
    const float yy = dir.y * dir.y;
    const float zz = dir.z * dir.z;
    const float xy = dir.x * dir.y;
    const float yz = dir.y * dir.z;
    const float xz = dir.x * dir.z;

    float3 partial_sum = float3(0.0f, 0.0f, 0.0f);

    switch (local_idx)
    {
        // Thread 0: Handles coefficients 0-3 (All of degree 0 and 1)
        case 0:
#if NUM_COEFFS > 1
            return eval_deg0(c, dir) + eval_deg1(c, dir);
#else
            return eval_deg0(c, dir);
#endif
            break;

        // Thread 1: Handles coefficients 4-7 (First part of degree 2)
        case 1:
#if NUM_COEFFS > 4
            return SH_C2[0] * c[4] * xy +
                   SH_C2[1] * c[5] * yz +
                   SH_C2[2] * c[6] * (2.0f * zz - xx - yy) +
                   SH_C2[3] * c[7] * xz;
    #endif
            break;

        // Thread 2: Handles coefficients 8-11 (Last part of degree 2, first part of degree 3)
        case 2:
            // Degree 2
            partial_sum += SH_C2[4] * c[8] * (xx - yy);
            // Degree 3
#if NUM_COEFFS > 9
            partial_sum += SH_C3[0] * y * (3.0f * xx - yy) * c[9];
            partial_sum += SH_C3[1] * xy * z * c[10];
            partial_sum += SH_C3[2] * y * (4.0f * zz - xx - yy) * c[11];
#endif
            return partial_sum;
            break;

        // Thread 3: Handles coefficients 12-15 (Last part of degree 3)
        case 3:
#if NUM_COEFFS > 9
            partial_sum += SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * c[12];
            partial_sum += SH_C3[4] * x * (4.0f * zz - xx - yy) * c[13];
            partial_sum += SH_C3[5] * z * (xx - yy) * c[14];
            partial_sum += SH_C3[6] * x * (xx - 3.0f * yy) * c[15];
#endif
            return partial_sum;
            break;
    }

    return partial_sum;
}

[shader("mesh")]
[outputtopology("triangle")]
[numthreads(GROUP_SIZE, 1, 1)]
void meshmain(
    in uint3 threadId: SV_DispatchThreadID,
    OutputVertices<v2f, GROUP_SIZE>  vertices,
    OutputIndices<uint3, GROUP_SIZE> triangles) {
        
    SetMeshOutputCounts(GROUP_SIZE, GROUP_SIZE);

    const uint meshVertexId = threadId.x % GROUP_SIZE;
    const uint tetVertexId  = threadId.x % 4;

    v2f o = {};
    o.pos = 0;

    if (threadId.x/4 < scene.numTets)
    {
        const uint tetId = sortPayloads[threadId.x/4];
        // o.baseColor     = tetColors.Load<float3>(tetId * sizeof(float3));

        o.tetDensity = scene.load_tet_density(tetId);
        if (o.tetDensity > densityThreshold)
        {
            const uint vertexId = scene.load_index(tetId, tetVertexId);
            const float3 vertex = scene.load_vertex(vertexId);
            o.pos = mul(viewProjection, float4(vertex, 1));
            o.rayDir = vertex - rayOrigin;

            const uint baseLaneId = (WaveGetLaneIndex() / 4) * 4;

            #define WaveReadTetVar(x, id)        WaveReadLaneAt(x, baseLaneId + id)
            #define WaveReadTetVarRel(x, offset) WaveReadTetVar(x, (WaveGetLaneIndex() + offset) % 4)

            const uint3 tri = TetrahedronScene::kTetTriangles[tetVertexId];

            // outward facing normal
            const float3 n = cross(
                WaveReadTetVar(vertex, tri[2]) - WaveReadTetVar(vertex, tri[0]),
                WaveReadTetVar(vertex, tri[1]) - WaveReadTetVar(vertex, tri[0]));

            const float num = dot(n, WaveReadTetVar(vertex, tri[0]) - rayOrigin);

            o.planeNumerators = float4(
                WaveReadTetVar(num, 0),
                WaveReadTetVar(num, 1),
                WaveReadTetVar(num, 2),
                WaveReadTetVar(num, 3) );
            o.planeDenominators = float4(
                dot(WaveReadTetVar(n, 0), o.rayDir),
                dot(WaveReadTetVar(n, 1), o.rayDir),
                dot(WaveReadTetVar(n, 2), o.rayDir),
                dot(WaveReadTetVar(n, 3), o.rayDir) );

            // o.baseColor     = tetColors.Load<float3>(tetId * sizeof(float3));
            // Calculate SH collaboratively

            const SHCoeffs coeffs = SHCoeffs(tetId);
            const float3 pos = tetCentroids.Load<float3>(tetId * sizeof(float3));
            const float3 dir = normalize(pos - rayOrigin);
            float3 partialColor = eval_sh_partial(tetVertexId, coeffs, dir);
            float3 p0 = WaveReadTetVar(partialColor, 0);
            float3 p1 = WaveReadTetVar(partialColor, 1);
            float3 p2 = WaveReadTetVar(partialColor, 2);
            float3 p3 = WaveReadTetVar(partialColor, 3);

            const float3 v0 = WaveReadTetVar(vertex, 0);

            const float offset = tetOffsets.Load(tetId);
            float3 colorGradient = scene.load_tet_gradient(tetId);
            float offset2 = dot(rayOrigin - v0, colorGradient);
            o.baseColor = softplus(p0+p1+p2+p3, 10);
            // o.baseColor     = softplus(tetColors.Load<float3>(tetId * sizeof(float3)), 10);
            o.baseColor += offset + offset2;

            o.dc_dt = dot(colorGradient, o.rayDir);

            // float3 color = 0;
            // color += scene.load_tet_vertex_color(tetId, tetVertexId).rgb;
            // color += vertexColors.Load(vertexId).rgb;
            // color = softplus(color, 10);

            // o.c0 = WaveReadTetVar(color, 0);
            // o.c1 = WaveReadTetVar(color, 1);
            // o.c2 = WaveReadTetVar(color, 2);
            // o.c3 = WaveReadTetVar(color, 3);
            //
            // o.v0 = WaveReadTetVar(vertex, 0);
            // o.v1 = WaveReadTetVar(vertex, 1);
            // o.v2 = WaveReadTetVar(vertex, 2);
            // o.v3 = WaveReadTetVar(vertex, 3);
        }
    }

    vertices[meshVertexId] = o;
    triangles[meshVertexId] = (meshVertexId/4)*4 + TetrahedronScene::kTetTriangles[tetVertexId];
}

[shader("fragment")]
float4 fsmain(v2f v) : SV_Target {
    {
        const float d = length(v.rayDir);
        v.rayDir /= d;
        v.planeDenominators /= d;
        v.dc_dt /= d;
    }

    float3 rayDir = v.rayDir;
    float opticalDepth = v.tetDensity;
    float4 all_t = v.planeNumerators / v.planeDenominators;

    float4 t_enter = select(v.planeDenominators > 0.0f, all_t, float4(-FLT_MAX));
    float4 t_exit  = select(v.planeDenominators < 0.0f, all_t, float4(FLT_MAX));

    // 3. Find the latest entry point and the earliest exit point using horizontal max/min.
    // This finds the final intersection interval [t.x, t.y].
    float2 t = float2(
        max(t_enter.x, max(t_enter.y, max(t_enter.z, t_enter.w))),
        min(t_exit.x, min(t_exit.y, min(t_exit.z, t_exit.w)))
    );
    
    // bool parallel_miss = any((abs(v.planeDenominators) < 1e-6f) & (v.planeNumerators > 0.0f));
    // if (parallel_miss) return float4(0,0,0,1);

    opticalDepth *= t.y - t.x;
    const float T = exp(-opticalDepth);
    float3 color = {
        integrate_channel(t.x, t.y, v.baseColor.r, v.dc_dt, v.tetDensity),
        integrate_channel(t.x, t.y, v.baseColor.g, v.dc_dt, v.tetDensity),
        integrate_channel(t.x, t.y, v.baseColor.b, v.dc_dt, v.tetDensity)
    };

    return float4(color, T);
}
