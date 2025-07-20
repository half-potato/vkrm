import Rose.Core.MathUtils;
import Rose.Core.Quaternion;
import Scene.TetrahedronScene;
import SortUtils;

using namespace vkDelTet;

ParameterBlock<TetrahedronScene> scene;
StructuredBuffer<uint2> sortBuffer;
ByteAddressBuffer tetColors; // precomputed SH data

uniform float4x4 viewProjection;
uniform float4x4 invProjection;
uniform quat     cameraRotation;
uniform float3   rayOrigin;
uniform float    densityThreshold;
uniform float2   outputResolution;

struct v2f {
    float4 pos : SV_Position;
    float4 color : COLOR0;
    float4x3 tetVertices : TEXCOORD0;
    float  tetDensity;
    float3 baseColor;
    float3 colorGradient;
    float3 rayDir;
};

[shader("vertex")]
v2f vsmain(uint tetId : SV_VertexID, out float psize : SV_PointSize) {
    tetId = sortBuffer[tetId].y;

    v2f o = {};

    o.tetDensity = scene.load_tet_density(tetId);

    if (o.tetDensity > densityThreshold) {
        o.baseColor.xyz = tetColors.Load<float3>(tetId * sizeof(float3));
        o.colorGradient = scene.load_tet_gradient(tetId);
        o.tetVertices = scene.load_tet_vertices(tetId);
        o.rayDir = o.tetVertices[0] - rayOrigin;

        float4x4 verts = float4x4(
            float4(o.tetVertices[0], 1),
            float4(o.tetVertices[1], 1),
            float4(o.tetVertices[2], 1),
            float4(o.tetVertices[3], 1));
        verts = mul(verts, transpose(viewProjection)); // reverse-transpose needed since verts is column matrix

        verts[0] /= verts[0].w;
        verts[1] /= verts[1].w;
        verts[2] /= verts[2].w;
        verts[3] /= verts[3].w;

        const float2 mn = min(min(verts[0].xy, verts[1].xy), min(verts[2].xy, verts[3].xy));
        const float2 mx = max(max(verts[0].xy, verts[1].xy), max(verts[2].xy, verts[3].xy));
        o.pos = float4(lerp(mn, mx, 0.5), max(max(verts[0].z, verts[1].z), max(verts[2].z, verts[3].z)), 1);

        const float2 extent = outputResolution * (mx - mn);
        psize = max(extent.x, extent.y) * 0.5;
    } else {
        o.pos = -2;
        psize = 0;
    }
    return o;
}


[shader("fragment")]
float4 fsmain(v2f v, float4 fragPos : SV_Position) : SV_Target {
    float2 ndc_xy = (fragPos.xy / outputResolution) * 2.0 - 1.0;
    float4 ndc = float4(ndc_xy, v.pos.z / v.pos.w, 1.0);
    float4 world_pos_h = mul(invProjection, ndc);
    float3 world_pos = world_pos_h.xyz / world_pos_h.w;
    float3 rayDir = normalize(world_pos - rayOrigin);

    float opticalDepth = v.tetDensity;

    float2 t = float2(0, FLT_MAX);
    if (!intersect_ray_tetrahedron(rayOrigin, rayDir, v.tetVertices, t)) {
        discard; // Exit if the ray misses the tetrahedron
    }
    
    opticalDepth *= t.y - max(t.x, 0);

    float T = exp(-opticalDepth);

    float dc_dt = dot(v.colorGradient, rayDir);
    // float offset = dot(rayOrigin - v.tetVertices[0], v.colorGradient);
    float3 color = {
        integrate_channel(t.x, t.y, v.baseColor.r, dc_dt, v.tetDensity),
        integrate_channel(t.x, t.y, v.baseColor.g, dc_dt, v.tetDensity),
        integrate_channel(t.x, t.y, v.baseColor.b, dc_dt, v.tetDensity)
    };
    color = opticalDepth > 0 ? color : 0;

    return float4(color, T);
}
