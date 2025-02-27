import Rose.Core.MathUtils;
import Rose.Core.Quaternion;
import Scene.TetrahedronScene;
import SortUtils;

using namespace vkDelTet;

ParameterBlock<TetrahedronScene> scene;
StructuredBuffer<uint2> sortBuffer;

uniform float4x4 viewProjection;
uniform float4x4 invProjection;
uniform quat     cameraRotation;
uniform float3   rayOrigin;
uniform float    densityThreshold;
uniform float2   outputResolution;

struct v2f {
    float4 pos : SV_Position;
    float4 color : COLOR0;
    float4x3 tetVerts : TEXCOORD0;
};

[shader("vertex")]
v2f vsmain(uint tetId : SV_VertexID, out float psize : SV_PointSize) {
    tetId = sortBuffer[tetId].y;

    v2f o = {};

    const float4 sphere = scene.spheres[tetId];
    const float4 tetColor = scene.load_rgb_density(tetId);

    if (tetColor.a > densityThreshold) {
        o.tetVerts = scene.load_tet_vertices(tetId);

        float4x4 verts = float4x4(
            float4(o.tetVerts[0], 1),
            float4(o.tetVerts[1], 1),
            float4(o.tetVerts[2], 1),
            float4(o.tetVerts[3], 1));
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
    o.color = tetColor;
    return o;
}


[shader("fragment")]
float4 fsmain(v2f i) : SV_Target {
    float4 fragColor = float4(0, 0, 0, 1);

    float4 ndc = float4((i.pos.xy / outputResolution) * 2 - 1, 0.5, 1);
    float3 rayDirection = normalize(q_rotate_vector(cameraRotation, mul(invProjection, ndc).xyz));

    float2 t;
    if (intersect_ray_tetrahedron(rayOrigin, rayDirection, i.tetVerts, t)) {
        if (t.y > 0) {
            float dt = t.y - max(t.x, 0);
            float T = exp(-dt * i.color.a);
            fragColor = float4((1 - T) * i.color.rgb, T);
        }
    }

    return fragColor;
}