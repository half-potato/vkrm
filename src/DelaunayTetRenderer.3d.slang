import Rose.Core.Quaternion;
import Rose.Core.MathUtils;
import TetrahedronScene;
import SortUtils;

using namespace RoseEngine;

#ifndef REORDER_TETS
#define REORDER_TETS 0
#endif
#ifndef SCALE_DENSITY_BY_SIZE
#define SCALE_DENSITY_BY_SIZE 0
#endif
static const bool kReorderTets        = (bool)REORDER_TETS;
static const bool kScaleDensityBySize = (bool)SCALE_DENSITY_BY_SIZE;

ParameterBlock<TetrahedronScene> scene;
StructuredBuffer<uint2>  sortBuffer;
StructuredBuffer<uint>   triangles;
StructuredBuffer<float4> triangleColors;
RasterizerOrderedTexture2D<float> depthBuffer;

uniform float4x4 view;
uniform float4x4 projection;
uniform float4x4 viewProjection;
uniform float4x4 invProjection;
uniform float2   outputResolution;
uniform float    densityThreshold;
uniform quat     cameraRotation;
uniform float3   rayOrigin;
uniform float    densityScale;
uniform float    pointSize;

struct v2f_pos {
    float4 pos : SV_Position;
    float4 color : COLOR0;
    float3 viewPos : TEXCOORD0;
};

struct v2f_isect {
    float4 pos : SV_Position;
    float4 color : COLOR0;
    float4x3 tetVerts : TEXCOORD0;
};

#ifdef VS_NO_ISECT
typedef v2f_pos v2f;
#else
typedef v2f_isect v2f;
#endif

float lengthsq(float2 v) { return dot(v, v); }

[shader("vertex")]
v2f vs_sphere_center(uint tetId : SV_VertexID, out float psize : SV_PointSize) {
    if (!kReorderTets) tetId = sortBuffer[tetId].y;

    v2f o = {};

    float4 sphere = scene.spheres[tetId];

    float4 c = scene.load_rgb_density(tetId);

    if (c.a > densityThreshold) {
		c.a *= densityScale;
		if (kScaleDensityBySize) c.a *= sphere.w;

        float3 pos;
        psize = pointSize;

        #ifdef VS_NO_ISECT
        o.pos = mul(view, float4(sphere.xyz, 1));
        psize /= length(o.pos);
        o.pos = mul(projection, o.pos);
        #else
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

        float2 mn = min(min(verts[0].xy, verts[1].xy), min(verts[2].xy, verts[3].xy));
        float2 mx = max(max(verts[0].xy, verts[1].xy), max(verts[2].xy, verts[3].xy));
        o.pos = float4(lerp(mn, mx, 0.5), max(max(verts[0].z, verts[1].z), max(verts[2].z, verts[3].z)), 1);

        float2 extent = outputResolution * (mx - mn);
        psize = max(extent.x, extent.y) * 0.5;

        #endif
    } else {
        o.pos = -2;
        c = 0;
        psize = 0;
    }
    o.color = c;
    return o;
}

[shader("vertex")]
v2f vs_tet(uint tetId : SV_InstanceID, uint vertexId : SV_VertexID) {
    if (!kReorderTets) tetId = sortBuffer[tetId].y;

    v2f o = {};
    float4 c = scene.load_rgb_density(tetId);

    if (c.a > densityThreshold) {
		c.a *= densityScale;
		if (kScaleDensityBySize) c.a *= scene.spheres[tetId].w;
        
        // get triangle indices within tet
        const uint tetVertexId = TetrahedronScene::kTetTriangles[(vertexId / 3) % 4][vertexId % 3];

        float3 vertex;

        #ifdef VS_NO_ISECT
        vertex = scene.load_vertex(tetId, tetVertexId);
        o.viewPos = mul(view, float4(vertex, 1)).xyz;
        #else
        // load all tet vertices
        o.tetVerts = scene.load_tet_vertices(tetId);
        vertex = o.tetVerts[tetVertexId];
        #endif

        o.pos = mul(viewProjection, float4(vertex, 1));
    } else {
        o.pos = -2;
        c = 0;
    }
	o.color = c;
    return o;
}

[shader("fragment")]
float4 fs_color(float4 color : COLOR) : SV_Target {
    float T = exp(-color.a);
    return float4((1 - T) * color.rgb, T);
}

[shader("fragment")]
float4 fs_isect(v2f_isect i) : SV_Target {
    float4 ndc = float4((i.pos.xy / outputResolution) * 2 - 1, 0.5, 1);
    float3 rayDirection = normalize(q_rotate_vector(cameraRotation, mul(invProjection, ndc).xyz));

    float4 c = float4(0, 0, 0, 1);

    float2 t;
    if (intersect_ray_tetrahedron(rayOrigin, rayDirection, i.tetVerts, t)) {
        if (t.y > 0) {
            float dt = t.y - max(t.x, 0);
            float T = exp(-dt * i.color.a);
            c = float4((1 - T) * i.color.rgb, T);
        }
    }

    return c;
}

[shader("fragment")]
float4 fs_depth(v2f_pos i) : SV_Target {
    float t = length(i.viewPos);
    float t0 = depthBuffer[(uint2)i.pos.xy];
    depthBuffer[(uint2)i.pos.xy] = t;

    float dt = max(0, t - t0);

    float T = exp(-dt * i.color.a);
    return float4((1 - T) * i.color.rgb, T);
}
