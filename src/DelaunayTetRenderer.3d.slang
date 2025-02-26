import Rose.Core.Quaternion;
import Rose.Core.MathUtils;
import TetrahedronScene;

#ifndef REORDER_TETS
#define REORDER_TETS 0
#endif
static const bool kReorderTets = REORDER_TETS;

ParameterBlock<TetrahedronScene> scene;
StructuredBuffer<uint2> sortBuffer;

uniform float4x4 view;
uniform float4x4 projection;
uniform float4x4 invProjection;
uniform float2   outputResolution;
uniform float    densityThreshold;
uniform float    farZ;
uniform quat     cameraRotation;
uniform float3   rayOrigin;
uniform float    densityScale;
uniform float    pointSize;

struct v2f {
    float4 pos : SV_Position;
    float4 color : COLOR0;
    float4x3 tetVerts : TEXCOORD0;
};

[shader("vertex")]
v2f vs_spheres(uint tetId: SV_VertexID, out float psize: SV_PointSize) {
    if (!kReorderTets) tetId = sortBuffer[tetId].y;

    v2f o = {};

    float4 sphere = scene.spheres[tetId];

	float3 sphereCameraPos = mul(view, float4(sphere.xyz, 1.0)).xyz;

    float4 c = scene.load_rgb_density(tetId);

    if (c.a > densityThreshold) {
		c.a *= densityScale;
		#ifdef SCALE_DENSITY_BY_SIZE
		c.a *= sphere.w;
		#endif

		o.tetVerts = scene.load_tet_vertices(tetId);

        o.pos = mul(projection, float4(sphereCameraPos, 1));
        float cameraDist = length(sphereCameraPos);
        psize = pointSize / cameraDist;
    } else {
        o.pos = -2;
        c = 0;
        psize = 0;
    }
    o.color = c;
    return o;
}

static const uint3 kTetTriangles[4] = {
	// 0 ---- 1
	//  \    / \
	//   \  /   \
	//     2 ----3
	uint3(0, 1, 2),
	uint3(1, 2, 3),
	uint3(2, 3, 0),
	uint3(3, 0, 1),
};

[shader("vertex")]
v2f vs_tet(uint tetId: SV_InstanceID, uint vertexId: SV_VertexID) {
    if (!kReorderTets) tetId = sortBuffer[tetId].y;

    v2f o = {};
    float4 c = scene.load_rgb_density(tetId);

    if (c.a > densityThreshold) {
		c.a *= densityScale;
		#ifdef SCALE_DENSITY_BY_SIZE
		c.a *= scene.spheres[tetId].w;
		#endif

        uint3 tri = kTetTriangles[(vertexId / 3) % 4];

        o.tetVerts = scene.load_tet_vertices(tetId);

        // centroid and camera must be on opposite sides of triangle
        float3 centroid = 0.25 * (o.tetVerts[0] + o.tetVerts[1] + o.tetVerts[2] + o.tetVerts[3]);
        float3 toCentroid = centroid  - o.tetVerts[tri[0]];
        float3 toCamera   = rayOrigin - o.tetVerts[tri[0]];

        float3 n = cross(
            o.tetVerts[tri[1]] - o.tetVerts[tri[0]],
            o.tetVerts[tri[2]] - o.tetVerts[tri[0]]);

        if (dot(toCentroid, n) * dot(toCamera, n) > 0) {
			float3 viewVert = mul(view, float4(o.tetVerts[tri[vertexId % 3]], 1)).xyz;
			o.pos = mul(projection, float4(viewVert, 1));
        } else {
            o.pos = -2;
            c = 0;
		}
    } else {
        o.pos = -2;
        c = 0;
    }
	o.color = c;
    return o;
}

[shader("fragment")]
float4 fs_color(float4 color: COLOR) : SV_Target {
    float T = exp(-color.a);
    return float4((1 - T) * color.rgb, T);
}

[shader("fragment")]
float4 fs_isect(v2f i) : SV_Target {
    float4 ndc = float4(((i.pos.xy + 0.5) / outputResolution) * 2 - 1, 0.5, 1);
    float4 viewPos = mul(invProjection, ndc);
    float3 rayDirection = q_rotate_vector(cameraRotation, normalize(viewPos.xyz));

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



struct v2f_tri {
    float4 pos : SV_Position;
    float4 color : COLOR;
};

[shader("vertex")]
v2f_tri vs_tri(float3 vertex: POSITION, float4 color: COLOR) {
    v2f_tri o;
    o.pos = mul(projection, mul(view, float4(vertex, 1)));
    o.color = color;
    return o;
}

[shader("fragment")]
float4 fs_tri(v2f_tri i) : SV_Target {
    float T = exp(-i.color.a);
    return float4((1 - T) * i.color.rgb, T);
}
