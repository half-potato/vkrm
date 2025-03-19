import Rose.Core.MathUtils;
import Scene.TetrahedronScene;
import SortUtils;

using namespace vkDelTet;

ParameterBlock<TetrahedronScene> scene;
StructuredBuffer<uint2> sortBuffer;

uniform float4x4 viewProjection;
uniform float    densityThreshold;
uniform float    pointSize;

struct v2f {
    float4 pos : SV_Position;
    float4 color : COLOR0;
    float  psize : SV_PointSize;
};

[shader("vertex")]
v2f vsmain(uint tetId : SV_VertexID) {
    tetId = sortBuffer[tetId].y;

    v2f o = {};

    o.color = scene.load_rgb_density(tetId);

    if (o.color.a > densityThreshold) {
        const float4 sphere = scene.spheres[tetId];
        o.color.a *= sphere.w;
        o.pos = mul(viewProjection, float4(sphere.xyz, 1));
        o.psize = pointSize / length(o.pos);
    } else {
        o.pos = -2;
        o.color = 0;
        o.psize = 0;
    }
        
    return o;
}

[shader("fragment")]
float4 fsmain(float4 color : COLOR) : SV_Target {
    return EvalTetColor(color);
}
