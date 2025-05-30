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
v2f vsmain(uint vertId : SV_VertexID) {
    v2f o = {};

    o.color = float4(1,1,1,0);

    o.pos = mul(viewProjection, float4(scene.load_vertex(vertId), 1));
    o.psize = pointSize / length(o.pos);
        
    return o;
}

[shader("fragment")]
float4 fsmain(float4 color: COLOR) : SV_Target {
    return color;
}
