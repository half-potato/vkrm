import Rose.Core.MathUtils;
import Scene.TetrahedronScene;
import SortUtils;

using namespace vkDelTet;

ParameterBlock<TetrahedronScene> scene;
StructuredBuffer<uint2> sortBuffer;
ByteAddressBuffer tetCentroids;
ByteAddressBuffer tetColors; // precomputed SH data

uniform float4x4 viewProjection;
uniform float    densityThreshold;
uniform float    pointSize;

struct v2f {
    float4 pos : SV_Position;
    float4 color : COLOR0;
    float  psize : SV_PointSize;
};

[shader("vertex")]
v2f vsmain(uint vertId: SV_VertexID) {
    const uint tetId = sortBuffer[vertId].y;

    float T = exp(-scene.load_tet_density(tetId));

    v2f o = {};
    o.color = float4(tetColors.Load<float3>(tetId * sizeof(float3)) * (1 - T), T);

    o.pos = mul(viewProjection, float4(tetCentroids.Load<float3>(tetId*sizeof(float3)), 1));
    o.psize = pointSize / length(o.pos);
        
    return o;
}

[shader("fragment")]
float4 fsmain(float4 color: COLOR) : SV_Target {
    return color;
}
