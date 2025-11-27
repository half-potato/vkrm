import Rose.Core.MathUtils;
import Rose.Core.Quaternion;
import Scene.TetrahedronScene;
import SortUtils;

using namespace vkDelTet;

ParameterBlock<TetrahedronScene> scene;
StructuredBuffer<uint> sortPayloads;
ByteAddressBuffer tetColors; // precomputed SH data

uniform float4x4 viewProjection;
uniform float4x4 invProjection;
uniform quat     cameraRotation;
uniform float3   rayOrigin;
uniform float    densityThreshold;
uniform float2   outputResolution;
uniform int      selectedVertex;
uniform int      selectedTet;

struct v2f {
    float4 pos : SV_Position;
    float4 color : COLOR0;
};

[shader("vertex")]
v2f vsmain(uint tetId : SV_VertexID, out float psize : SV_PointSize) {
}


[shader("fragment")]
float4 fsmain(v2f v, float4 fragPos : SV_Position) : SV_Target {
}

