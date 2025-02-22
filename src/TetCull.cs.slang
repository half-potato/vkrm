import Rose.Core.MathUtils;
import Rose.Core.Quaternion;
import TetrahedronScene;

ParameterBlock<TetrahedronScene> scene;

RWStructuredBuffer<uint> visibleTets;

uniform float4x4 view;
uniform float4x4 projection;
uniform float3   cameraPosition;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 threadId: SV_DispatchThreadID) {
    uint tetId = threadId.x;
    if (tetId >= scene.numTets)
        return;

    float4 sphere = scene.spheres[tetId];

    float3 vc = mul(view, float4(sphere, 1));


}

/*
// https://www.shadertoy.com/view/XdBGzd
float sphereArea;
float2 sphereScreenPos;
{
	// transform to camera space

	float r2 = sphere.w * sphere.w;
	float z2 = c.z*c.z;
	float l2 = dot(c,c);

	sphereArea = -M_PI*focalLength*focalLength*r2*sqrt(abs((l2-r2)/(r2-z2)))/(r2-z2);

	//return area;


	//-- debug stuff ---

	// axis
	float2 axa = focalLength*sqrt(-r2*(r2-l2)/((l2-z2)*(r2-z2)*(r2-z2)))*float2( c.x,c.y);
	float2 axb = focalLength*sqrt(-r2*(r2-l2)/((l2-z2)*(r2-z2)*(r2-l2)))*float2(-c.y,c.x);

	//area = length(axa)*length(axb)*3.141593;

	// center
	sphereScreenPos = focalLength*c.z*c.xy/(z2-r2);
}
*/