import TetrahedronScene;

using namespace vkDelTet;

ParameterBlock<TetrahedronScene> scene;
RWStructuredBuffer<float4> outputSpheres;

double4 computeCircumsphere(double3 A, double3 B, double3 C, double3 D) {
    // Compute vectors from v0 to other vertices
    double3 a = B - A;
    double3 b = C - A;
    double3 c = D - A;

    // Compute squares of lengths
    double aa = dot(a, a);
    double bb = dot(b, b);
    double cc = dot(c, c);

    // Compute cross products
    double3 cross_bc = cross(b, c);
    double3 cross_ca = cross(c, a);
    double3 cross_ab = cross(a, b);

    // Compute denominator
    double denominator = 2.0f * dot(a, cross_bc);

    // Guard against division by zero
    if (abs(denominator) < 1e-12)
        return 0.0;
        
    double3 relative_circumcenter = (
        aa * cross_bc +
        bb * cross_ca +
        cc * cross_ab
    ) / denominator;

    // Return absolute position
    double3 center = A + relative_circumcenter;
    double  radius = length(relative_circumcenter);

    return double4(center, radius);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 threadId: SV_DispatchThreadID) {
    uint tetId = threadId.x;
    if (tetId >= scene.numTets)
        return;

    float4x3 tet = scene.load_tet_vertices(tetId);

    outputSpheres[tetId] = (float4)computeCircumsphere(tet[0], tet[1], tet[2], tet[3]);
}
