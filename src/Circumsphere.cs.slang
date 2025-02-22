import TetrahedronScene;
import StablePower;

ParameterBlock<TetrahedronScene> scene;
RWStructuredBuffer<float4> outputSpheres;


double DistSq(double3 p, double3 q) {
    double3 d = p - q;
    return dot(d, d);
}

// Fallback: returns center & radius of bounding sphere for points (A,B,C,D)
// by picking the farthest pair, then using midpoint and half-distance.
void boundingSphereFor4Points(
    double3 A, double3 B, double3 C, double3 D,
    out double3 center, out double radius) {
    // We'll brute force check all 6 pairs for the largest distance.
    double distAB = DistSq(A, B);
    double distAC = DistSq(A, C);
    double distAD = DistSq(A, D);
    double distBC = DistSq(B, C);
    double distBD = DistSq(B, D);
    double distCD = DistSq(C, D);

    // Track which pair is largest
    double bestDist = distAB;
    double3 bestP  = A, bestQ  = B;

    if (distAC > bestDist) { bestDist = distAC; bestP = A; bestQ = C; }
    if (distAD > bestDist) { bestDist = distAD; bestP = A; bestQ = D; }
    if (distBC > bestDist) { bestDist = distBC; bestP = B; bestQ = C; }
    if (distBD > bestDist) { bestDist = distBD; bestP = B; bestQ = D; }
    if (distCD > bestDist) { bestDist = distCD; bestP = C; bestQ = D; }

    // bestP, bestQ is the farthest pair
    center = 0.5 * (bestP + bestQ);

    radius = sqrt(bestDist) * 0.5;
}

void computeCircumsphere(
    double3 A, double3 B, double3 C, double3 D,
    out double3 center, out double radius) {
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

    if (abs(denominator) < 1e-12) {
        boundingSphereFor4Points(A, B, C, D, center, radius);
    } else {
        // Guard against division by zero
        double3 relative_circumcenter = (
            aa * cross_bc +
            bb * cross_ca +
            cc * cross_ab
        ) / denominator;

        // Return absolute position
        center = A + relative_circumcenter;
        radius = length(relative_circumcenter);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 threadId: SV_DispatchThreadID) {
    uint tetId = threadId.x;
    if (tetId >= scene.numTets)
        return;

    float4x3 tet = scene.load_tet_vertices(tetId);

    double3 center;
    double radius;
    computeCircumsphere(tet[0], tet[1], tet[2], tet[3], center, radius);

    // outputSpheres[tetId] = computeCircumsphere(tet[0], tet[1], tet[2], tet[3]);
    outputSpheres[tetId] = float4((float3)center, (float)radius);
}
