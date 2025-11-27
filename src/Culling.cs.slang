import Scene.TetrahedronScene;
import Rose.Core.MathUtils;
import Rose.Core.Quaternion;

// Must match MeshShaderRenderer.3d.slang
#define GROUP_SIZE 32
#define TETS_PER_GROUP (GROUP_SIZE / 4)

using namespace vkDelTet;
using namespace RoseEngine;

ParameterBlock<TetrahedronScene> scene;
uniform float4x4 viewProjection;
uniform float4x4 invProjection;
uniform quat     cameraRotation;
uniform float3   rayOrigin;
uniform uint     numBlocks;
uniform float2   outputResolution;
RWByteAddressBuffer markedTets;

RWByteAddressBuffer drawArgs;   // Buffer to hold arguments for an indirect draw call
RWByteAddressBuffer insDrawArgs;   // Buffer to hold arguments for an indirect draw call
RWByteAddressBuffer meshDrawArgs;   // Buffer to hold arguments for an indirect draw call
RWByteAddressBuffer blockSumAtomicCounter;

[shader("compute")]
[numthreads(64, 1, 1)]
void markTets(uint3 threadId: SV_DispatchThreadID) {
    if (threadId.x >= scene.numTets)
        return;
    uint tetId = threadId.x;

    const float4x3 tet = scene.load_tet_vertices(tetId);

    // Project all 4 vertices into clip space
    float4 verts[4];
    verts[0] = mul(float4(tet[0], 1), transpose(viewProjection));
    verts[1] = mul(float4(tet[1], 1), transpose(viewProjection));
    verts[2] = mul(float4(tet[2], 1), transpose(viewProjection));
    verts[3] = mul(float4(tet[3], 1), transpose(viewProjection));

    // A simple frustum check requires testing all 8 corners of the AABB in clip space
    // A simpler approach is to check the AABB in NDC space.
    
    // Perspective divide to get Normalized Device Coordinates (NDC)
    verts[0] /= verts[0].w;
    verts[1] /= verts[1].w;
    verts[2] /= verts[2].w;
    verts[3] /= verts[3].w;

    // Calculate the AABB of the tetrahedron in NDC space
    float3 mn = min(min(verts[0].xyz, verts[1].xyz), min(verts[2].xyz, verts[3].xyz));
    float3 mx = max(max(verts[0].xyz, verts[1].xyz), max(verts[2].xyz, verts[3].xyz));
    const float2 extent = outputResolution * (mx - mn).xy;

    // Check if the AABB is completely outside the viewing frustum
    bool in_frustum = !(mx.x < -1.0f || mn.x > 1.0f || mx.y < -1.0f || mn.y > 1.0f || mx.z < 0.f);
    
    // Add your density check here as well
    float density = scene.load_tet_density(tetId);
    bool visible = in_frustum && !(extent.x * extent.y < 1);

    // Write 1 if visible, 0 otherwise
    markedTets.Store<uint>(tetId * sizeof(uint), visible ? 1 : 0);
}

// Use a larger group size for scans if possible
#define SCAN_GROUP_SIZE 256
groupshared uint shared_data[SCAN_GROUP_SIZE * 2];

[shader("compute")]
[numthreads(SCAN_GROUP_SIZE, 1, 1)]
void prefix_sum(uint3 groupId: SV_GroupID, uint groupThreadId: SV_GroupThreadID) {
    uint global_idx_base = (groupId.x * SCAN_GROUP_SIZE + groupThreadId) * 2;

    // 1. Load data into shared memory
    if (global_idx_base < scene.numTets) {
        shared_data[groupThreadId * 2] = markedTets.Load<uint>(global_idx_base * sizeof(uint));
    } else {
        shared_data[groupThreadId * 2] = 0;
    }
    if (global_idx_base + 1 < scene.numTets) {
        shared_data[groupThreadId * 2 + 1] = markedTets.Load<uint>((global_idx_base + 1) * sizeof(uint));
    } else {
        shared_data[groupThreadId * 2 + 1] = 0;
    }

    GroupMemoryBarrierWithGroupSync();

    // 2. Up-Sweep (Reduction) Phase: Build a sum tree in shared memory.
    uint stride = 1;
    for (uint d = (SCAN_GROUP_SIZE * 2) >> 1; d > 0; d >>= 1) {
        if (groupThreadId < d) {
            uint i = stride * (2 * groupThreadId + 1) - 1;
            uint j = stride * (2 * groupThreadId + 2) - 1;
            shared_data[j] += shared_data[i];
        }
        stride *= 2;
        GroupMemoryBarrierWithGroupSync();
    }

    // 3. Store the total sum for this block and clear the last element.
    if (groupThreadId == 0) {
        uint countInThisBlock = shared_data[SCAN_GROUP_SIZE * 2 - 1];
        uint runningTotalBeforeThis;
        blockSumAtomicCounter.InterlockedAdd(0, countInThisBlock, runningTotalBeforeThis);
        uint totalVisibleCount = runningTotalBeforeThis;
        
        drawArgs.Store<uint>(sizeof(uint)*0, totalVisibleCount * 12);
        drawArgs.Store<uint>(sizeof(uint)*1, 1);
        drawArgs.Store<uint>(sizeof(uint)*2, 0);
        drawArgs.Store<uint>(sizeof(uint)*3, 0);

        uint groupCountX = (totalVisibleCount + TETS_PER_GROUP - 1) / TETS_PER_GROUP;
        meshDrawArgs.Store<uint>(sizeof(uint)*0, groupCountX);
        meshDrawArgs.Store<uint>(sizeof(uint)*1, 1);
        meshDrawArgs.Store<uint>(sizeof(uint)*2, 1);
        uint workgroupSize = 32;

        insDrawArgs.Store<uint>(sizeof(uint)*0, 3 * 4);
        insDrawArgs.Store<uint>(sizeof(uint)*1, totalVisibleCount);
        insDrawArgs.Store<uint>(sizeof(uint)*2, 0);
        insDrawArgs.Store<uint>(sizeof(uint)*3, 0);
        insDrawArgs.Store<uint>(sizeof(uint)*4, 0);
    }
}