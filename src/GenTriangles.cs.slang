import Rose.Core.MathUtils;
import Rose.Core.Quaternion;
import Rose.Core.Indirect;
import TetrahedronScene;

ParameterBlock<TetrahedronScene> scene;
RWByteAddressBuffer triangles;
RWStructuredBuffer<VkDrawIndexedIndirectCommand> indirectArgs;

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

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 threadId: SV_DispatchThreadID) {
    uint tetId = threadId.x;
    if (tetId >= scene.numTets)
        return;

    uint4 inds = scene.indices[tetId];

    // write tris to triangles buffer

    for (uint i = 0; i < 4; i++) {
        uint3 tri = kTetTriangles[i];
        tri = uint3(inds[tri[0]], inds[tri[1]], inds[tri[2]]);
        triangles.Store3((12 * tetId + 3*i) * sizeof(uint), tri);
    }

    if (tetId == 0) {
        VkDrawIndexedIndirectCommand cmd = {};
        cmd.indexCount = 12 * scene.numTets;
        cmd.instanceCount = 1;
        cmd.firstIndex = 0;
        cmd.vertexOffset = 0;
        cmd.firstInstance = 0;
        indirectArgs[0] = cmd;
    }
}