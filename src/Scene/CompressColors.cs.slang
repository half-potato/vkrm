#include <Rose/Core/DxgiFormatConvert.h>

using namespace RoseEngine;

StructuredBuffer<float4> input;
RWStructuredBuffer<uint> output;
RWStructuredBuffer<uint16_t> outputW;

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 id: SV_DispatchThreadID, uniform uint size) {
    if (id.x < size) {
        float4 c = input[id.x];

        uint rgba = D3DX_FLOAT4_to_R10G10B10A2_UNORM(float4(c.bgr, c.a > 0 ? 1 : 0));

        output[id.x] = rgba;
        outputW[id.x] = (uint16_t)f32tof16(c.w);
    }
}